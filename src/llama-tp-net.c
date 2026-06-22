#include "llama-tp-net.h"
#include <stdlib.h>

int llama_tp_enabled(void) {
    const char * s = getenv("LLAMA_TP_SIZE");
    return s && atoi(s) > 1;
}

#ifdef LLAMA_TP_UCX

#define _GNU_SOURCE
#include <ucp/api/ucp.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define TP_TAG 0x7A1C
#define TP_TMP_FLOATS (1u << 20)   // 4 MB scratch (max all-reduce length)

struct tp_net {
    ucp_context_h ctx;
    ucp_worker_h  worker;
    ucp_ep_h      ep;
    float *       tmp;
};

static int tcp_server(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(s, (void *)&a, sizeof a)) { perror("tp bind"); return -1; }
    listen(s, 1); int c = accept(s, 0, 0); close(s); return c;
}
static int tcp_client(const char * ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = inet_addr(ip);
    for (int t = 0; connect(s, (void *)&a, sizeof a); t++) { if (t > 1200) return -1; usleep(100000); }
    return s;
}
static void xchg(int fd, const void * out, size_t outlen, void ** in, size_t * inlen) {
    uint64_t l = outlen; if (write(fd, &l, 8) < 0 || write(fd, out, outlen) < 0) {}
    uint64_t rl = 0; if (read(fd, &rl, 8) < 0) {}
    *in = malloc(rl); size_t got = 0;
    while (got < rl) { ssize_t n = read(fd, (char *)*in + got, rl - got); if (n <= 0) break; got += n; }
    *inlen = rl;
}
static void wait_req(ucp_worker_h w, void * req) {
    if (req == NULL) return;
    if (UCS_PTR_IS_ERR(req)) { fprintf(stderr, "tp req error\n"); abort(); }
    while (ucp_request_check_status(req) == UCS_INPROGRESS) ucp_worker_progress(w);
    ucp_request_free(req);
}

static struct tp_net * tp_net_init(int rank, const char * server_ip, int port) {
    struct tp_net * net = calloc(1, sizeof(struct tp_net));
    net->tmp = aligned_alloc(64, TP_TMP_FLOATS * sizeof(float));
    ucp_config_t * cfg; ucp_config_read(NULL, NULL, &cfg);
    ucp_params_t up; up.field_mask = UCP_PARAM_FIELD_FEATURES; up.features = UCP_FEATURE_TAG;
    if (ucp_init(&up, cfg, &net->ctx) != UCS_OK) { fprintf(stderr, "tp ucp_init fail\n"); return NULL; }
    ucp_config_release(cfg);
    ucp_worker_params_t wp; wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE; wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucp_worker_create(net->ctx, &wp, &net->worker);
    ucp_address_t * myaddr; size_t myaddr_len; ucp_worker_get_address(net->worker, &myaddr, &myaddr_len);
    int fd = (rank == 1) ? tcp_server(port) : tcp_client(server_ip, port);
    if (fd < 0) { fprintf(stderr, "tp bootstrap fail\n"); return NULL; }
    void * peeraddr; size_t peeraddr_len; xchg(fd, myaddr, myaddr_len, &peeraddr, &peeraddr_len); close(fd);
    ucp_ep_params_t ep_p; ep_p.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; ep_p.address = (ucp_address_t *)peeraddr;
    if (ucp_ep_create(net->worker, &ep_p, &net->ep) != UCS_OK) { fprintf(stderr, "tp ep fail\n"); return NULL; }
    free(peeraddr); ucp_worker_release_address(net->worker, myaddr);
    fprintf(stderr, "llama-tp: inter-node transport up (rank %d)\n", rank);
    return net;
}

static struct tp_net * g_net = NULL;
static int g_init_done = 0;

void llama_tp_allreduce_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                           int ith, int nth, void * userdata) {
    (void)a; (void)nth; (void)userdata;
    if (ith != 0) return;                       // single thread drives the UCX worker
    if (!g_init_done) {                          // lazy bootstrap on the worker thread, first call
        g_init_done = 1;
        const char * sz = getenv("LLAMA_TP_SIZE");
        if (sz && atoi(sz) > 1) {
            int rank = getenv("LLAMA_TP_RANK") ? atoi(getenv("LLAMA_TP_RANK")) : 0;
            int port = getenv("LLAMA_TP_PORT") ? atoi(getenv("LLAMA_TP_PORT")) : 13700;
            g_net = tp_net_init(rank, getenv("LLAMA_TP_PEER"), port);
        }
    }
    if (!g_net) return;
    float * buf = (float *) dst->data;
    size_t n = (size_t) ggml_nelements(dst);
    if (n > TP_TMP_FLOATS) { fprintf(stderr, "tp: count %zu too big\n", n); abort(); }
    ucp_request_param_t p; p.op_attr_mask = 0;
    void * sreq = ucp_tag_send_nbx(g_net->ep, buf, n * sizeof(float), TP_TAG, &p);
    void * rreq = ucp_tag_recv_nbx(g_net->worker, g_net->tmp, n * sizeof(float), TP_TAG, (ucp_tag_t)-1, &p);
    wait_req(g_net->worker, sreq);
    wait_req(g_net->worker, rreq);
    for (size_t i = 0; i < n; i++) buf[i] += g_net->tmp[i];
}

#else  // no UCX: op is a no-op

void llama_tp_allreduce_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                           int ith, int nth, void * userdata) {
    (void)dst; (void)a; (void)ith; (void)nth; (void)userdata;
}

#endif // LLAMA_TP_UCX
