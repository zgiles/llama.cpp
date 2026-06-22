// L2 inter-node transport over UCX/IB (no MPI). 2-node exchange-sum all-reduce.
// Bootstrap: plain TCP on the IB subnet swaps UCX worker addresses; data path native IB RC.
#define _GNU_SOURCE
#include "tp_net.h"
#include <ucp/api/ucp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define TP_NET_TAG 0x7A1C
#define TP_NET_TMP_FLOATS (1u << 20)   // 4 MB scratch; max all-reduce length

struct tp_net {
    ucp_context_h ctx;
    ucp_worker_h  worker;
    ucp_ep_h      ep;
    float *       tmp;
    int           node_rank;
    int           n_nodes;
};

static int tcp_server(const char * bind_ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port);
    a.sin_addr.s_addr = bind_ip ? inet_addr(bind_ip) : INADDR_ANY;
    if (bind(s, (void *)&a, sizeof a)) { perror("bind"); return -1; }
    listen(s, 1); int c = accept(s, 0, 0); close(s); return c;
}
static int tcp_client(const char * ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = inet_addr(ip);
    for (int tries = 0; connect(s, (void *)&a, sizeof a); tries++) { if (tries > 600) return -1; usleep(100000); }
    return s;
}
static void xchg(int fd, const void * out, size_t outlen, void ** in, size_t * inlen) {
    uint64_t l = outlen; (void)!write(fd, &l, 8); (void)!write(fd, out, outlen);
    uint64_t rl; (void)!read(fd, &rl, 8); *in = malloc(rl);
    size_t got = 0; while (got < rl) { ssize_t n = read(fd, (char *)*in + got, rl - got); if (n <= 0) break; got += n; }
    *inlen = rl;
}

static void wait_req(ucp_worker_h w, void * req) {
    if (req == NULL) return;
    if (UCS_PTR_IS_ERR(req)) { fprintf(stderr, "tp_net req error: %s\n", ucs_status_string(UCS_PTR_STATUS(req))); abort(); }
    while (ucp_request_check_status(req) == UCS_INPROGRESS) ucp_worker_progress(w);
    ucp_request_free(req);
}

// in-place SUM of `buf` across nodes (2-node exchange-sum over IB RC).
void tp_net_allreduce(tp_net * net, float * buf, size_t count) {
    if (count > TP_NET_TMP_FLOATS) { fprintf(stderr, "tp_net: count %zu exceeds tmp\n", count); abort(); }
    ucp_request_param_t p; p.op_attr_mask = 0;
    void * sreq = ucp_tag_send_nbx(net->ep, buf, count * sizeof(float), TP_NET_TAG, &p);
    void * rreq = ucp_tag_recv_nbx(net->worker, net->tmp, count * sizeof(float), TP_NET_TAG, (ucp_tag_t)-1, &p);
    wait_req(net->worker, sreq);
    wait_req(net->worker, rreq);
    for (size_t i = 0; i < count; i++) buf[i] += net->tmp[i];
}

// function-pointer target on tp_comm (void* signature for the decoupled L2 hook)
static void tp_net_allreduce_static(void * vnet, float * buf, size_t count) {
    tp_net_allreduce((tp_net *)vnet, buf, count);
}

tp_net * tp_net_init(int n_nodes, int node_rank, const char * server_ip, int port) {
    if (n_nodes != 2) { fprintf(stderr, "tp_net: only n_nodes=2 supported for now\n"); return NULL; }
    tp_net * net = calloc(1, sizeof(tp_net));
    net->n_nodes = n_nodes; net->node_rank = node_rank;
    net->tmp = aligned_alloc(64, TP_NET_TMP_FLOATS * sizeof(float));

    ucp_config_t * cfg; ucp_config_read(NULL, NULL, &cfg);
    ucp_params_t up; up.field_mask = UCP_PARAM_FIELD_FEATURES; up.features = UCP_FEATURE_TAG;
    if (ucp_init(&up, cfg, &net->ctx) != UCS_OK) { fprintf(stderr, "ucp_init fail\n"); ucp_config_release(cfg); return NULL; }
    ucp_config_release(cfg);
    ucp_worker_params_t wp; wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE; wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucp_worker_create(net->ctx, &wp, &net->worker);
    ucp_address_t * myaddr; size_t myaddr_len; ucp_worker_get_address(net->worker, &myaddr, &myaddr_len);

    // rank 1 is the server (binds), rank 0 connects.
    int fd = (node_rank == 1) ? tcp_server(NULL, port) : tcp_client(server_ip, port);
    if (fd < 0) { fprintf(stderr, "tp_net bootstrap fail\n"); return NULL; }
    void * peeraddr; size_t peeraddr_len;
    xchg(fd, myaddr, myaddr_len, &peeraddr, &peeraddr_len);
    close(fd);

    ucp_ep_params_t ep_p; ep_p.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; ep_p.address = (ucp_address_t *)peeraddr;
    if (ucp_ep_create(net->worker, &ep_p, &net->ep) != UCS_OK) { fprintf(stderr, "ep_create fail\n"); return NULL; }
    free(peeraddr);
    ucp_worker_release_address(net->worker, myaddr);
    return net;
}

void tp_net_attach(tp_net * net, tp_comm * c) {
    c->net           = net;
    c->net_allreduce = tp_net_allreduce_static;
    c->n_nodes       = net->n_nodes;
    c->node_rank     = net->node_rank;
}

void tp_net_finalize(tp_net * net) {
    if (!net) return;
    ucp_request_param_t p; p.op_attr_mask = 0;
    void * creq = ucp_ep_close_nbx(net->ep, &p);
    if (creq && !UCS_PTR_IS_ERR(creq)) wait_req(net->worker, creq);
    ucp_worker_destroy(net->worker);
    ucp_cleanup(net->ctx);
    free(net->tmp); free(net);
}
