// L2 inter-node all-reduce over UCX/IB (the "remote TP" level). No MPI/PMIx — peers are
// identified by exchanged UCX worker addresses, bootstrapped over a plain TCP socket.
// Standalone validated test now; the core (tp_net_init / tp_net_allreduce_f32) is the piece
// that wires into tp_allreduce.c's L2 hook.
//
//   gcc -O3 -march=native -std=c11 tp_net_ucx.c -o tp_net_test -I$UCX_DEV/include \
//       -L$UCX/lib -lucp -lucs -Wl,-rpath,$UCX/lib
//   # node 124 (server):  UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,sm,self ./tp_net_test server 0.0.0.0   13400
//   # node 121 (client):  UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,sm,self ./tp_net_test client 172.16.0.124 13400
#define _GNU_SOURCE
#include <ucp/api/ucp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define TAG 0x1234

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

// ---- tiny TCP bootstrap: exchange a blob both ways over one connection ----
static int tcp_server(const char*bind_ip, int port){
    int s=socket(AF_INET,SOCK_STREAM,0); int one=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(port); a.sin_addr.s_addr=inet_addr(bind_ip);
    if(bind(s,(void*)&a,sizeof a)){perror("bind");exit(1);} listen(s,1);
    int c=accept(s,0,0); close(s); return c;
}
static int tcp_client(const char*ip,int port){
    int s=socket(AF_INET,SOCK_STREAM,0); struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(port); a.sin_addr.s_addr=inet_addr(ip);
    while(connect(s,(void*)&a,sizeof a)){ usleep(100000); } return s;
}
static void xchg(int fd, const void*out, size_t outlen, void**in, size_t*inlen){
    uint64_t l=outlen; write(fd,&l,8); write(fd,out,outlen);
    uint64_t rl; read(fd,&rl,8); *in=malloc(rl); size_t got=0; while(got<rl){ ssize_t n=read(fd,(char*)*in+got,rl-got); if(n<=0)break; got+=n; } *inlen=rl;
}

static ucp_worker_h worker; static ucp_ep_h ep;

static void wait_req(void*req){
    if(req==NULL) return;
    if(UCS_PTR_IS_ERR(req)){ fprintf(stderr,"ucx req error: %s\n", ucs_status_string(UCS_PTR_STATUS(req))); exit(1); }
    while(ucp_request_check_status(req)==UCS_INPROGRESS) ucp_worker_progress(worker);
    ucp_request_free(req);
}

// 2-node all-reduce (SUM): exchange buffers with peer, add. sym both sides.
static void net_allreduce(float*buf, float*tmp, size_t n){
    ucp_request_param_t p; p.op_attr_mask=0;
    void*sreq=ucp_tag_send_nbx(ep, buf, n*sizeof(float), TAG, &p);
    void*rreq=ucp_tag_recv_nbx(worker, tmp, n*sizeof(float), TAG, (ucp_tag_t)-1, &p);
    wait_req(sreq); wait_req(rreq);
    for(size_t i=0;i<n;i++) buf[i]+=tmp[i];
}

int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: %s server|client <ip> <port>\n",argv[0]); return 1; }
    int is_server = strcmp(argv[1],"server")==0;
    const char*ip=argv[2]; int port=atoi(argv[3]);

    // UCX init
    ucp_config_t*cfg; ucp_config_read(NULL,NULL,&cfg);
    ucp_params_t up; up.field_mask=UCP_PARAM_FIELD_FEATURES; up.features=UCP_FEATURE_TAG;
    ucp_context_h ctx; if(ucp_init(&up,cfg,&ctx)!=UCS_OK){fprintf(stderr,"ucp_init fail\n");return 1;} ucp_config_release(cfg);
    ucp_worker_params_t wp; wp.field_mask=UCP_WORKER_PARAM_FIELD_THREAD_MODE; wp.thread_mode=UCS_THREAD_MODE_SINGLE;
    ucp_worker_create(ctx,&wp,&worker);
    ucp_address_t*myaddr; size_t myaddr_len; ucp_worker_get_address(worker,&myaddr,&myaddr_len);

    // TCP bootstrap + address exchange
    int fd = is_server ? tcp_server(ip,port) : tcp_client(ip,port);
    void*peeraddr; size_t peeraddr_len;
    xchg(fd, myaddr, myaddr_len, &peeraddr, &peeraddr_len);

    ucp_ep_params_t ep_p; ep_p.field_mask=UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; ep_p.address=(ucp_address_t*)peeraddr;
    if(ucp_ep_create(worker,&ep_p,&ep)!=UCS_OK){fprintf(stderr,"ep_create fail\n");return 1;}

    int rank = is_server ? 1 : 0;   // server=node1, client=node0
    size_t sizes[]={256,1024,4096,8192,16384,32768,65536}; int ns=7;
    if(rank==0){ printf("# UCX/IB 2-node all-reduce\n# %-10s %-12s %-10s\n","floats","lat_us","verify"); }
    for(int s=0;s<ns;s++){
        size_t n=sizes[s]; float*buf=malloc(n*4),*tmp=malloc(n*4);
        for(size_t i=0;i<n;i++) buf[i]=(float)(rank+1);
        // warmup + correctness (buf becomes 1+2=3)
        net_allreduce(buf,tmp,n);
        int ok = (buf[0]==3.0f && buf[n-1]==3.0f);
        // timed (reset each iter so values stay bounded: re-set then reduce)
        int iters=2000; double t0=now_us();
        for(int it=0;it<iters;it++){ for(size_t i=0;i<n;i++) buf[i]=(float)(rank+1); net_allreduce(buf,tmp,n); }
        double lat=(now_us()-t0)/iters;
        if(rank==0) printf("  %-10zu %-12.3f %s\n", n, lat, ok?"OK":"BAD");
        free(buf);free(tmp);
    }
    // tear down (best-effort)
    void*creq=ucp_ep_close_nbx(ep, &(ucp_request_param_t){.op_attr_mask=0}); if(creq && !UCS_PTR_IS_ERR(creq)) wait_req(creq);
    ucp_worker_release_address(worker,myaddr); ucp_worker_destroy(worker); ucp_cleanup(ctx); close(fd);
    return 0;
}
