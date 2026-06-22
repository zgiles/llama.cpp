// Full hierarchical all-reduce: 2 nodes x 2 sockets, exercising L1 (in-process shmem) + L2
// (UCX/IB) through the single tp_allreduce_f32 API. Validates the GLOBAL sum and times it.
//
//   gcc -O3 -march=native -std=c11 -pthread tp_full_test.c tp_allreduce.c tp_net.c \
//       -o tp_full_test -I$UCXDEV/include -L$UCX/lib -lucp -lucs -Wl,-rpath,$UCX/lib
//   # node 124 (rank1/server): ... ./tp_full_test server 0.0.0.0      13401 [K]
//   # node 121 (rank0/client): ... ./tp_full_test client 172.16.0.124 13401 [K]
#define _GNU_SOURCE
#include "tp_allreduce.h"
#include "tp_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#define MAXN  (32768)
#define NSIZE 5
static const size_t SIZES[NSIZE] = {1024, 4096, 8192, 16384, 32768};
#define ITERS 5000

static tp_comm  g_comm;
static tp_net * g_net;
static float  * g_scratch;
static int      g_nsock, g_K, g_nth, g_node_rank, g_port;
static const char * g_server_ip;
static int      g_cpu[TP_MAX_LOCAL][512], g_ncpu[TP_MAX_LOCAL];
static float  * g_send[TP_MAX_LOCAL], * g_recv[TP_MAX_LOCAL];
static double   g_lat_us[TP_MAX_LOCAL * 512][NSIZE];
static int      g_ok[TP_MAX_LOCAL * 512];

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void parse_cpulist(const char*s,int*arr,int*n){ int k=0; while(*s&&*s!='\n'){int a=0,b; while(*s>='0'&&*s<='9'){a=a*10+(*s-'0');s++;} b=a; if(*s=='-'){s++;b=0;while(*s>='0'&&*s<='9'){b=b*10+(*s-'0');s++;}} for(int c=a;c<=b&&k<512;c++)arr[k++]=c; if(*s==',')s++;} *n=k; }
static int is_primary_core(int cpu){ char p[160]; snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",cpu); FILE*f=fopen(p,"r"); if(!f)return 1; char b[256]={0}; int first=-1; if(fgets(b,sizeof b,f))sscanf(b,"%d",&first); fclose(f); return first<0||first==cpu; }

typedef struct { int sid, lid, ith; } targ;

static void * worker(void * p) {
    targ * t = (targ *)p; int sid=t->sid, lid=t->lid, ith=t->ith;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(g_cpu[sid][lid % g_ncpu[sid]], &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set); sched_yield();
    tp_sense st = {0,0};
    float * send = g_send[sid], * recv = g_recv[sid];
    size_t b = MAXN/g_K, lo=(size_t)lid*b, ll=(lid==g_K-1)?(MAXN-lo):b;
    for (size_t i=lo;i<lo+ll;i++){ send[i]=0; recv[i]=0; }
    { size_t bb=MAXN/g_nth,o=(size_t)ith*bb,l=(ith==g_nth-1)?(MAXN-o):bb; for(size_t i=o;i<o+l;i++) g_scratch[i]=0; }

    // ith==0 brings up the inter-node transport, then attaches it (sets n_nodes=2). The
    // post-init barrier publishes the attach to every thread before any all-reduce runs.
    if (ith==0) {
        g_net = tp_net_init(2, g_node_rank, g_server_ip, g_port);
        if (!g_net){ fprintf(stderr,"tp_net_init failed\n"); exit(1); }
        tp_net_attach(g_net, &g_comm);
    }
    tp_hier_barrier(&g_comm, sid, lid, &st);

    // expected GLOBAL sum: over nodes r in {0,1}, sockets s, value (r*10 + s+1)
    float expect = 0; for(int r=0;r<2;r++) for(int s=0;s<g_nsock;s++) expect += (float)(r*10 + s + 1);
    float myval = (float)(g_node_rank*10 + sid + 1);

    for (int s=0;s<NSIZE;s++){
        size_t n=SIZES[s];
        for (size_t i=0;i<n;i++) send[i]=myval;
        tp_allreduce_f32(&g_comm, sid, lid, ith, &st, send, recv, n);
        if (s==NSIZE-1){ int ok=1; for(size_t i=0;i<n;i++) if(recv[i]!=expect){ok=0;break;} g_ok[ith]=ok; }
        tp_hier_barrier(&g_comm, sid, lid, &st);
        double t0=now_us();
        for(int it=0;it<ITERS;it++){ for(size_t i=0;i<n;i++) send[i]=myval; tp_allreduce_f32(&g_comm,sid,lid,ith,&st,send,recv,n); }
        g_lat_us[ith][s]=(now_us()-t0)/ITERS;
        tp_hier_barrier(&g_comm, sid, lid, &st);
    }
    if (ith==0) tp_net_finalize(g_net);
    return NULL;
}

int main(int argc, char ** argv) {
    if (argc < 4){ fprintf(stderr,"usage: %s server|client <ip> <port> [K]\n",argv[0]); return 1; }
    g_node_rank = strcmp(argv[1],"server")==0 ? 1 : 0;
    g_server_ip = argv[2]; g_port = atoi(argv[3]);

    for (int i=0;i<TP_MAX_LOCAL;i++){ char path[128]; snprintf(path,sizeof path,"/sys/devices/system/node/node%d/cpulist",i); FILE*f=fopen(path,"r"); if(!f)break; char buf[512]; if(fgets(buf,sizeof buf,f)) parse_cpulist(buf,g_cpu[g_nsock],&g_ncpu[g_nsock]),g_nsock++; fclose(f); }
    if (g_nsock==0){ g_nsock=1; g_ncpu[0]=8; for(int c=0;c<8;c++) g_cpu[0][c]=c; }
    for (int s=0;s<g_nsock;s++){ int w=0; for(int i=0;i<g_ncpu[s];i++) if(is_primary_core(g_cpu[s][i])) g_cpu[s][w++]=g_cpu[s][i]; g_ncpu[s]=w; }
    g_K = (argc>4)?atoi(argv[4]):g_ncpu[0]; if(g_K<1)g_K=1; g_nth=g_nsock*g_K;

    g_scratch = aligned_alloc(64, MAXN*sizeof(float));
    for (int s=0;s<g_nsock;s++){ g_send[s]=aligned_alloc(64,MAXN*sizeof(float)); g_recv[s]=aligned_alloc(64,MAXN*sizeof(float)); }
    tp_comm_init_local(&g_comm, g_nsock, g_K, g_scratch, MAXN);

    pthread_t th[TP_MAX_LOCAL*512]; targ ta[TP_MAX_LOCAL*512]; int idx=0;
    for (int s=0;s<g_nsock;s++) for(int l=0;l<g_K;l++){ ta[idx].sid=s; ta[idx].lid=l; ta[idx].ith=idx; pthread_create(&th[idx],NULL,worker,&ta[idx]); idx++; }
    for (int i=0;i<g_nth;i++) pthread_join(th[i],NULL);

    if (g_node_rank==0){
        int all_ok=1; for(int i=0;i<g_nth;i++) all_ok&=g_ok[i];
        printf("# full L1+L2 hierarchical all-reduce: 2 nodes x %d sockets x %d threads/socket\n", g_nsock, g_K);
        printf("# correctness (global sum): %s\n", all_ok?"PASS":"FAIL");
        printf("# %-8s %-10s %-12s\n","floats","bytes","lat_us");
        for(int s=0;s<NSIZE;s++){ double lat=0; for(int i=0;i<g_nth;i++) if(g_lat_us[i][s]>lat)lat=g_lat_us[i][s]; printf("  %-8zu %-10zu %-12.3f\n",SIZES[s],SIZES[s]*4,lat); }
    }
    return 0;
}
