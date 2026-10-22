/* Stage 2: Restores raw P,C,D from v0.10 O.dvg into the same divERGe binary.
 * Recreates geometry from a fixture exported from the original T.dvg.
 * Calls divERGe's *own* crossing projectors (p,m,d), and writes only Gamma/M
 * native matrices to a small directory.  NO FRG FLOW is run.
 *
 * Build: gcc -O2 -std=c11 -fopenmp -DCTYPESGEN=1 -I<source>/src \
 *   restore_full_projected.c -L<package> -Wl,-rpath,<package> -ldivERGe -lm -o restore_full_projected
 */
#define CTYPESGEN 1
#include "diverge_model.h"
#include "diverge_model_internals.h"
#include "diverge_flow_step.h"
#include "diverge_internals_struct.h"
#include "tu/diverge_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>

extern void diverge_init(int*,char***);
extern void diverge_finalize(void);
#define MUST(expr,msg) do {if(!(expr)){fprintf(stderr,"ERROR: %s at %s:%d\n", msg,__FILE__,__LINE__);exit(2);}}while(0)
static void rd(FILE *f,void *buf,size_t size,size_t n){MUST(fread(buf,size,n,f)==n,"short read");}
static void sk(FILE*f,int64_t offset){MUST(fseeko(f,(off_t)offset,SEEK_SET)==0,"seek failed");}

static diverge_model_t *read_fixture(char *path,int *norb_out,int *nk_out){
    FILE*f=fopen(path,"r");MUST(f,"open model fixture");
    int nk,nkf,ns;MUST(fscanf(f,"%d %d %d",&nk,&nkf,&ns)==3,"fixture header");
    MUST(ns==2 && nk>0 && nkf>0,"requires spinful/valid mesh");
    diverge_model_t*m=diverge_model_init();MUST(m,"model init");
    snprintf(m->name,sizeof(m->name),"stage2_archived_model_restore");
    m->n_orb=3;m->n_spin=ns;m->SU2=0;
    m->nk[0]=m->nk[1]=nk;m->nk[2]=1;
    m->nkf[0]=m->nkf[1]=nkf;m->nkf[2]=1;
    for(int i=0;i<3;i++)for(int j=0;j<3;j++) MUST(fscanf(f,"%lf",&m->lattice[i][j])==1,"lattice");
    for(int i=0;i<3;i++)for(int j=0;j<3;j++) MUST(fscanf(f,"%lf",&m->positions[i][j])==1,"positions");
    int n;MUST(fscanf(f,"%d",&n)==1 && n>=0,"n_hop");
    m->n_hop=n;m->hop=calloc((size_t)n,sizeof(rs_hopping_t));MUST(m->hop,"hops allocation");
    for(int i=0;i<n;i++){
        rs_hopping_t *h=m->hop+i;
        MUST(fscanf(f,"%ld %ld %ld %ld %ld %ld %ld %lf %lf",
            (long*)&h->R[0],(long*)&h->R[1],(long*)&h->R[2],
            (long*)&h->o1,(long*)&h->o2,(long*)&h->s1,(long*)&h->s2,
            &h->t.x,&h->t.y)==9,"hopping line");
    }
    MUST(fscanf(f,"%d",&n)==1 && n>=0,"n_vert");
    m->n_vert=n;m->vert=calloc((size_t)n,sizeof(rs_vertex_t));MUST(m->vert,"vertex allocation");
    for(int i=0;i<n;i++){
        rs_vertex_t*v=m->vert+i;
        MUST(fscanf(f," %c %ld %ld %ld %ld %ld %ld %ld %ld %ld %lf %lf",
          &v->chan,(long*)&v->R[0],(long*)&v->R[1],(long*)&v->R[2],
          (long*)&v->o1,(long*)&v->o2,(long*)&v->s1,(long*)&v->s2,
          (long*)&v->s3,(long*)&v->s4,&v->V.x,&v->V.y)==12,"interaction line");
    }
    fclose(f);
    int err=diverge_model_validate(m);MUST(err==0,"model validation failed");
    diverge_model_internals_common(m);
    *norb_out=3;*nk_out=nk;
    return m;
}

static void check_geometry(FILE *f,const int64_t*h,diverge_model_t*m){
    const tu_data_t*tu=(const tu_data_t*)m->internals->tu_data;
    size_t nf=(size_t)m->n_tu_ff;
    int64_t *arr=malloc(sizeof(int64_t)*nf*3);MUST(arr,"geometry buffer allocation");
    long bad=0;
    sk(f,h[10]);rd(f,arr,sizeof(int64_t),nf);
    for(size_t i=0;i<nf;i++)bad+=(arr[i]!=tu->mi_to_ofrom[i]);
    sk(f,h[12]);rd(f,arr,sizeof(int64_t),nf);
    for(size_t i=0;i<nf;i++)bad+=(arr[i]!=tu->mi_to_oto[i]);
    sk(f,h[14]);rd(f,arr,sizeof(int64_t),3*nf);
    for(size_t i=0;i<3*nf;i++)bad+=(arr[i]!=tu->mi_to_R[i]);
    free(arr);
    printf("GEOMETRY mismatches %ld (all %zu FF-triplets tested)\n",bad,nf);fflush(stdout);
    MUST(bad==0,"MISMATCH: T reconstructed basis is not identical to O geometry");
}

static size_t native_index(int q,int s4,int s3,int f2,int s2,int s1,int f1,int ns,int nf){
    return ((((((size_t)q*ns+s4)*ns+s3)*nf+f2)*ns+s2)*ns+s1)*nf+f1;
}
static size_t saved_index(int a,int b,int f2,int d,int e,int f1,int q,int ns,int nf,int nq){
    return ((((((size_t)a*ns+b)*nf+f2)*ns+d)*ns+e)*nf+f1)*nq+q;
}

static void reconstruct_from_O(FILE*f,const int64_t*h,diverge_flow_step_t*step,int nf,int nq,int ns){
    const int offsets[3]={84,86,88};const char names[4]="PCD";
    size_t total=(size_t)nq*nf*nf*ns*ns*ns*ns;
    complex128_t *saved=malloc(sizeof(complex128_t)*total);MUST(saved,"allocate input raw channel, increase --mem if needed");
    for(int c=0;c<3;c++){
        diverge_flow_step_vertex_t v=diverge_flow_step_vertex(step,names[c]);
        MUST(v.ary && v.n_orbff==nf && v.q_0==0 && v.q_1==nq,"native buffer dimensions wrong");
        sk(f,h[offsets[c]]);rd(f,saved,sizeof(complex128_t),total);
        double maxabs=0;
        // Native (q,s4,s3,f2,s2,s1,f1); stored (spin permutations,f2,...,f1,q)
        for(int q=0;q<nq;q++)for(int s4=0;s4<ns;s4++)for(int s3=0;s3<ns;s3++)
        for(int f2=0;f2<nf;f2++)for(int s2=0;s2<ns;s2++)for(int s1=0;s1<ns;s1++)for(int f1=0;f1<nf;f1++){
            int a=s4,b=s3,d=s2,e=s1;
            if(c==1){a=s2;b=s3;d=s4;e=s1;}
            if(c==2){a=s3;b=s2;d=s4;e=s1;}
            size_t ni=native_index(q,s4,s3,f2,s2,s1,f1,ns,nf);
            size_t si=saved_index(a,b,f2,d,e,f1,q,ns,nf,nq);
            v.ary[ni]=saved[si];
            double mag=hypot(v.ary[ni].x,v.ary[ni].y);
            if(mag>maxabs)maxabs=mag;
        }
        printf("RESTORED %c, entries=%zu, max|value|=%.12g\n",names[c],total,maxabs);fflush(stdout);
    }
    free(saved);
}

static void dump_matrix(char channel,diverge_flow_step_t*step,int q,int nf,int ns,char*prefix){
    diverge_flow_step_vertex_t v=diverge_flow_step_vertex(step,channel);
    MUST(v.ary && v.n_orbff==nf,"projection missing");
    int qn=v.q_1-v.q_0;MUST(q>=0 && q<qn,"selected q out of range");
    size_t nmat=(size_t)nf*nf*ns*ns*ns*ns;
    char fname[1024];snprintf(fname,sizeof(fname),"%s_%c_q%d.bin",prefix,channel,q);
    FILE*out=fopen(fname,"wb");MUST(out,"open output projected matrix");
    MUST(fwrite(v.ary+(size_t)q*nmat,sizeof(complex128_t),nmat,out)==nmat,"write projected matrix");
    fclose(out);
    printf("OUTPUT %s bytes=%zu\n",fname,16*nmat);fflush(stdout);
}

int main(int argc,char**argv){
    if(argc!=6){fprintf(stderr,"Usage: %s <model_fixture.txt> <original_O.dvg> <tu_distance> <out_prefix> <q_csv>\n",argv[0]);return 1;}
    const char*fixture=argv[1],*O=argv[2];double dist=atof(argv[3]);char*prefix=argv[4];
    int nq_check=0,norb=0,nk=0;
    diverge_init(&argc,&argv);
    diverge_model_t*m=read_fixture((char*)fixture,&norb,&nk);
    diverge_model_internals_tu(m,dist);
    int nf=(int)m->n_tu_ff;
    printf("REBUILT nff=%d nk=%d dist=%.9g\n",nf,nk,dist);fflush(stdout);
    FILE*f=fopen(O,"rb");MUST(f,"cannot open O.dvg");
    int64_t h[256];rd(f,h,sizeof(int64_t),256);
    MUST(h[1]==norb && h[2]==2 && h[6]==0,"wrong O spin/orbital model");
    MUST(h[7]==nf && h[3]==nk*nk && h[5]==nk*nk && h[9]==0,"wrong O geometry or q symmetry reduction");
    int ns=2,nq=(int)h[3];size_t nn=(size_t)nq*nf*nf*ns*ns*ns*ns;
    const int offs[5]={84,86,88,90,92};
    int64_t size=0;sk(f,0);MUST(fseeko(f,0,SEEK_END)==0,"file seek end");size=ftello(f);
    for(int j=0;j<5;j++) MUST(h[offs[j]+1]==(int64_t)nn && h[offs[j]]>=2048 && h[offs[j]]+(int64_t)nn*16<=size,"truncated O channel/loop");
    check_geometry(f,h,m);
    diverge_flow_step_t*step=diverge_flow_step_init(m,"tu","PCD");MUST(step,"flow step init failed");
    reconstruct_from_O(f,h,step,nf,nq,ns);
    fclose(f);
    char *qcsv=strdup(argv[5]);MUST(qcsv,"csv allocation");
    char *tok=strtok(qcsv,",");int qq[10],nqsel=0;
    while(tok){MUST(nqsel<10,"too many q values");long q=strtol(tok,NULL,10);MUST(q>=0 && q<nq,"q outside mesh");qq[nqsel++]=(int)q;tok=strtok(NULL,",");}
    MUST(nqsel>0,"empty q selection");
    for(int c=0;c<3;c++)for(int qi=0;qi<nqsel;qi++) dump_matrix("pdm"[c],step,qq[qi],nf,ns,prefix);
    free(qcsv);diverge_flow_step_free(step);diverge_model_free(m);diverge_finalize();
    puts("DONE: source-projected native matrices exported; original O.dvg remains unchanged");
    return 0;
}
