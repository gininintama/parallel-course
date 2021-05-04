#include <iostream>
#include <stdlib.h>
#include <windows.h>


#include <immintrin.h>

using namespace std;

#define N 50

long long head,tail,freq;
__m256d t1,t2,t3,t4;
double** SerialLU(double** A){
    for(int k=0;k<N;k++)
    {
        double temp=A[k][k];
        for(int j=k;j<N;j++)
        {
            A[k][j]/=temp;
        }
        A[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            double temp2=A[i][k];
            for(int j=k+1;j<N;j++)
            {
                A[i][j]-=temp2*A[k][j];
            }
            A[i][k]=0;
        }
    }
    return A;
}
double** avxLU(double** A){
    int n1=N/4;
    int n2=N%4;
    for(int k=0;k<N;k++)
    {
        //double temp[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
        t1=_mm256_set_pd(A[k][k],A[k][k],A[k][k],A[k][k]);
        //t1[0]=t1[1]=t1[2]=t1[3]=A[k][k];
        int t=A[k][k];int j;
        t1=_mm256_set_pd(t,t,t,t);
        for(j=k;j<N-n2;j+=4)
        {
           // A[k][j]/=temp;
           t2=_mm256_set_pd(A[k][j],A[k][j+1],A[k][j+2],A[k][j+3]);
           t3=_mm256_div_pd(t2,t1);

           //_mm256_storeu_pd(A[k]+j,t3);
           A[k][j+3]=t3[0];
           A[k][j+2]=t3[1];
           A[k][j+1]=t3[2];
           A[k][j]=t3[3];
        }
        for(;j%4!=n2;j++)
                A[k][j]/=t;
        A[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            //double temp2=A[i][k];
            t1=_mm256_set_pd(A[i][k],A[i][k],A[i][k],A[i][k]);
            //t1[0]=t1[1]=t1[2]=t1[3]=A[i][k];
            int t=A[i][k];int j;
            t1=_mm256_set_pd(t,t,t,t);
            for(j=k+1;j<N-n2;j+=4)
            {
                //A[i][j]-=temp2*A[k][j];
                t2=_mm256_set_pd(A[k][j],A[k][j+1],A[k][j+2],A[k][j+3]);
                t4=_mm256_set_pd(A[i][j],A[i][j+1],A[i][j+2],A[i][j+3]);
                t3=_mm256_mul_pd(t1,t2);
                __m256d t5=_mm256_sub_pd(t4,t3);


                A[i][j+3]=t5[0];
                A[i][j+2]=t5[1];
                A[i][j+1]=t5[2];
                A[i][j]=t5[3];
            }
            for(;j%4!=n2;j++)
                 A[i][j]-=t*A[k][j];
            A[i][k]=0;
        }
    }
    return A;
}
double** sseLU(double** A){
    int n1=N/4;
    int n2=N%4;
    for(int k=0;k<N;k++)
    {
        //double temp[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
        t1=_mm256_set_pd(A[k][k],A[k][k],A[k][k],A[k][k]);
        //t1[0]=t1[1]=t1[2]=t1[3]=A[k][k];
        int t=A[k][k];int j;
        t1=_mm256_set_pd(t,t,t,t);
        for(j=k;j<N-n2;j+=4)
        {
           // A[k][j]/=temp;
           t2=_mm256_set_pd(A[k][j],A[k][j+1],A[k][j+2],A[k][j+3]);
           t3=_mm256_div_pd(t2,t1);

           //_mm256_storeu_pd(A[k]+j,t3);
           A[k][j+3]=t3[0];
           A[k][j+2]=t3[1];
           A[k][j+1]=t3[2];
           A[k][j]=t3[3];
        }
        for(;j%4!=n2;j++)
                A[k][j]/=t;
        A[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            //double temp2=A[i][k];
            t1=_mm256_set_pd(A[i][k],A[i][k],A[i][k],A[i][k]);
            //t1[0]=t1[1]=t1[2]=t1[3]=A[i][k];
            int t=A[i][k];int j;
            t1=_mm256_set_pd(t,t,t,t);
            for(j=k+1;j<N-n2;j+=4)
            {
                //A[i][j]-=temp2*A[k][j];
                t2=_mm256_set_pd(A[k][j],A[k][j+1],A[k][j+2],A[k][j+3]);
                t4=_mm256_set_pd(A[i][j],A[i][j+1],A[i][j+2],A[i][j+3]);
                t3=_mm256_mul_pd(t1,t2);
                __m256d t5=_mm256_sub_pd(t4,t3);


                A[i][j+3]=t5[0];
                A[i][j+2]=t5[1];
                A[i][j+1]=t5[2];
                A[i][j]=t5[3];
            }
            for(;j%4!=n2;j++)
                 A[i][j]-=t*A[k][j];
            A[i][k]=0;
        }
    }
    return A;
}
int main()
{
    double **A=new double*[N];
    for(int i=0;i<N;i++)
        A[i]=new double[N];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
            {
                while(1)
                {
                    int k=rand()%100;
                    if(!k)
                        continue;
                    else
                    {
                        A[i][j]=k;break;
                    }
                }
            }
    }
    //print(A);
    cout<<endl<<"串行的时间："<<endl;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    double **temp1=SerialLU(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<(tail-head)*1000.0/freq<<endl;
    //print(temp1);
    cout<<endl<<"SSE并行后的时间1："<<endl;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    double **temp2=avxLU(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<(tail-head)*1000.0/freq<<endl;
    //print(temp2);
    cout<<endl<<"SSE并行后的时间2："<<endl;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    double **temp3=sseLU(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<(tail-head)*1000.0/freq<<endl;
    return 0;
}
