#include <iostream>
#include <immintrin.h>
#include <nmmintrin.h>
#include <stdlib.h>
#include <windows.h>
#include <stdio.h>
#include <time.h>
using namespace std;



#define N 1023
#define REP 10
long long head,tail,freq;
long long timeofSerialLU[REP],timeofsseLU[REP],timeofavxLU[REP],timeofalsseLU[REP];
long long timeofsseLU1[REP],timeofurolsseLU[REP];
void print(float **matrix)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << matrix[i][j]<<" ";
        }
        cout << endl;
    }

}
void SerialLU(float** A,int num){
    float **B=new float*[N];
    for(int i=0;i<N;i++)
        B[i]=new float[N];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
                B[i][j]=A[i][j];
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int k=0;k<N;k++)
    {
        float temp=B[k][k];
        for(int j=k;j<N;j++)
        {
            B[k][j]/=temp;
        }
        B[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            float temp2=B[i][k];
            for(int j=k+1;j<N;j++)
            {
                B[i][j]-=temp2*B[k][j];
            }
            B[i][k]=0;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    timeofSerialLU[num]=(tail-head)*1000.0;
    //print(B);
}

void sseLU(float** A,int num){
    float **B=new float*[N];
    for(int i=0;i<N;i++)
        B[i]=new float[N];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
                B[i][j]=A[i][j];
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m128 t1,t2,t3,t4;
    for(int k=0;k<N;k++)
    {
        int j=k;
        t1=_mm_set1_ps(B[k][k]);
        if(N-j<4)
        {
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
        else
        {
            while(j<N-(N%4))
            {
                t2=_mm_loadu_ps(B[k]+j);
                t3=_mm_div_ps(t2,t1);
                _mm_storeu_ps(B[k]+j,t3);
                j+=4;
            }
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }

         B[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int j=k+1;
            t1=_mm_set1_ps(B[i][k]);

        if(N-j<4)
        {
            while(j<N)
            {
               B[i][j]-=t1[0]*B[k][j];
               j++;
            }
        }
        else
        {
            while(j<N-(N%4))
            {
                t2=_mm_loadu_ps(B[k]+j);
                t4=_mm_loadu_ps(B[i]+j);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                _mm_storeu_ps(B[i]+j,t5);
                j+=4;
            }
            while(j<N)
            {
                B[i][j]-=t1[0]*B[k][j];
                j++;
            }
        }
        B[i][k]=0;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    timeofsseLU[num]=(tail-head)*1000.0;
    //print(B);
}

void avxLU(float** A,int num){
    float **B=new float*[N];
    for(int i=0;i<N;i++)
        B[i]=new float[N];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
                B[i][j]=A[i][j];
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m256 t1,t2,t3,t4;
    for(int k=0;k<N;k++)
    {
        int j=k;
        t1=_mm256_set1_ps(B[k][k]);
        if(N-j<8)
        {
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
        else
        {
            while(j<N-(N%8))
            {
                t2=_mm256_loadu_ps(B[k]+j);
                t3=_mm256_div_ps(t2,t1);
                _mm256_storeu_ps(B[k]+j,t3);
                j+=8;
            }
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }

        B[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int j=k+1;
            t1=_mm256_set1_ps(B[i][k]);

        if(N-j<8)
        {
            while(j<N)
            {
               B[i][j]-=t1[0]*B[k][j];
               j++;
            }
        }
        else
        {
            while(j<N-(N%8))
            {
                t2=_mm256_loadu_ps(B[k]+j);
                t4=_mm256_loadu_ps(B[i]+j);
                __m256 t5=_mm256_sub_ps(t4,_mm256_mul_ps(t1,t2));
                _mm256_storeu_ps(B[i]+j,t5);
                j+=8;
            }
            while(j<N)
            {
                B[i][j]-=t1[0]*B[k][j];
                j++;
            }
        }
        B[i][k]=0;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    timeofavxLU[num]=(tail-head)*1000.0;
    //print(B);
}

void aligned_sseLU(float** A,int num){
    float **B=new float*[N];
    for(int i=0;i<N;i++)
        B[i]=new float[N];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
                B[i][j]=A[i][j];
    }
   QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
   QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m128 t1,t2,t3,t4;
    for(int k=0;k<N;k++)
    {
        int j=k;
        t1=_mm_set1_ps(B[k][k]);
        while(j<N&&((uintptr_t)(B[k]+j)%16)!=0)
        {
            B[k][j]/=t1[0];
            j++;
        }
        if(N-j<4)
        {
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
        else
        {
            while(j<N-(N%4))
            {
                t2=_mm_load_ps(B[k]+j);
                t3=_mm_div_ps(t2,t1);
                _mm_store_ps(B[k]+j,t3);
                j+=4;
            }
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
        B[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int j=k+1;
            t1=_mm_set1_ps(B[i][k]);
        while(j<N&&((uintptr_t)(B[k]+j)%16)!=0)
            {
               B[i][j]-=t1[0]*B[k][j];
               j++;
            }
        if(N-j<4)
        {
            while(j<N)
            {
               B[i][j]-=t1[0]*B[k][j];
               j++;
            }
        }
        else
        {
            while(j<N-(N%4))
            {
                t2=_mm_load_ps(B[k]+j);
                t4=_mm_load_ps(B[i]+j);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                _mm_store_ps(B[i]+j,t5);
                j+=4;
            }
            while(j<N)
            {
                B[i][j]-=t1[0]*B[k][j];
                j++;
            }
        }
        B[i][k]=0;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    timeofalsseLU[num]=(tail-head)*1000.0;
    //print(B);
}

void sseLU1(float** A,int num){
    float **B=new float*[N];
    for(int i=0;i<N;i++)
        B[i]=new float[N];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
                B[i][j]=A[i][j];
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m128 t1,t2,t3,t4;
    for(int k=0;k<N;k++)
    {
        float temp=B[k][k];
        for(int j=k;j<N;j++)
        {
            B[k][j]/=temp;
        }

        B[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int j=k+1;
            t1=_mm_set1_ps(B[i][k]);

        if(N-j<4)
        {
            while(j<N)
            {
               B[i][j]-=t1[0]*B[k][j];
               j++;
            }
        }
        else
        {
            while(j<N-(N%4))
            {
                t2=_mm_loadu_ps(B[k]+j);
                t4=_mm_loadu_ps(B[i]+j);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                _mm_storeu_ps(B[i]+j,t5);
                j+=4;
            }
            while(j<N)
            {
                B[i][j]-=t1[0]*B[k][j];
                j++;
            }
        }
        B[i][k]=0;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    timeofsseLU1[num]=(tail-head)*1000.0;
    //print(B);
}

void urolsseLU(float** A,int num){
    float **B=new float*[N];
    for(int i=0;i<N;i++)
        B[i]=new float[N];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
                B[i][j]=A[i][j];
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    __m128 t1,t2,t3,t4;
    __m128 t1_1,t2_1,t3_1,t4_1;
    for(int k=0;k<N;k++)
    {
        int j=k;
        t1=_mm_set1_ps(B[k][k]);
        if(N-j<4)
        {
            while(j<N-(N%2))
            {
                B[k][j]/=t1[0];
                B[k][j+1]/=t1[0];
                j+=2;
            }
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
        else
        {
            while(j<N-(N%8))
            {
                t2=_mm_loadu_ps(B[k]+j);
                t2_1=_mm_loadu_ps(B[k]+j+4);
                t3=_mm_div_ps(t2,t1);
                t3_1=_mm_div_ps(t2_1,t1);
                _mm_storeu_ps(B[k]+j,t3);
                _mm_storeu_ps(B[k]+j+4,t3_1);
                j+=8;
            }
            while(j<N-(N%2))
            {
                B[k][j]/=t1[0];
                B[k][j+1]/=t1[0];
                j+=2;
            }
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
         B[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            int j=k+1;
            t1=_mm_set1_ps(B[i][k]);

        if(N-j<4)
        {
            while(j<N-(N%2))
            {
               B[i][j]-=t1[0]*B[k][j];
               B[i][j+1]-=t1[0]*B[k][j+1];
               j+=2;
            }
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
        else
        {
            while(j<N-(N%8))
            {
                t2=_mm_loadu_ps(B[k]+j);
                t4=_mm_loadu_ps(B[i]+j);
                t2_1=_mm_loadu_ps(B[k]+j+4);
                t4_1=_mm_loadu_ps(B[i]+j+4);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                __m128 t5_1=_mm_sub_ps(t4_1,_mm_mul_ps(t1,t2_1));
                 _mm_storeu_ps(B[i]+j,t5);
                _mm_storeu_ps(B[i]+j+4,t5_1);
                j+=8;
            }
            while(j<N-(N%2))
            {
                B[i][j]-=t1[0]*B[k][j];
                B[i][j+1]-=t1[0]*B[k][j+1];
                j+=2;
            }
            while(j<N)
            {
                B[k][j]/=t1[0];
                j++;
            }
        }
        B[i][k]=0;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    timeofurolsseLU[num]=(tail-head)*1000.0;
    //print(B);
}

int main()
{
    float **A=new float*[N];
    for(int i=0;i<N;i++)
        A[i]=new float[N];
    srand((unsigned)time(NULL));
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
                A[i][j]=(float)(1+rand()%100);
    }
    long long sumofSerialLU=0,sumofsseLU=0,sumofavxLU=0,sumofalsseLU=0;
    long long sumofsseLU1=0,sumofurolssseLU=0;
    for(int t=0;t<REP;t++)
        {
            //SerialLU(A,t);
            sseLU(A,t);
            //avxLU(A,t);
            //aligned_sseLU(A,t);
            //sseLU1(A,t);
            urolsseLU(A,t);
            //sumofSerialLU+=timeofSerialLU[t];
            sumofsseLU+=timeofsseLU[t];
            //sumofavxLU+=timeofavxLU[t];
            //sumofalsseLU+=timeofalsseLU[t];
            //sumofsseLU1+=timeofsseLU1[t];
            sumofurolssseLU+=timeofurolsseLU[t];
        }
    //printf("\n%s%.4lf\n","串行的时间：",(float)sumofSerialLU/(REP*freq));
    printf("\n%s%.4lf\n","普通SSE并行后的时间：",(float)sumofsseLU/(REP*freq));
    //printf("\n%s%.4lf\n","AVX并行后的时间：",(float)sumofavxLU/(REP*freq));
    //printf("\n%s%.4lf\n","SSE对齐并行后的时间：",(float)sumofalsseLU/(REP*freq));
    //printf("\n%s%.4lf\n","内循环二SSE并行后的时间：",(float)sumofsseLU1/(REP*freq));
    printf("\n%s%.4lf\n","循环展开SSE并行后的时间：",(float)sumofurolssseLU/(REP*freq));
    return 0;
}
