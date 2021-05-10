#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include <algorithm>
#include <pthread.h>
using namespace std;

const int N = 4;

float mat[N][N];
float test[N][N];

typedef struct
{
	int	threadId;
} threadParm_t;

const int thread_num = 2;
long long head, tail, freq;        // timers
pthread_t threads[thread_num];
threadParm_t threadParm[thread_num];
pthread_barrier_t barrier1;
const int task = 1;
const int seg = task * thread_num;


void init_mat(float test[][N])
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			test[i][j] = (float)(1+rand()%100);
}

void reset_mat(float mat[][N], float test[][N])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			mat[i][j] = test[i][j];
}

void naive_lu(float mat[][N])
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
			mat[k][j] = mat[k][j] / mat[k][k];
		mat[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}

void *block_pthread(void *parm)
{
    threadParm_t *p = (threadParm_t *)parm;
    int id = p->threadId;
    int s = id * (N / thread_num);
    int e = (id + 1) * (N / thread_num);

	for (int k = 0; k < N; k++)
	{
	    if (k + 1 >= s && k < e)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
        }
        pthread_barrier_wait(&barrier1);
        /*int block = (N - k - 1) / thread_num;
        if(N-k-1<thread_num)
        {
            for (int i = k + 1; i < N; i++)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0;
            }
        }
        else{
            for (int i = k + 1 + id * block; i < k + 1 + (id + 1) * block; i++)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0;
            }
        }*/
        int block = (N - k - 1) / thread_num;
        int temp=(N-k-1)&thread_num;
        if(!block)
        {
            for (int i = k + 1; i < k +1+temp&&i<N; i++)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0;
            }
        }
        for (int i = k + 1 + id * block; i < k + 1 + (id + 1) * block; i++)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0;
            }
	}
	pthread_exit(NULL);
}

void *recycle_pthread_lu(void *parm)
{
	threadParm_t *p = (threadParm_t *)parm;
	int id = p->threadId;
    int s = id * task;
	int e = (id + 1) * task;
	for (int k = 0; k < N; k++)
	{
	    if ((k + 1) % seg >= s && (k + 1) % seg < e)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
        }
        pthread_barrier_wait(&barrier1);
        for (int i = k + 1; i < N; i++)
        {
            if (i % seg >= s && i % seg < e)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0;
            }
        }
		pthread_barrier_wait(&barrier1);
	}
	pthread_exit(NULL);
}

void *recycle_pthread_sse_lu(void *parm)
{
	threadParm_t *p = (threadParm_t *)parm;
	int id = p->threadId;
	int s = id * task;
	int e = (id + 1) * task;
	__m128 t1, t2, t3,t4;
	for (int k = 0; k < N; k++)
	{
		if ((k + 1) % seg >= s && (k + 1) % seg < e)
		{
		    int j=k+1;
			t1=_mm_set1_ps(mat[k][k]);
            if(N-j<4)
            {
                while(j<N)
                {
                    mat[k][j]/=t1[0];
                    j++;
                }
            }
            else
            {
                while(j<N-(N%4))
                {
                    t2=_mm_loadu_ps(mat[k]+j);
                    t3=_mm_div_ps(t2,t1);
                    _mm_storeu_ps(mat[k]+j,t3);
                    j+=4;
                }
                while(j<N)
                {
                    mat[k][j]/=t1[0];
                    j++;
                }
            }
			mat[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1; i < N; i++)
		{
			if (i % seg >= s && i % seg < e)
			{
                int j=k+1;
                t1=_mm_set1_ps(mat[i][k]);

                if(N-j<4)
                {
                    while(j<N)
                    {
                        mat[i][j]-=t1[0]*mat[k][j];
                        j++;
                    }
                }
                else
                {
                    while(j<N-(N%4))
                    {
                        t2=_mm_loadu_ps(mat[k]+j);
                        t4=_mm_loadu_ps(mat[i]+j);
                        __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                        _mm_storeu_ps(mat[i]+j,t5);
                        j+=4;
                    }
                    while(j<N)
                    {
                        mat[i][j]-=t1[0]*mat[k][j];
                        j++;
                    }
                }
				mat[i][k] = 0;
			}
		}
		pthread_barrier_wait(&barrier1);
	}
	pthread_exit(NULL);
}

void *recycle_pthread_avx_lu(void *parm)
{
	threadParm_t *p = (threadParm_t *)parm;
	int id = p->threadId;
	int s = id * task;
	int e = (id + 1) * task;
	__m256 t1, t2, t3,t4;
		for (int k = 0; k < N; k++)
	{
		if ((k + 1) % seg >= s && (k + 1) % seg < e)
		{
		    int j=k+1;
			t1=_mm256_set1_ps(mat[k][k]);
            if(N-j<8)
            {
                while(j<N)
                {
                    mat[k][j]/=t1[0];
                    j++;
                }
            }
            else
            {
                while(j<N-(N%8))
                {
                    t2=_mm256_loadu_ps(mat[k]+j);
                    t3=_mm256_div_ps(t2,t1);
                    _mm256_storeu_ps(mat[k]+j,t3);
                    j+=8;
                }
                while(j<N)
                {
                    mat[k][j]/=t1[0];
                    j++;
                }
            }
			mat[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1; i < N; i++)
		{
			if (i % seg >= s && i % seg < e)
			{
				 int j=k+1;
                t1=_mm256_set1_ps(mat[i][k]);

                if(N-j<8)
                {
                    while(j<N)
                    {
                        mat[i][j]-=t1[0]*mat[k][j];
                        j++;
                    }
                }
                else
                {
                    while(j<N-(N%8))
                    {
                        t2=_mm256_loadu_ps(mat[k]+j);
                        t4=_mm256_loadu_ps(mat[i]+j);
                        __m256 t5=_mm256_sub_ps(t4,_mm256_mul_ps(t1,t2));
                        _mm256_storeu_ps(mat[i]+j,t5);
                        j+=8;
                    }
                    while(j<N)
                    {
                        mat[i][j]-=t1[0]*mat[k][j];
                        j++;
                    }
                }
				mat[i][k] = 0;
			}
		}
		pthread_barrier_wait(&barrier1);
	}
	pthread_exit(NULL);
}

void print_mat(float mat[][N])
{
    if (N > 16)
        return;
	cout << endl;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			cout << mat[i][j] << " ";
		cout << endl;
	}
	cout << endl;
}


int main()
{
    pthread_barrier_init(&barrier1, NULL, thread_num);
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
	init_mat(test);

	reset_mat(mat, test);
	print_mat(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	naive_lu(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "naive LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (int i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, block_pthread, (void *)&threadParm[i]);
	}
	for (int i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
    cout << "block pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (long i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, recycle_pthread_lu, (void *)&threadParm[i]);
	}
	for (long i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "recycle pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (long i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, recycle_pthread_sse_lu, (void *)&threadParm[i]);
	}
	for (long i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "recycle sse pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (long i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, recycle_pthread_avx_lu, (void *)&threadParm[i]);
	}
	for (long i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "recycle avx pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	pthread_barrier_destroy(&barrier1);
	return 0;
}
