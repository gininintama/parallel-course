#include<iostream>
#include<math.h>
#include<cstdlib>
#include<windows.h>
#include<stdlib.h>
#include<xmmintrin.h>
#include<fstream>
#include<immintrin.h>
#include<pthread.h>
#include <time.h>

#define THREAD_NUM 4
#define N 1000
#define I 20
int re,re1;long long temp;
using namespace std;

float** A, ** B;

typedef struct {
	int threatId;
}threadParm_t;

pthread_barrier_t	barrier;
long long head, freq, tail;

// 单线程
void LU_single() {

	for (int k = 0; k < N; k++) {
		for (int j = k + 1; j < N; j++) {
			B[k][j] = B[k][j] / B[k][k];
		}
		B[k][k] = 1.0;
		for (int i = k + 1; i < N; i++) {
			for (int l = k + 1; l < N; l++) {
				B[i][l] = B[i][l] - B[i][k] * B[k][l];
			}
			B[i][k] = 0;
		}
	}
}
// 方式一
void* LU_pthread_1(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int r = p->threatId;
	for (int k = 0; k < N; k++) {

		for (int i = k + 1; i < N; i++) {

			if ((i % THREAD_NUM) == r) {

				B[i][k] = B[i][k] / B[k][k];

				for (int j = k + 1; j < N; j++) {
					B[i][j] = B[i][j] - B[i][k] * B[k][j];
				}

			}
		}
		pthread_barrier_wait(&barrier);
	}

	pthread_exit(0);

	return 0;
}
// 方式一使用sse
void* LU_pthread_1_sse(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int r = p->threatId;
	__m128 t1, t2, t3;

	for (int k = 0; k < N; k++) {

		for (int i = k + 1; i < N; i++) {

			if ((i % THREAD_NUM) == r) {

				B[i][k] = B[i][k] / B[k][k];

				int offset = (N - k - 1) % 4;
				for (int j = k + 1; j < k + 1 + offset; j++) {
					B[i][j] = B[i][j] - B[i][k] * B[k][j];
				}
				t1 = _mm_set1_ps(B[i][k]);
				for (int j = k + 1 + offset; j < N; j += 4) {
					t2 = _mm_load_ps(B[k] + j);
					t3 = _mm_load_ps(B[i] + j);
					_mm_store_ps(B[i] + j, _mm_sub_ps(t3, _mm_mul_ps(t1, t2)));
				}

			}
		}
		pthread_barrier_wait(&barrier);
	}

	pthread_exit(0);

	return 0;
}

// 方式一使用avx
void* LU_pthread_1_avx(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int r = p->threatId;
	__m256  t1, t2, t3;

	for (int k = 0; k < N; k++) {

		for (int i = k + 1; i < N; i++) {

			if ((i % THREAD_NUM) == r) {

				B[i][k] = B[i][k] / B[k][k];

				int offset = (N - k - 1) % 8;
				for (int j = k + 1; j < k + 1 + offset; j++) {
					B[i][j] = B[i][j] - B[i][k] * B[k][j];
				}
				t2 = _mm256_set_ps(B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k]);
				for (int j = k + 1 + offset; j < N; j += 8) {
					t3 = _mm256_loadu_ps(B[k] + j);
					t1 = _mm256_loadu_ps(B[i] + j);
					t2 = _mm256_mul_ps(t2, t3);
					t1 = _mm256_sub_ps(t1, t2);
					_mm256_storeu_ps(B[i] + j, t1);
				}

			}
		}
		pthread_barrier_wait(&barrier);
	}
	pthread_exit(0);

	return 0;
}
void init_re()
{
    for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < i; j++) {

			B[i][j] = 0;
		}
		B[i][i] = 1;
	}
}
// 方式二
void* LU_pthread_2(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int r = p->threatId;
	for (int k = 0; k < N; k++) {
		for (int j = k + 1 + r; j < N; j += THREAD_NUM) {
			B[k][j] = B[k][j] / B[k][k];
		}
		B[k][k] = 1;

		pthread_barrier_wait(&barrier);

		for (int i = k + 1 + r; i < N; i += THREAD_NUM) {
			for (int j = k + 1; j < N; j++) {
				B[i][j] = B[i][j] - B[i][k] * B[k][j];
			}
			B[i][k] = 0;
		}
		pthread_barrier_wait(&barrier);
	}

	pthread_exit(0);
	return 0;
}
//方式二使用sse
void* LU_pthread_2_sse(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int r = p->threatId;
	__m128  t1, t2, t3;
	for (int k = 0; k < N; k++) {
		for (int j = k + 1 + r; j < N; j += THREAD_NUM) {
			B[k][j] = B[k][j] / B[k][k];
		}
		B[k][k] = 1;
		pthread_barrier_wait(&barrier);
		for (int i = k + 1 + r; i < N; i += THREAD_NUM) {
			int offset = (N - k - 1) % 4;
			for (int j = k + 1; j < k + 1 + offset; j++) {
				B[i][j] = B[i][j] - B[i][k] * B[k][j];
			}
			t1 = _mm_set1_ps(B[i][k]);
			for (int j = k + 1 + offset; j < N; j += 4) {
				t2 = _mm_load_ps(B[k] + j);
				t3 = _mm_load_ps(B[i] + j);
				_mm_store_ps(B[i] + j, _mm_sub_ps(t3, _mm_mul_ps(t1, t2)));
			}
			B[i][k] = 0;
		}
		pthread_barrier_wait(&barrier);
	}

	pthread_exit(0);
	return 0;
}
//方式二使用avx
void* LU_pthread_2_avx(void* parm) {
	threadParm_t* p = (threadParm_t*)parm;
	int r = p->threatId;
	__m256  t1, t2, t3;

	for (int k = 0; k < N; k++) {
		for (int j = k + 1 + r; j < N; j += THREAD_NUM) {
			B[k][j] = B[k][j] / B[k][k];
		}
		B[k][k] = 1;

		pthread_barrier_wait(&barrier);

		for (int i = k + 1 + r; i < N; i += THREAD_NUM) {
			int offset = (N - k - 1) % 8;
			for (int j = k + 1; j < k + 1 + offset; j++) {
				B[i][j] = B[i][j] - B[i][k] * B[k][j];
			}
			t2 = _mm256_set_ps(B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k]);
			for (int j = k + 1 + offset; j < N; j += 8) {
				t3 = _mm256_loadu_ps(B[k] + j);
				t1 = _mm256_loadu_ps(B[i] + j);
				t2 = _mm256_mul_ps(t2, t3);
				t1 = _mm256_sub_ps(t1, t2);
				_mm256_storeu_ps(B[i] + j, t1);
			}
			B[i][k] = 0;
		}
		pthread_barrier_wait(&barrier);
	}
	pthread_exit(0);
	return 0;
}
void init()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			B[i][j] = A[i][j];
	}
}
void print(float** matrix) {

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

int main() {
	int i;
	pthread_t thread[THREAD_NUM];
	threadParm_t threadParm[THREAD_NUM];
	A = new float* [N];
	for (int i = 0; i < N; i++)
		A[i] = new float[N];
	B = new float* [N];
	for (int i = 0; i < N; i++)
		B[i] = new float[N];
	//test
	/*
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i < j)
				A[i][j] = i + 1;
			else
				A[i][j] = j + 1;
		}
	}
	init();
	print(A);
	LU_single();
	print(B);
	init();
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	for (i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_1, (void *)&threadParm[i]);
	}
	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}
	init_re();
	print(B);
	*/
	srand((unsigned)time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			A[i][j] = (float)(1 + rand() % 100);
	}
	cout << "N=" << N << endl;
	//int re,re1;long long temp;
	//单线程
	re=re1=I;temp=0;
	while(re){
	init();
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	LU_single();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	temp+=(tail - head) * 1000.0;
	re--;
	}
	cout << "single thread: " <<  temp/ (re1*freq) << "ms" << endl;

	//方法一
	re=re1=I;temp=0;
	while(re){
	init();
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_1, (void*)&threadParm[i]);
	}
	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	temp+=(tail - head) * 1000.0;
	re--;
	}
	cout << endl << "LU_pthread_1: " << temp/ (re1*freq) << "ms" << endl;
	pthread_barrier_destroy(&barrier);
	//方法一sse
	re=re1=I;temp=0;
	while(re){
	init();
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_1_sse, (void*)&threadParm[i]);
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	temp+=(tail - head) * 1000.0;
	re--;
	}
	cout << "LU_pthread_1_sse: " << temp/ (re1*freq) << "ms" << endl;
	pthread_barrier_destroy(&barrier);

	//方法一avx
	re=re1=I;temp=0;
	while(re){
	init();
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_1_avx, (void*)&threadParm[i]);
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	temp+=(tail - head) * 1000.0;
	re--;
	}
	cout << "LU_pthread_1_avx: " << temp/ (re1*freq) << "ms" << endl;
	pthread_barrier_destroy(&barrier);
	//方法二
	re=re1=I;temp=0;
	while(re){
	init();
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_2, (void*)&threadParm[i]);
	}
	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	temp+=(tail - head) * 1000.0;
	re--;
	}
	cout << endl << "LU_pthread_2: " << temp/ (re1*freq) << "ms" << endl;
	pthread_barrier_destroy(&barrier);
	//方法二sse
	re=re1=I;temp=0;
	while(re){
	init();
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_2_sse, (void*)&threadParm[i]);
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	temp+=(tail - head) * 1000.0;
	re--;
	}
	cout << "LU_pthread_2_sse: " << temp/ (re1*freq) << "ms" << endl;
	pthread_barrier_destroy(&barrier);
	//方法二avx
	re=re1=I;temp=0;
	while(re){
	init();
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_2_avx, (void*)&threadParm[i]);
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	temp+=(tail - head) * 1000.0;
	re--;
	}
	cout << "LU_pthread_2_avx: " << temp/ (re1*freq) << "ms" << endl;
	pthread_barrier_destroy(&barrier);

}
