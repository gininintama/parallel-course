#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include <omp.h>

using namespace std;

const int N = 1024;
float A[N][N],B[N][N];

const int threadnum = 4;
const int cyc = 25;
long long head, tail, freq;
int re, re1; long long temp;

void serial_LU(float mat[][N])
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

void omp_LU_static(float mat[][N])
{
	#pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		#pragma omp for schedule(static)
		for (int j = k + 1; j < N; j++)
		{
			mat[k][j] = mat[k][j] / mat[k][k];
		}
		mat[k][k] = 1.0;
		#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}

void omp_LU_dynamic(float mat[][N])
{
	#pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		#pragma omp for schedule(dynamic, 24)
		for (int j = k + 1; j < N; j++)
		{
			mat[k][j] = mat[k][j] / mat[k][k];
		}
		mat[k][k] = 1.0;
		#pragma omp for schedule(dynamic, 24)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}

void omp_LU_guided(float mat[][N])
{
	#pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		#pragma omp for schedule(guided, 24)
		for (int j = k + 1; j < N; j++)
		{
			mat[k][j] = mat[k][j] / mat[k][k];
		}
		mat[k][k] = 1.0;
		#pragma omp for schedule(guided, 24)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}
void omp_LU_SSE_static(float mat[][N]) {
	__m128 t1, t2, t3, t4;
	int j;
	#pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		j = k;
		t1 = _mm_set1_ps(mat[k][k]);
		if (N - j < 4)
		{
			for (j; j < N; j++) {

				mat[k][j] /= t1[0];
			}
		}
		else
		{
			for (j = k; j < N - (N % 4); j += 4)
			{
				t2 = _mm_loadu_ps(mat[k] + j);
				t3 = _mm_div_ps(t2, t1);
				_mm_storeu_ps(mat[k] + j, t3);
			}
			int m = j;
			#pragma omp for schedule(static)
			for (j = m; j < N; j++)
			{
				mat[k][j] /= t1[0];
			}
		}

		mat[k][k] = 1.0;
		#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			int j = k + 1;
			t1 = _mm_set1_ps(mat[i][k]);

			if (N - j < 4)
			{
				while (j < N)
				{
					mat[i][j] -= t1[0] * mat[k][j];
					j++;
				}
			}
			else
			{
				while (j < N - (N % 4))
				{
					t2 = _mm_loadu_ps(mat[k] + j);
					t4 = _mm_loadu_ps(mat[i] + j);
					__m128 t5 = _mm_sub_ps(t4, _mm_mul_ps(t1, t2));
					_mm_storeu_ps(mat[i] + j, t5);
					j += 4;
				}
				while (j < N)
				{
					mat[i][j] -= t1[0] * mat[k][j];
					j++;
				}
			}
			mat[i][k] = 0;
		}
	}
}
void omp_LU_SSE_dynamic(float mat[][N]) {
	__m128 t1, t2, t3, t4;
	int j;
	#pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		j = k;
		t1 = _mm_set1_ps(mat[k][k]);
		if (N - j < 4)
		{
			for (j; j < N; j++) {

				mat[k][j] /= t1[0];

			}
		}
		else
		{
			for (j = k; j < N - (N % 4); j += 4)
			{
				t2 = _mm_loadu_ps(mat[k] + j);
				t3 = _mm_div_ps(t2, t1);
				_mm_storeu_ps(mat[k] + j, t3);
			}
			int m = j;
			#pragma omp for schedule(dynamic, 24)
			for (j = m; j < N; j++)
			{
				mat[k][j] /= t1[0];
			}
		}

		mat[k][k] = 1.0;
		#pragma omp for schedule(dynamic, 24)
		for (int i = k + 1; i < N; i++)
		{
			int j = k + 1;
			t1 = _mm_set1_ps(mat[i][k]);

			if (N - j < 4)
			{
				while (j < N)
				{
					mat[i][j] -= t1[0] * mat[k][j];
					j++;
				}
			}
			else
			{
				while (j < N - (N % 4))
				{
					t2 = _mm_loadu_ps(mat[k] + j);
					t4 = _mm_loadu_ps(mat[i] + j);
					__m128 t5 = _mm_sub_ps(t4, _mm_mul_ps(t1, t2));
					_mm_storeu_ps(mat[i] + j, t5);
					j += 4;
				}
				while (j < N)
				{
					mat[i][j] -= t1[0] * mat[k][j];
					j++;
				}
			}
			mat[i][k] = 0;
		}
	}
}
void omp_LU_SSE_guided(float mat[][N]) {
	__m128 t1, t2, t3, t4;
	int j;
 #pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		j = k;
		t1 = _mm_set1_ps(mat[k][k]);
		if (N - j < 4)
		{
			for (j; j < N; j++) {

				mat[k][j] /= t1[0];
			}
		}
		else
		{
			for (j = k; j < N - (N % 4); j += 4)
			{
				t2 = _mm_loadu_ps(mat[k] + j);
				t3 = _mm_div_ps(t2, t1);
				_mm_storeu_ps(mat[k] + j, t3);
			}
			int m = j;
		#pragma omp for schedule(guided, 24)
			for (j = m; j < N; j++)
			{
				mat[k][j] /= t1[0];
			}
		}

		mat[k][k] = 1.0;
		#pragma omp for schedule(guided, 24)
		for (int i = k + 1; i < N; i++)
		{
			int j = k + 1;
			t1 = _mm_set1_ps(mat[i][k]);

			if (N - j < 4)
			{
				while (j < N)
				{
					mat[i][j] -= t1[0] * mat[k][j];
					j++;
				}
			}
			else
			{
				while (j < N - (N % 4))
				{
					t2 = _mm_loadu_ps(mat[k] + j);
					t4 = _mm_loadu_ps(mat[i] + j);
					__m128 t5 = _mm_sub_ps(t4, _mm_mul_ps(t1, t2));
					_mm_storeu_ps(mat[i] + j, t5);
					j += 4;
				}
				while (j < N)
				{
					mat[i][j] -= t1[0] * mat[k][j];
					j++;
				}
			}
			mat[i][k] = 0;
		}
	}
}
void omp_LU_AVX_static(float mat[][N]) {
	__m256 t1, t2, t3, t4;
	#pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		int j = k;
		t1 = _mm256_set1_ps(B[k][k]);
		if (N - j < 8)
		{
			while (j < N)
			{
				B[k][j] /= t1[0];
				j++;
			}
		}
		else
		{
			while (j < N - (N % 8))
			{
				t2 = _mm256_loadu_ps(B[k] + j);
				t3 = _mm256_div_ps(t2, t1);
				_mm256_storeu_ps(B[k] + j, t3);
				j += 8;
			}
			int m = j;
			#pragma omp for schedule(static)
			for (j = m; j < N; j++)
			{
				mat[k][j] /= t1[0];
			}
		}
		B[k][k] = 1.0;
		#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			int j = k + 1;
			t1 = _mm256_set1_ps(B[i][k]);

			if (N - j < 8)
			{
				while (j < N)
				{
					B[i][j] -= t1[0] * B[k][j];
					j++;
				}
			}
			else
			{
				while (j < N - (N % 8))
				{
					t2 = _mm256_loadu_ps(B[k] + j);
					t4 = _mm256_loadu_ps(B[i] + j);
					__m256 t5 = _mm256_sub_ps(t4, _mm256_mul_ps(t1, t2));
					_mm256_storeu_ps(B[i] + j, t5);
					j += 8;
				}
				while (j < N)
				{
					B[i][j] -= t1[0] * B[k][j];
					j++;
				}
			}
			B[i][k] = 0;
		}
	}
}
void omp_LU_AVX_dynamic(float mat[][N]) {
	__m256 t1, t2, t3, t4;
#pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		int j = k;
		t1 = _mm256_set1_ps(B[k][k]);
		if (N - j < 8)
		{
			while (j < N)
			{
				B[k][j] /= t1[0];
				j++;
			}
		}
		else
		{
			while (j < N - (N % 8))
			{
				t2 = _mm256_loadu_ps(B[k] + j);
				t3 = _mm256_div_ps(t2, t1);
				_mm256_storeu_ps(B[k] + j, t3);
				j += 8;
			}
			int m = j;
		#pragma omp for schedule(dynamic,24)
			for (j = m; j < N; j++)
			{
				mat[k][j] /= t1[0];
			}
		}
		B[k][k] = 1.0;
#pragma omp for schedule(dynamic,24)
		for (int i = k + 1; i < N; i++)
		{
			int j = k + 1;
			t1 = _mm256_set1_ps(B[i][k]);

			if (N - j < 8)
			{
				while (j < N)
				{
					B[i][j] -= t1[0] * B[k][j];
					j++;
				}
			}
			else
			{
				while (j < N - (N % 8))
				{
					t2 = _mm256_loadu_ps(B[k] + j);
					t4 = _mm256_loadu_ps(B[i] + j);
					__m256 t5 = _mm256_sub_ps(t4, _mm256_mul_ps(t1, t2));
					_mm256_storeu_ps(B[i] + j, t5);
					j += 8;
				}
				while (j < N)
				{
					B[i][j] -= t1[0] * B[k][j];
					j++;
				}
			}
			B[i][k] = 0;
		}
	}
}

void omp_LU_AVX_guided(float mat[][N]) {
	__m256 t1, t2, t3, t4;
    #pragma omp parallel num_threads(threadnum)
	for (int k = 0; k < N; k++)
	{
		int j = k;
		t1 = _mm256_set1_ps(B[k][k]);
		if (N - j < 8)
		{
			while (j < N)
			{
				B[k][j] /= t1[0];
				j++;
			}
		}
		else
		{
			while (j < N - (N % 8))
			{
				t2 = _mm256_loadu_ps(B[k] + j);
				t3 = _mm256_div_ps(t2, t1);
				_mm256_storeu_ps(B[k] + j, t3);
				j += 8;
			}
			int m = j;
            #pragma omp for schedule(guided,24)
			for (j = m; j < N; j++)
			{
				mat[k][j] /= t1[0];
			}
		}
		B[k][k] = 1.0;
        #pragma omp for schedule(guided,24)
		for (int i = k + 1; i < N; i++)
		{
			int j = k + 1;
			t1 = _mm256_set1_ps(B[i][k]);

			if (N - j < 8)
			{
				while (j < N)
				{
					B[i][j] -= t1[0] * B[k][j];
					j++;
				}
			}
			else
			{
				while (j < N - (N % 8))
				{
					t2 = _mm256_loadu_ps(B[k] + j);
					t4 = _mm256_loadu_ps(B[i] + j);
					__m256 t5 = _mm256_sub_ps(t4, _mm256_mul_ps(t1, t2));
					_mm256_storeu_ps(B[i] + j, t5);
					j += 8;
				}
				while (j < N)
				{
					B[i][j] -= t1[0] * B[k][j];
					j++;
				}
			}
			B[i][k] = 0;
		}
	}
}

void init()
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A[i][j] = rand() / 100;
}

void reset()
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			B[i][j] = A[i][j];
}
void print(float mat[][N])
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			cout << mat[i][j] << " ";
		cout << endl;
	}
}

int main()
{
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	init();
	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		serial_LU(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "serial_LU: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);

	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_static(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_static: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);


	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_dynamic(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_dynamic: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);
	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_guided(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_guided: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);

	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_SSE_static(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_SSE_static: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);
	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_SSE_dynamic(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_SSE_dynamic: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);

	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_SSE_guided(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_SSE_guided: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);

	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_AVX_static(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_AVX_static: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);

	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_AVX_dynamic(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}
	cout << "omp_LU_AVX_dynamic: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);
	re = re1 = cyc; temp = 0;
	while (re) {
		reset();
		//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		omp_LU_AVX_guided(B);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		temp += (tail - head) * 1000.0;
		re--;
	}

	cout << "omp_LU_AVX_guided: " << temp / (re1 * freq) << "ms" << endl;
	//print(B);
	cout << endl;
	return 0;
}
