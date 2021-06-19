#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
//#include <windows.h>
#include <omp.h>
#include <pmmintrin.h>
using namespace std;
static const int N = 2000;
static const int task = 1;
static const int thread_num = 4;

float A[N][N];
float B[N][N];
long long head, tail, freq;

//��ʼ������
void initA()
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = rand() / 100;
}
//��������
void resetB(float B[][N], float A[][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = A[i][j];
}
//�����ӡ
void print(float B[][N])
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << B[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}
//�����㷨
void serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            B[k][j] = B[k][j] / B[k][k];
        B[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                B[i][j] = B[i][j] - B[i][k] * B[k][j];
            B[i][k] = 0.0;
        }
    }
}
//��ѭ������ʵ��mpi
void mpi_recycle()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        //0�Ž��̽���
        resetB(B, A);
        double time;
        time = MPI_Wtime();
        //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&B[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        //����0���߳��Լ�������
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                //            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&B[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {
                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;
                }
            }
        }
        //        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&B[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout << "mpi_recycle time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        time = MPI_Wtime() - time;
        cout << "mpi_recycle time cost: " << (double)time * 1000 << " ms\n";
        //print(B);
    }
    else
    {
        //        ��0�Ž����Ƚ�������
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&B[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //�����0���߳��Լ�������
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                //            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&B[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {
                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;
                }
            }
        }
        //        �����������Ž��̷��ؽ��
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&B[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//ѭ������mpi+openmp
void mpi_omp_recycle()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        //0�Ž��̽��о����ʼ��
        resetB(B, A);
        double time;
        time = MPI_Wtime();
        //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
#pragma omp parallel num_threads(thread_num)
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&B[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        //����0���߳��Լ�������
        __m128 t1, t2, t3, t4;
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
#pragma omp for schedule(static)
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                //            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&B[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
#pragma omp for schedule(static)
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {
                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;
                }
            }
        }
        //        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&B[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout << "mpi_lu_recycle_run1 time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        time = MPI_Wtime() - time;
        cout << "mpi_omp_recycle time cost: " << (double)time * 1000 << " ms\n";
        //print(B);
    }
    else
    {
        //        ��0�Ž����Ƚ�������
#pragma omp parallel for num_threads(thread_num)
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&B[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //�����0���߳��Լ�������
        int seg = task * num_proc;
        __m128 t1, t2, t3, t4;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
#pragma omp for schedule(static)
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                //            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&B[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
#pragma omp for schedule(static)
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {

                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;

                }
            }
        }
        //        �����������Ž��̷��ؽ��
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&B[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//ѭ������mpi+sse
void mpi_sse_recycle()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        //0�Ž��̽��о����ʼ��
        resetB(B, A);
        double time;
        time = MPI_Wtime();
        //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&B[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        //����0���߳��Լ�������
        __m128 t1, t2, t3, t4;
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                //            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&B[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {
                    int j = k + 1;
                    t1 = _mm_set1_ps(B[i][k]);

                    if (N - j < 4)
                    {
                        while (j < N)
                        {
                            B[i][j] -= t1[0] * B[k][j];
                            j++;
                        }
                    }
                    else
                    {
                        int m = j;
                        for (j = m; j < N - (N % 4); j += 4)
                        {
                            t2 = _mm_loadu_ps(B[k] + j);
                            t4 = _mm_loadu_ps(B[i] + j);
                            __m128 t5 = _mm_sub_ps(t4, _mm_mul_ps(t1, t2));
                            _mm_storeu_ps(B[i] + j, t5);

                        }
                        m = j;
                        for (j = m; j < N; j++)
                        {
                            B[i][j] -= t1[0] * B[k][j];

                        }
                    }
                    B[i][k] = 0;
                }
            }
        }
        //        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&B[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout << "mpi_lu_recycle_run1 time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        time = MPI_Wtime() - time;
        cout << "mpi_sse_recycle time cost: " << (double)time * 1000 << " ms\n";
        //print(B);
    }
    else
    {
        //        ��0�Ž����Ƚ�������
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&B[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //�����0���߳��Լ�������
        int seg = task * num_proc;
        __m128 t1, t2, t3, t4;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                //            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&B[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {

                    int j = k + 1;
                    t1 = _mm_set1_ps(B[i][k]);

                    if (N - j < 4)
                    {
                        while (j < N)
                        {
                            B[i][j] -= t1[0] * B[k][j];
                            j++;
                        }
                    }
                    else
                    {
                        int m = j;
                        for (j = m; j < N - (N % 4); j += 4)
                        {
                            t2 = _mm_loadu_ps(B[k] + j);
                            t4 = _mm_loadu_ps(B[i] + j);
                            __m128 t5 = _mm_sub_ps(t4, _mm_mul_ps(t1, t2));
                            _mm_storeu_ps(B[i] + j, t5);
                        }
                        m = j;
                        for (j = m; j < N; j++)
                        {
                            B[i][j] -= t1[0] * B[k][j];
                        }
                    }
                    B[i][k] = 0;
                }
            }
        }
        //        �����������Ž��̷��ؽ��
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&B[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//���黮��ʵ��mpi
void mpi_block()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        resetB(B, A);
        double time;
        time = MPI_Wtime();
        //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&B[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&B[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }

        int block = N / num_proc;
        //    δ���������ֵ�ʣ�ಿ��
        int remain = N % num_proc;
        int begin = rank * block;
        //    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
        int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (k >= begin && k < end)
            {
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��֮��Ľ��̷�����Ϣ
                for (int p = rank + 1; p < num_proc; p++)
                    MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                int cur_p = k / block;
                //            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
                if (cur_p < rank)
                    MPI_Recv(&B[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = begin; i < end && i < N; i++)
            {
                if (i >= k + 1)
                {
                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;
                }
            }
        }

        //        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&B[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&B[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout << "mpi_block time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        time = MPI_Wtime() - time;
        cout << "mpi_block time cost: " << (double)time * 1000 << " ms\n";
       // print(B);
    }
    else
    {
        //        ��0�Ž����Ƚ�������
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&B[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&B[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int block = N / num_proc;
        //    δ���������ֵ�ʣ�ಿ��
        int remain = N % num_proc;
        int begin = rank * block;
        //    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
        int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (k >= begin && k < end)
            {
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��֮��Ľ��̷�����Ϣ
                for (int p = rank + 1; p < num_proc; p++)
                    MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                int cur_p = k / block;
                //            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
                if (cur_p < rank)
                    MPI_Recv(&B[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = begin; i < end && i < N; i++)
            {
                if (i >= k + 1)
                {
                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;
                }
            }
        }

        //        �����������Ž��̷��ؽ��
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&B[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&B[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//�黮��mpi+openmp
void mpi_omp_block()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        resetB(B, A);
        double time;
        time = MPI_Wtime();
        //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
#pragma omp parallel num_threads(thread_num)
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&B[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&B[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }

        int block = N / num_proc;
        //    δ���������ֵ�ʣ�ಿ��
        int remain = N % num_proc;
        int begin = rank * block;
        //    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
        int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
        __m128 t1, t2, t3, t4;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (k >= begin && k < end)
            {
#pragma omp for schedule(static)
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                 //            ��֮��Ľ��̷�����Ϣ
                for (int p = rank + 1; p < num_proc; p++)
                    MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                int cur_p = k / block;
                //            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
                if (cur_p < rank)
                    MPI_Recv(&B[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //#pragma omp for schedule(static)
            for (int i = begin; i < end && i < N; i++)
            {
                if (i >= k + 1)
                {
                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;
                }
            }
        }
        //        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&B[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&B[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout << "mpi_omp_block time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        time = MPI_Wtime() - time;
        cout << "mpi_omp_block time cost: " << (double)time * 1000 << " ms\n";
        //print(B);
    }
    else
    {
        //        ��0�Ž����Ƚ�������
#pragma omp parallel num_threads(thread_num)
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&B[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&B[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int block = N / num_proc;
        //    δ���������ֵ�ʣ�ಿ��
        int remain = N % num_proc;
        int begin = rank * block;
        //    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
        int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
        __m128 t1, t2, t3, t4;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (k >= begin && k < end)
            {
#pragma omp for schedule(static)
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
             //            ��֮��Ľ��̷�����Ϣ
                for (int p = rank + 1; p < num_proc; p++)
                    MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                int cur_p = k / block;
                //            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
                if (cur_p < rank)
                    MPI_Recv(&B[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //#pragma omp for schedule(static)
            for (int i = begin; i < end && i < N; i++)
            {
                if (i >= k + 1)
                {
                    for (int j = k + 1; j < N; j++)
                        B[i][j] = B[i][j] - B[i][k] * B[k][j];
                    B[i][k] = 0.0;

                }
            }
        }
        //        �����������Ž��̷��ؽ��
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&B[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&B[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//�黮��mpi+sse
void mpi_sse_block()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        resetB(B, A);
        double time;
        time = MPI_Wtime();
        //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&B[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&B[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }

        int block = N / num_proc;
        //    δ���������ֵ�ʣ�ಿ��
        int remain = N % num_proc;
        int begin = rank * block;
        //    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
        int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
        __m128 t1, t2, t3, t4;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (k >= begin && k < end)
            {
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��֮��Ľ��̷�����Ϣ
                for (int p = rank + 1; p < num_proc; p++)
                    MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                int cur_p = k / block;
                //            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
                if (cur_p < rank)
                    MPI_Recv(&B[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = begin; i < end && i < N; i++)
            {
                if (i >= k + 1)
                {
                    int j = k + 1;
                    t1 = _mm_set1_ps(B[i][k]);

                    if (N - j < 4)
                    {
                        while (j < N)
                        {
                            B[i][j] -= t1[0] * B[k][j];
                            j++;
                        }
                    }
                    else
                    {
                        int m = j;
                        //#pragma omp for schedule(guided, 20)
                        for (j = m; j < N - (N % 4); j += 4)
                        {
                            t2 = _mm_loadu_ps(B[k] + j);
                            t4 = _mm_loadu_ps(B[i] + j);
                            __m128 t5 = _mm_sub_ps(t4, _mm_mul_ps(t1, t2));
                            _mm_storeu_ps(B[i] + j, t5);

                        }
                        m = j;
                        //#pragma omp for schedule(guided, 20)
                        for (j = m; j < N; j++)
                        {
                            B[i][j] -= t1[0] * B[k][j];

                        }
                    }
                    B[i][k] = 0;
                }
            }
        }
        //        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&B[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&B[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout << "mpi_omp_block time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        time = MPI_Wtime() - time;
        cout << "mpi_sse_block time cost: " << (double)time * 1000 << " ms\n";
        //print(B);
    }
    else
    {
        //        ��0�Ž����Ƚ�������
#pragma omp parallel num_threads(thread_num)
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&B[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&B[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int block = N / num_proc;
        //    δ���������ֵ�ʣ�ಿ��
        int remain = N % num_proc;
        int begin = rank * block;
        //    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
        int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
        __m128 t1, t2, t3, t4;
        for (int k = 0; k < N; k++)
        {
            //        �жϵ�ǰ���Ƿ����Լ�������
            if (k >= begin && k < end)
            {
#pragma omp for schedule(static)
                for (int j = k + 1; j < N; j++)
                    B[k][j] = B[k][j] / B[k][k];
                B[k][k] = 1.0;
                //            ��֮��Ľ��̷�����Ϣ
                for (int p = rank + 1; p < num_proc; p++)
                    MPI_Send(&B[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
                int cur_p = k / block;
                //            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
                if (cur_p < rank)
                    MPI_Recv(&B[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //#pragma omp for schedule(static)
            for (int i = begin; i < end && i < N; i++)
            {
                if (i >= k + 1)
                {
                    int j = k + 1;
                    t1 = _mm_set1_ps(B[i][k]);

                    if (N - j < 4)
                    {
                        while (j < N)
                        {
                            B[i][j] -= t1[0] * B[k][j];
                            j++;
                        }
                    }
                    else
                    {
                        int m = j;
                        for (j = m; j < N - (N % 4); j += 4)
                        {
                            t2 = _mm_loadu_ps(B[k] + j);
                            t4 = _mm_loadu_ps(B[i] + j);
                            __m128 t5 = _mm_sub_ps(t4, _mm_mul_ps(t1, t2));
                            _mm_storeu_ps(B[i] + j, t5);

                        }
                        m = j;
                        for (j = m; j < N; j++)
                        {
                            B[i][j] -= t1[0] * B[k][j];

                        }
                    }
                    B[i][k] = 0;

                }
            }
        }
        //        �����������Ž��̷��ؽ��
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&B[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&B[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
int main() {
    initA();

    /*
    resetB(B, A);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    serial();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    print(B);
    cout << "serial time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
    */
    MPI_Init(NULL, NULL);
    mpi_recycle();
    //mpi_omp_recycle();
    //mpi_sse_recycle();
    mpi_block();
    //mpi_omp_block();
    //mpi_sse_block();
    MPI_Finalize();

    return 0;
}

