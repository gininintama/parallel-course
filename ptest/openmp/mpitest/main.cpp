#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <omp.h>
#include <pmmintrin.h>
using namespace std;
static const int N = 2000;
static const int task = 1;
static const int threadnum = 4;

float test[N][N];
float mat[N][N];
long long head, tail, freq;

//��ʼ������
void init_mat(float test[][N])
{
    srand((unsigned) time(NULL));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            test[i][j] = rand() / 100;
}
//��������
void reset_mat(float mat[][N], float test[][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = test[i][j];
}
//�����ӡ
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
//�����㷨
void naive_lu()
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
            mat[i][k] = 0.0;
        }
    }
}
//��ѭ������ʵ��mpi
void mpi_lu_recycle_run()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        //0�Ž��̽���
        reset_mat(mat, test);
        double time;
        //time = MPI_Wtime();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&mat[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        //����0���߳��Լ�������
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
//        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    mat[k][j] = mat[k][j] / mat[k][k];
                mat[k][k] = 1.0;
//            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
//            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&mat[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {
                    for (int j = k + 1; j < N; j++)
                        mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                    mat[i][k] = 0.0;
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
                MPI_Recv(&mat[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "mpi_lu_recycle_run time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        //time = MPI_Wtime() - time;
        //cout<<"mpi_lu_recycle_run time cost: "<<(double)time*1000 <<" ms\n";
        print_mat(mat);
    }
    else
    {
//        ��0�Ž����Ƚ�������
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&mat[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //�����0���߳��Լ�������
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
//        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    mat[k][j] = mat[k][j] / mat[k][k];
                mat[k][k] = 1.0;
//            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
//            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&mat[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {
                    for (int j = k + 1; j < N; j++)
                        mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                    mat[i][k] = 0.0;
                }
            }
        }
//        �����������Ž��̷��ؽ��
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&mat[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//ѭ������mpi+openmp
void mpi_lu_recycle_combine_run1()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        //0�Ž��̽��о����ʼ��
        reset_mat(mat, test);
        double time;
        //time = MPI_Wtime();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
#pragma omp parallel num_threads(threadnum)
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&mat[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        //����0���߳��Լ�������
        __m128 t1,t2,t3,t4;
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
//        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                #pragma omp for schedule(static)
                for (int j = k + 1; j < N; j++)
                    mat[k][j] = mat[k][j] / mat[k][k];
                mat[k][k] = 1.0;
//            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
//            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&mat[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            #pragma omp for schedule(static)
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {
                    for (int j = k + 1; j < N; j++)
                        mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                    mat[i][k] = 0.0;
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
                MPI_Recv(&mat[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "mpi_lu_recycle_run1 time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        //time = MPI_Wtime() - time;
        //cout<<"mpi_lu_recycle_combine_run1 time cost: "<<(double)time*1000 <<" ms\n";
        print_mat(mat);
    }
    else
    {
//        ��0�Ž����Ƚ�������
#pragma omp parallel for num_threads(threadnum)
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&mat[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //�����0���߳��Լ�������
        int seg = task * num_proc;
        __m128 t1,t2,t3,t4;
        for (int k = 0; k < N; k++)
        {
//        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                #pragma omp for schedule(static)
                for (int j = k + 1; j < N; j++)
                    mat[k][j] = mat[k][j] / mat[k][k];
                mat[k][k] = 1.0;
//            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
//            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&mat[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            #pragma omp for schedule(static)
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
                {

                    for (int j = k + 1; j < N; j++)
                        mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                    mat[i][k] = 0.0;

                }
            }
        }
//        �����������Ž��̷��ؽ��
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&mat[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//ѭ������mpi+sse
void mpi_lu_recycle_combine_run2()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        //0�Ž��̽��о����ʼ��
        reset_mat(mat, test);
        double time;
        //time = MPI_Wtime();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&mat[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        //����0���߳��Լ�������
        __m128 t1,t2,t3,t4;
        int seg = task * num_proc;
        for (int k = 0; k < N; k++)
        {
//        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    mat[k][j] = mat[k][j] / mat[k][k];
                mat[k][k] = 1.0;
//            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
//            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&mat[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
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
            int m = j;
            for (j = m; j < N - (N % 4); j += 4)
            {
                t2=_mm_loadu_ps(mat[k]+j);
                t4=_mm_loadu_ps(mat[i]+j);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                _mm_storeu_ps(mat[i]+j,t5);

            }
            m = j;
			for (j = m; j < N; j++)
            {
                mat[i][j]-=t1[0]*mat[k][j];

            }
        }
        mat[i][k]=0;
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
                MPI_Recv(&mat[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "mpi_lu_recycle_run2 time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        //time = MPI_Wtime() - time;
        //cout<<"mpi_lu_recycle_combine_run2 time cost: "<<(double)time*1000 <<" ms\n";
        print_mat(mat);
    }
    else
    {
//        ��0�Ž����Ƚ�������
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&mat[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //�����0���߳��Լ�������
        int seg = task * num_proc;
        __m128 t1,t2,t3,t4;
        for (int k = 0; k < N; k++)
        {
//        �жϵ�ǰ���Ƿ����Լ�������
            if (int((k % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    mat[k][j] = mat[k][j] / mat[k][k];
                mat[k][k] = 1.0;
//            ��ɼ�������������̷�����Ϣ
                for (int p = 0; p < num_proc; p++)
                    if (p != rank)
                        MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else
            {
//            �����ǰ�в����Լ������񣬽������Ե�ǰ�д�����̵���Ϣ
                MPI_Recv(&mat[k], N, MPI_FLOAT, int((k % seg) / task), 2,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i = k + 1; i < N; i++)
            {
                if (int((i % seg) / task) == rank)
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
            int m = j;
            for (j = m; j < N - (N % 4); j += 4)
            {
                t2=_mm_loadu_ps(mat[k]+j);
                t4=_mm_loadu_ps(mat[i]+j);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                _mm_storeu_ps(mat[i]+j,t5);
            }
            m = j;
			for (j = m; j < N; j++)
            {
                mat[i][j]-=t1[0]*mat[k][j];
            }
        }
        mat[i][k]=0;
                }
            }
        }
//        �����������Ž��̷��ؽ��
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&mat[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

//���黮��ʵ��mpi
void mpi_lu_block_run()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        reset_mat(mat, test);
        double time;
        //time = MPI_Wtime();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
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
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            ��֮��Ľ��̷�����Ϣ
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
//            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0.0;
            }
        }
    }

//        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "mpi_lu_block_run time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        //time = MPI_Wtime() - time;
        //cout<<"mpi_lu_block_run time cost: "<<(double)time*1000 <<" ms\n";
        print_mat(mat);
    }
    else
    {
//        ��0�Ž����Ƚ�������
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
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
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            ��֮��Ľ��̷�����Ϣ
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
//            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0.0;
            }
        }
    }

//        �����������Ž��̷��ؽ��
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//�黮��mpi+openmp
void mpi_lu_block_combine_run1()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        reset_mat(mat, test);
        double time;
        //time = MPI_Wtime();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
#pragma omp parallel num_threads(threadnum)
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }

    int block = N / num_proc;
//    δ���������ֵ�ʣ�ಿ��
    int remain = N % num_proc;
    int begin = rank * block;
//    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
//        �жϵ�ǰ���Ƿ����Լ�������
        if (k >= begin && k < end)
        {
            #pragma omp for schedule(static)
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            ��֮��Ľ��̷�����Ϣ
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
//            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //#pragma omp for schedule(static)
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0.0;
            }
        }
    }
//        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout << "mpi_lu_block_combine_run1 time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        //time = MPI_Wtime() - time;
        //cout<<"mpi_lu_block_combine_run1 time cost: "<<(double)time*1000 <<" ms\n";
        print_mat(mat);
    }
    else
    {
//        ��0�Ž����Ƚ�������
#pragma omp parallel num_threads(threadnum)
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    int block = N / num_proc;
//    δ���������ֵ�ʣ�ಿ��
    int remain = N % num_proc;
    int begin = rank * block;
//    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
//        �жϵ�ǰ���Ƿ����Լ�������
        if (k >= begin && k < end)
        {
            #pragma omp for schedule(static)
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            ��֮��Ľ��̷�����Ϣ
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
//            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //#pragma omp for schedule(static)
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0.0;

            }
        }
    }
//        �����������Ž��̷��ؽ��
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
//�黮��mpi+sse
void mpi_lu_block_combine_run2()
{
    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        reset_mat(mat, test);
        double time;
        //time = MPI_Wtime();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
//        ��0�Ž��̽������񻮷�
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }

    int block = N / num_proc;
//    δ���������ֵ�ʣ�ಿ��
    int remain = N % num_proc;
    int begin = rank * block;
//    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
//        �жϵ�ǰ���Ƿ����Լ�������
        if (k >= begin && k < end)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            ��֮��Ľ��̷�����Ϣ
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
//            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
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
            int m = j;
            //#pragma omp for schedule(guided, 20)
            for (j = m; j < N - (N % 4); j += 4)
            {
                t2=_mm_loadu_ps(mat[k]+j);
                t4=_mm_loadu_ps(mat[i]+j);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                _mm_storeu_ps(mat[i]+j,t5);

            }
            m = j;
			//#pragma omp for schedule(guided, 20)
			for (j = m; j < N; j++)
            {
                mat[i][j]-=t1[0]*mat[k][j];

            }
        }
        mat[i][k]=0;
            }
        }
    }
//        ������0�Ž����Լ��������������������̴���֮��Ľ��
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "mpi_lu_block_combine_run1 time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
        //time = MPI_Wtime() - time;
        //cout<<"mpi_lu_block_combine_run2 time cost: "<<(double)time*1000 <<" ms\n";
        print_mat(mat);
    }
    else
    {
//        ��0�Ž����Ƚ�������
#pragma omp parallel num_threads(threadnum)
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    int block = N / num_proc;
//    δ���������ֵ�ʣ�ಿ��
    int remain = N % num_proc;
    int begin = rank * block;
//    ��ǰ����Ϊ���һ������ʱ���账��ʣ�ಿ��
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
//        �жϵ�ǰ���Ƿ����Լ�������
        if (k >= begin && k < end)
        {
            #pragma omp for schedule(static)
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            ��֮��Ľ��̷�����Ϣ
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
//            �����������ڵ�ǰ����ǰһ���̵������������Ϣ
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //#pragma omp for schedule(static)
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
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
            int m = j;
            for (j = m; j < N - (N % 4); j += 4)
            {
                t2=_mm_loadu_ps(mat[k]+j);
                t4=_mm_loadu_ps(mat[i]+j);
                __m128 t5=_mm_sub_ps(t4,_mm_mul_ps(t1,t2));
                _mm_storeu_ps(mat[i]+j,t5);

            }
            m = j;
			for (j = m; j < N; j++)
            {
                mat[i][j]-=t1[0]*mat[k][j];

            }
        }
        mat[i][k]=0;

            }
        }
    }
//        �����������Ž��̷��ؽ��
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
int main(){
    init_mat(test);
    /*
    reset_mat(mat, test);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    naive_lu();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    print_mat(mat);
    cout << "naive_lu time cost: "<< (tail - head) * 1000 / freq << "ms" << endl;
    */
    MPI_Init(NULL, NULL);
    //mpi_lu_recycle_run();
    mpi_lu_recycle_combine_run1();
    //mpi_lu_recycle_combine_run2();
    //mpi_lu_block_run();
    //mpi_lu_block_combine_run1();
    //mpi_lu_block_combine_run2();
    MPI_Finalize();

    return 0;
}

