#include<iostream>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include<cmath>
#include<string>
#include<vector>
using namespace std;
int N = 1000;
//矩阵与向量初始化
vector<vector<double> > A(N, vector<double>(N, 0));
vector<double> b(N, 1);
vector<double> r(N, -1);//残差向量
vector<double> d(N, 0);//方向向量
vector<double> x(N, 0);//解向量
//验证结果
void print_b(vector<double>& b) {
    for (int i = 0; i < b.size(); i++) {
        cout << b[i] << " ";
    }
    cout << endl;
}
//计算内积
double compute_neiji(vector<double>& a, vector<double>& b, int myid, int ThreadSize) {
    double result = 0;
    for (int i = myid; i < N; i = i + ThreadSize) {
        result += a[i] * b[i];
    }
    return result;
}
//更新残差
vector<double>& update_cancha(vector<vector<double> >& a, vector<double>& x, vector<double>& b, int myid, int ThreadSize) {

    double temp = 0;
    for (int i = myid; i < N; i = i + ThreadSize) {
        temp = 0;
        for (int j = 0; j < N; j++) {
            temp += a[i][j] * x[j];
        }
        r[i] = temp - b[i];
    }
    return r;
}

int main(int argc, char* argv[]) {
    int Size, myid;
    MPI_Status status;
    double receive, send;
    int receiveId;
    //mpi计时
    double startwtime, endwtime;
    //将各个before_r变量累和到进程0，为了计算方向向量d
    double sum_before_r = 0;
    double before_r = 0;
    //将各个now_r变量累加到进程0，为了计算步长length
    double sum_now_r = 0;
    double now_r = 0;
    int iterator = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    //所有进程初始化矩阵A   
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[i][j] = 2;
            }
            if (abs(i - j) == 1) {
                A[i][j] = -1;
            }
        }
    }
    //0号进程开启计时
    if (myid == 0)
        startwtime = MPI_Wtime();
    //每个进程都迭代N次
    for (iterator = 0; iterator < N; iterator++) {
        //计算各自的残差向量r

        before_r = compute_neiji(r, r, myid, Size);
        r = update_cancha(A, x, b, myid, Size);
        //计算各自的方向向量d

        now_r = compute_neiji(r, r, myid, Size);
        MPI_Allreduce(&now_r, &sum_now_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //精度设置为0.00001
        if (sum_now_r < 0.00001) {
            break;
        }
        MPI_Allreduce(&before_r, &sum_before_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double temp = sum_now_r / sum_before_r;
        for (int i = myid; i < N; i = i + Size) {
            d[i] = -r[i] + temp * d[i];
        }
        //如果不是0号进程就把数据发送给0号进程
        if (myid != 0)
        {
            for (int i = myid; i < N; i = i + Size) {
                MPI_Send(&d[i], 1, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
            }
        }
        //如果是0号进程就用接收到的数据对方向向量d进行更新与同步的操作
        if (myid == 0) {
            vector<int> count(Size, 0);
            for (receiveId = 1; receiveId < Size; receiveId++) {
                for (int i = receiveId; i < N; i = i + Size) {
                    MPI_Recv(&receive, 1, MPI_DOUBLE, receiveId, receiveId, MPI_COMM_WORLD, &status);
                    d[receiveId + count[receiveId] * Size] = receive;
                    count[receiveId]++;
                }
            }
        }
        MPI_Bcast(&d[0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //计算步长length

        double sum_num = 0;
        double num = compute_neiji(d, r, myid, Size);
        double sum_temp1 = 0;
        double temp1 = 0;
        double tmp = 0;
        for (int i = myid; i < N; i = i + Size) {
            tmp = 0;
            for (int j = 0; j < N; j++) {
                tmp += A[i][j] * d[j];
            }
            temp1 += tmp * d[i];
        }
        MPI_Reduce(&num, &sum_num, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&temp1, &sum_temp1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        double length = 0;
        //0号进程处进行最后一步
        if (myid == 0) {
            length = -sum_num / sum_temp1;
        }
        //将步长length同步
        MPI_Bcast(&length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //更新解向量x

        for (int i = myid; i < N; i = i + Size) {
            x[i] = x[i] + length * d[i];
        }
        //如果不是0号进程就把数据发送给0号进程
        if (myid != 0)
        {
            for (int i = myid; i < N; i = i + Size) {
                MPI_Send(&x[i], 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
            }
        }
        //如果是0号进程就用接收到的数据对解向量进行更新与同步的操作
        if (myid == 0) {
            vector<int> count(Size, 0);
            for (receiveId = 1; receiveId < Size; receiveId++) {
                for (int i = receiveId; i < N; i = i + Size) {
                    MPI_Recv(&receive, 1, MPI_DOUBLE, receiveId, 100, MPI_COMM_WORLD, &status);
                    x[receiveId + count[receiveId] * Size] = receive;
                    count[receiveId]++;
                }
            }
            endwtime = MPI_Wtime();
            //print_b(x);
            //cout<<"时间："<<endwtime-startwtime<<endl;
        }
        MPI_Bcast(&x[0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    //迭代结束后在0号进程打印信息
    if (myid == 0) {
        cout << "进程数： " << Size << endl;
        cout << "时间：" << endwtime - startwtime << endl;
        cout << "迭代次数：" << iterator << endl;
    }

    MPI_Finalize();
    return 0;
}
