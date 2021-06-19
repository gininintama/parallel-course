#include<iostream>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include<cmath>
#include<string>
#include<vector>
using namespace std;
int N = 1000;
//������������ʼ��
vector<vector<double> > A(N, vector<double>(N, 0));
vector<double> b(N, 1);
vector<double> r(N, -1);//�в�����
vector<double> d(N, 0);//��������
vector<double> x(N, 0);//������
//��֤���
void print_b(vector<double>& b) {
    for (int i = 0; i < b.size(); i++) {
        cout << b[i] << " ";
    }
    cout << endl;
}
//�����ڻ�
double compute_neiji(vector<double>& a, vector<double>& b, int myid, int ThreadSize) {
    double result = 0;
    for (int i = myid; i < N; i = i + ThreadSize) {
        result += a[i] * b[i];
    }
    return result;
}
//���²в�
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
    //mpi��ʱ
    double startwtime, endwtime;
    //������before_r�����ۺ͵�����0��Ϊ�˼��㷽������d
    double sum_before_r = 0;
    double before_r = 0;
    //������now_r�����ۼӵ�����0��Ϊ�˼��㲽��length
    double sum_now_r = 0;
    double now_r = 0;
    int iterator = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    //���н��̳�ʼ������A   
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
    //0�Ž��̿�����ʱ
    if (myid == 0)
        startwtime = MPI_Wtime();
    //ÿ�����̶�����N��
    for (iterator = 0; iterator < N; iterator++) {
        //������ԵĲв�����r

        before_r = compute_neiji(r, r, myid, Size);
        r = update_cancha(A, x, b, myid, Size);
        //������Եķ�������d

        now_r = compute_neiji(r, r, myid, Size);
        MPI_Allreduce(&now_r, &sum_now_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //��������Ϊ0.00001
        if (sum_now_r < 0.00001) {
            break;
        }
        MPI_Allreduce(&before_r, &sum_before_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double temp = sum_now_r / sum_before_r;
        for (int i = myid; i < N; i = i + Size) {
            d[i] = -r[i] + temp * d[i];
        }
        //�������0�Ž��̾Ͱ����ݷ��͸�0�Ž���
        if (myid != 0)
        {
            for (int i = myid; i < N; i = i + Size) {
                MPI_Send(&d[i], 1, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
            }
        }
        //�����0�Ž��̾��ý��յ������ݶԷ�������d���и�����ͬ���Ĳ���
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
        //���㲽��length

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
        //0�Ž��̴��������һ��
        if (myid == 0) {
            length = -sum_num / sum_temp1;
        }
        //������lengthͬ��
        MPI_Bcast(&length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //���½�����x

        for (int i = myid; i < N; i = i + Size) {
            x[i] = x[i] + length * d[i];
        }
        //�������0�Ž��̾Ͱ����ݷ��͸�0�Ž���
        if (myid != 0)
        {
            for (int i = myid; i < N; i = i + Size) {
                MPI_Send(&x[i], 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
            }
        }
        //�����0�Ž��̾��ý��յ������ݶԽ��������и�����ͬ���Ĳ���
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
            //cout<<"ʱ�䣺"<<endwtime-startwtime<<endl;
        }
        MPI_Bcast(&x[0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    //������������0�Ž��̴�ӡ��Ϣ
    if (myid == 0) {
        cout << "�������� " << Size << endl;
        cout << "ʱ�䣺" << endwtime - startwtime << endl;
        cout << "����������" << iterator << endl;
    }

    MPI_Finalize();
    return 0;
}
