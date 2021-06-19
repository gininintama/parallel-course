#include<iostream>
#include<vector>
#include<cmath>
#include<stdio.h>
#include<omp.h>
#include<sys/time.h>
using namespace std;
int threadnum = 2;
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
double compute_neiji(vector<double>& a, vector<double>& b) {
    double result = 0;
#pragma omp parallel for reduction(+:result) num_threads(threadnum)
    for (int i = 0; i < N; i++) {
        result += a[i] * b[i];
    }
    return result;
}
//���²в�
vector<double>& update_cancha(vector<vector<double> >& a, vector<double>& x, vector<double>& b) {
    double temp = 0;
    for (int i = 0; i < N; i++) {
        temp = 0;
#pragma omp parallel for reduction(+:temp) num_threads(threadnum)
        for (int j = 0; j < N; j++) {
            temp += a[i][j] * x[j];
        }
        r[i] = temp - b[i];
    }
    return r;
}
int main(int argv, char** args) {
    //��ȡ�߳���
    string s = args[1];
    threadnum = stoi(s);
    //��ʼ��ʱ
    timeval  start, end;
    gettimeofday(&start, NULL);
    //��ʼ��A
#pragma omp parallel for num_threads(threadnum)
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
    //��ʼ����
    int count = 0;
    for (int i = 0; i < N; i++) {
        count++;
        //����в�����r
        double before_r = compute_neiji(r, r);
        r = update_cancha(A, x, b);
        //���㷽������d
        double now_r = compute_neiji(r, r);
        if (now_r < 0.000001) {
            break;
        }
        double temp = now_r / before_r;
#pragma omp parallel for num_threads(threadnum)
        for (int j = 0; j < N; j++) {
            d[j] = -r[j] + temp * d[j];
        }
        //���㲽��length
        double num = compute_neiji(d, r);
        double temp1 = 0;
        for (int i = 0; i < N; i++) {
            temp = 0;
#pragma omp parallel for reduction(+:temp) num_threads(threadnum)
            for (int j = 0; j < N; j++) {
                temp += d[j] * A[i][j];
            }
            temp1 += temp * d[i];
        }
        double  length = -num / temp1;
        //���½�����x
#pragma omp parallel for num_threads(threadnum)
        for (int j = 0; j < N; j++) {
            x[j] = x[j] + length * d[j];
        }
    }
    gettimeofday(&end, NULL);
    double run_time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    run_time = run_time / 1000000;
    //cout<<"�����"<<endl;
    //print_b(x);
    cout << "�߳���: " << threadnum << endl;
    cout << "ʱ��: " << run_time << " s" << endl;
    cout << "��������: " << count << endl;
    return 0;
}
