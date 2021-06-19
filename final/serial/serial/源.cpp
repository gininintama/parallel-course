#include<iostream>
#include<vector>
#include<cmath>
#include<stdio.h>
#include "omp.h"
#include<sys/time.h>
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
double compute_neiji(vector<double>& a, vector<double>& b) {
    double result = 0;
    for (int i = 0; i < N; i++) {
        result += a[i] * b[i];
    }
    return result;
}
//更新残差
vector<double>& update_cancha(vector<vector<double> >& a, vector<double>& x, vector<double>& b) {
    double temp = 0;
    for (int i = 0; i < N; i++) {
        temp = 0;
        
        for (int j = 0; j < N; j++) {
            temp += a[i][j] * x[j];
        }
        r[i] = temp - b[i];
    }
    return r;
}
int main() {
    //初始化A
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
    //开始计时
    timeval  start, end;
    gettimeofday(&start, NULL);
    //开始迭代
    int count = 0;
    for (int i = 0; i < N; i++) {
        count++;
        //计算残差向量r
        double before_r = compute_neiji(r, r);
        r = update_cancha(A, x, b);
        //计算方向向量d
        double now_r = compute_neiji(r, r);
        if (now_r < 0.000001) {
            break;
        }
        double temp = now_r / before_r;
        for (int j = 0; j < N; j++) {
            d[j] = -r[j] + temp * d[j];
        }
        //计算步长length
        double num = compute_neiji(d, r);
        double temp1 = 0;
        for (int i = 0; i < N; i++) {
            temp = 0;
            for (int j = 0; j < N; j++) {
                temp += d[j] * A[i][j];
            }
            temp1 += temp * d[i];
        }
        double  length = -num / temp1;
        //更新解向量x
        for (int j = 0; j < N; j++) {
            x[j] = x[j] + length * d[j];
        }
    }
    //结束计时
    gettimeofday(&end, NULL);
    double run_time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    run_time = run_time / 1000000;
    //cout<<"结果："<<endl;
    //print_b(x);
    cout << "迭代次数：" << count << endl;
    cout << "时间：" << run_time << " s" << endl;
    return 0;
}
