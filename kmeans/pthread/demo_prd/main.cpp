//pthread
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include <algorithm>
#include <pthread.h>
#include<fstream>
#include<vector>
using namespace std;

ifstream fin("C:\\Users\\颜君的电脑\\Desktop\\kmeans\\input1.txt");
ofstream fout("C:\\Users\\颜君的电脑\\Desktop\\kmeans\\output1.txt");
pthread_barrier_t barrier1, barrier2;
pthread_mutex_t Mutex[100];
//分别表示簇、中心点、样本点和最后结果
vector< pair<double, double> > clusters[100];
vector< pair<double, double> > centroids;
vector< pair<double, double> > points;
vector<int> final_result;
//迭代次数
int iters = 100;
//线程数
const int nthreads = 1;
#define R 1000
//自定义结构体my_thread传参
struct my_thread {
    int tid, K;
    int left, right;
    int cluster_left, cluster_right;
};
//计算两个样本的欧式距离
double dist(pair<double, double> pa1, pair<double, double> pa2) {
    return (pa1.first - pa2.first) * (pa1.first - pa2.first) +
        (pa1.second - pa2.second) * (pa1.second - pa2.second);
}
//kmeans函数主体
void* kFunc(void* data) {
    my_thread t = *((my_thread*)data);
    //开启一次遍历
    for (int st = 0; st < iters; st++) {
        //遍历C个样本，按照距离中心点最小的原则，把每个点分给距离近的中心点所在的簇
        for (int i = t.left; i < t.right; i++) {
            int cl = -1;
            double Min = 100000000;
            //遍历C个样本，对于每个点points[j]，找到最近Min的中心点所在的簇号cl
            for (int j = 0; j < t.K; j++)
                if (Min > dist(points[i], centroids[j])) {
                    Min = dist(points[i], centroids[j]);
                    cl = j;
                }
            //把points[i]加入簇cl的容器中
            if (cl != -1) {
                pthread_mutex_lock(&Mutex[cl]);
                clusters[cl].push_back(points[i]);
                pthread_mutex_unlock(&Mutex[cl]);
            }
        }

        pthread_barrier_wait(&barrier1);

        if (t.cluster_left != -1) {
            //遍历C_cluster个簇，对于每个簇中的若干个观测，计算所有样本点的均值，作为下一次迭代的中心点
            for (int j = t.cluster_left; j < t.cluster_right; j++) {
                double meanx = 0, meany = 0;
                //遍历簇中的所有样本，先用变量保存所有坐标值的和
                for (int l = 0; l < clusters[j].size(); l++) {
                    meanx += clusters[j][l].first;
                    meany += clusters[j][l].second;
                }
                //再求各个坐标均值，然后均值作为新的迭代中心
                if (clusters[j].size() != 0) {
                    centroids[j].first = meanx / clusters[j].size();
                    centroids[j].second = meany / clusters[j].size();
                }
                //簇包含的样本可能改变，需要把簇的大小清零
                clusters[j].clear();
            }
        }
        pthread_barrier_wait(&barrier2);
    }

    return NULL;
}
//结果保存函数
void* fFunc(void* data) {
    my_thread t = *((my_thread*)data);

    for (int i = t.left; i < t.right; i++) {
        int cl = -1;
        double Min = 100000000;

        for (int j = 0; j < t.K; j++)
            if (Min > dist(points[i], centroids[j])) {
                Min = dist(points[i], centroids[j]);
                cl = j;
            }

        final_result[i] = cl;
    }
    return NULL;
}

int main() {
    int N, K;
    srand(time(NULL));
    long long head, tail, freq;
    //输入K：簇数、N：样本数
    fin >> K;
    fin >> N;
    //输入数据集
    for (int i = 0; i < N; i++) {
        double a, b;
        fin >> a >> b;
        points.push_back(make_pair(a, b));
    }
    //随机选取K个点作为初始中心点
    for (int i = 0; i < K; i++)
        centroids.push_back(points[rand() % N]);
    //传递参数数组thr_str
    my_thread thr_str[nthreads];
    //每个线程处理C个样本，包含C_cluster个簇
    int C = N / nthreads;
    int C_cluster = K / nthreads;
    //给每个线程的参数进行赋值
    for (int i = 0; i < nthreads; i++) {
        //线程id和簇数K
        thr_str[i].tid = i;
        thr_str[i].K = K;
        //处理样本的起点和终点
        thr_str[i].left = i * C;
        thr_str[i].right = (i + 1) * C;
        //处理簇的起点和终点
        thr_str[i].cluster_left = thr_str[i].cluster_right = -1;
        if (C_cluster == 0 && i < K) {
            thr_str[i].cluster_left = i;
            thr_str[i].cluster_right = i + 1;
        }
        else if (C_cluster > 0) {
            thr_str[i].cluster_left = i * C_cluster;
            thr_str[i].cluster_right = (i + 1) * C_cluster;
        }
    }
    //最后处理样本的终点 = 样本数N
    thr_str[nthreads - 1].right = N;
    //最后处理簇的终点 = 簇数K
    if (C_cluster > 0)
        thr_str[nthreads - 1].cluster_right = K;
    //线程数组t
    pthread_t t[nthreads];
    //barrier和mutex初始化
    pthread_barrier_init(&barrier1, NULL, nthreads);
    pthread_barrier_init(&barrier2, NULL, nthreads);
    for (int i = 0; i < K; i++)
        pthread_mutex_init(&Mutex[i], NULL);
    //调整结果容器的大小 = 样本个数N
    final_result.resize(N);
    //kmeans算法，同时windows下精准计时
    long long result=0;
    for(int i=0;i<R;i++){
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //kmeans主体
    for (int i = 0; i < nthreads; i++)
        pthread_create(&(t[i]), NULL, kFunc, &(thr_str[i]));
    for (int i = 0; i < nthreads; i++)
        pthread_join(t[i], NULL);
    //最终结果保存
    for (int i = 0; i < nthreads; i++)
        pthread_create(&(t[i]), NULL, fFunc, &(thr_str[i]));
    for (int i = 0; i < nthreads; i++)
        pthread_join(t[i], NULL);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
     result+=(tail - head) * 1000;
    }
    cout<<"Phread kmeans: "<<(double) result / (freq*R) <<" ms\n";
    //结果保存到output文件
    for (int i = 0; i < N; i++)
        fout << final_result[i] << '\n';
    fout.close();
    //barrier和mutex的销毁
    for (int i = 0; i < K; i++)
        pthread_mutex_destroy(&Mutex[i]);
    pthread_barrier_destroy(&barrier1);
    pthread_barrier_destroy(&barrier2);

    return 0;
}
