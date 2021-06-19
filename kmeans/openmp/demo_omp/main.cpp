#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<ctime>
#include<algorithm>
#include<math.h>
#include <windows.h>
#include <omp.h>
using namespace std;
//input文件保存数据集，output文件存储结果
ifstream fin("C:\\Users\\颜君的电脑\\Desktop\\kmeans\\demo\\input1.txt");
ofstream fout("C:\\Users\\颜君的电脑\\Desktop\\kmeans\\demo\\output1.txt");
//分别表示簇、中心点、样本点和最后结果
vector< pair<double, double> > clusters[100];
vector< pair<double, double> > centroids;
vector< pair<double, double> > points;
vector<int> final_result;
omp_lock_t Mutex[100];
//迭代次数
int iters = 100;
static int num_thread=4;
#define R 10
//计算两个样本的欧式距离
double dist(pair<double, double> pa1, pair<double, double> pa2) {
    return (pa1.first - pa2.first) * (pa1.first - pa2.first) +
        (pa1.second - pa2.second) * (pa1.second - pa2.second);
}
//N个样本，K个簇
void compute(int N, int K) {
    int i, j, cl, k, l;
    double meanx, meany;
    double Min;
    //开启一次迭代
    for (i = 0; i < iters; i++) {
        //遍历N个样本，按照距离中心点最小的原则，把每个点分给距离近的中心点所在的簇
        //omp_set_num_threads(num_thread);
        #pragma omp parallel shared(points, centroids, clusters, i, Mutex, N, K) private(j, Min, cl, k, l, meanx, meany)num_threads(num_thread)
        {
            //cout<<omp_get_num_threads();
            #pragma omp for
            for (j = 0; j < N; j++) {
                cl = -1;
                Min = 100000000;
                //遍历N个样本，对于每个点points[j]，找到最近Min的中心点所在的簇号cl
                for (k = 0; k < K; k++) {
                    if (Min > dist(points[j], centroids[k])) {
                        Min = dist(points[j], centroids[k]);
                        cl = k;
                    }
                }
                //把points[j]加入簇cl的容器中
                if (cl != -1) {
                    omp_set_lock(&Mutex[cl]);
                    clusters[cl].push_back( points[j] );
                    omp_unset_lock(&Mutex[cl]);
                }
            }
            #pragma omp barrier
            #pragma omp for
            //遍历K个簇，对于每个簇中的若干个观测，计算所有样本点的均值，作为下一次迭代的中心点
            for (j = 0; j < K; j++) {
                meanx = 0;
                meany = 0;
                //遍历簇中的所有样本，先用变量保存所有坐标值的和
                for (l = 0; l < clusters[j].size(); l++) {
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
            //开启下一次迭代，重复上诉的操作，直到达到指定的迭代次数iters
            #pragma omp barrier
        }
    }
    //omp_set_num_threads(num_thread);
    //遍历所有样本，把距离最近的中心点所在的簇作为最终结果输出
    #pragma omp parallel shared(points, centroids, final_result, N, K) private(i, j, Min, cl)num_threads(num_thread)
    {
        //cout<<omp_get_num_threads();
        #pragma omp for
        for (i = 0; i < N; i++) {
            cl = -1;
            Min = 100000000;
            //遍历N个样本，对于每个点points[i]，找到最近Min的中心点所在的簇号cl
            for (j = 0; j < K; j++)
                if (Min > dist(points[i], centroids[j])) {
                    Min = dist(points[i], centroids[j]);
                    cl = j;
                }
            //最终结果保存在final_result中
            final_result[i] = cl;
        }
    }
}

void compute1(int N, int K) {
    int i, j, cl, k, l;
    double meanx, meany;
    double Min;
    //开启一次迭代
    for (i = 0; i < iters; i++) {
        //遍历N个样本，按照距离中心点最小的原则，把每个点分给距离近的中心点所在的簇
        #pragma omp parallel num_threads(num_thread)
        {
            #pragma omp for
            for (j = 0; j < N; j++) {
                cl = -1;
                Min = 100000000;
                //遍历N个样本，对于每个点points[j]，找到最近Min的中心点所在的簇号cl
                for (k = 0; k < K; k++) {
                    if (Min > dist(points[j], centroids[k])) {
                        Min = dist(points[j], centroids[k]);
                        cl = k;
                    }
                }
                //把points[j]加入簇cl的容器中
                if (cl != -1) {
                    omp_set_lock(&Mutex[cl]);
                    clusters[cl].push_back( points[j] );
                    omp_unset_lock(&Mutex[cl]);
                }
            }
            #pragma omp barrier
            #pragma omp for
            //遍历K个簇，对于每个簇中的若干个观测，计算所有样本点的均值，作为下一次迭代的中心点
            for (j = 0; j < K; j++) {
                meanx = 0;
                meany = 0;
                //遍历簇中的所有样本，先用变量保存所有坐标值的和
                for (l = 0; l < clusters[j].size(); l++) {
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
            //开启下一次迭代，重复上诉的操作，直到达到指定的迭代次数iters
            #pragma omp barrier
        }
    }
    //遍历所有样本，把距离最近的中心点所在的簇作为最终结果输出
    #pragma omp parallel num_threads(num_thread)
    {
        #pragma omp for
        for (i = 0; i < N; i++) {
            cl = -1;
            Min = 100000000;
            //遍历N个样本，对于每个点points[i]，找到最近Min的中心点所在的簇号cl
            for (j = 0; j < K; j++)
                if (Min > dist(points[i], centroids[j])) {
                    Min = dist(points[i], centroids[j]);
                    cl = j;
                }
            //最终结果保存在final_result中
            final_result[i] = cl;
        }
    }
}

int main() {
    int N, K;
    srand(time(NULL));
    long long head, tail, freq;
    //输入K：簇数、N：样本数
    fin >> K;
    fin >> N;
    int i;
    //输入数据集
    for (i = 0; i < N; i++) {
        double a, b;
        fin >> a >> b;
        points.push_back(make_pair(a, b));
    }
    //随机选取K个点作为初始中心点
    for (i = 0; i < K; i++)
        centroids.push_back(points[rand() % N]);
    for (int i = 0; i < K; i++)
        omp_init_lock(&Mutex[i]);
    //调整结果容器的大小 = 样本个数N
    final_result.resize(N);
    //kmeans算法主体，同时windows下精准计时
    long long result=0;
    for(int i=0;i<R;i++){
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    compute(N, K);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    result+=(tail - head) * 1000;
    }
    cout<<"Openmp kmeans: "<<(double) result / (freq*R) <<" ms\n";
    for (int i = 0; i < K; i++)
        omp_destroy_lock(&Mutex[i]);
    //结果保存到output文件
    for (int i = 0; i < N; i++)
        fout << final_result[i] << '\n';
    fout.close();
    return 0;
}
