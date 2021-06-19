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

ifstream fin("C:\\Users\\�վ��ĵ���\\Desktop\\kmeans\\input1.txt");
ofstream fout("C:\\Users\\�վ��ĵ���\\Desktop\\kmeans\\output1.txt");
pthread_barrier_t barrier1, barrier2;
pthread_mutex_t Mutex[100];
//�ֱ��ʾ�ء����ĵ㡢������������
vector< pair<double, double> > clusters[100];
vector< pair<double, double> > centroids;
vector< pair<double, double> > points;
vector<int> final_result;
//��������
int iters = 100;
//�߳���
const int nthreads = 1;
#define R 1000
//�Զ���ṹ��my_thread����
struct my_thread {
    int tid, K;
    int left, right;
    int cluster_left, cluster_right;
};
//��������������ŷʽ����
double dist(pair<double, double> pa1, pair<double, double> pa2) {
    return (pa1.first - pa2.first) * (pa1.first - pa2.first) +
        (pa1.second - pa2.second) * (pa1.second - pa2.second);
}
//kmeans��������
void* kFunc(void* data) {
    my_thread t = *((my_thread*)data);
    //����һ�α���
    for (int st = 0; st < iters; st++) {
        //����C�����������վ������ĵ���С��ԭ�򣬰�ÿ����ָ�����������ĵ����ڵĴ�
        for (int i = t.left; i < t.right; i++) {
            int cl = -1;
            double Min = 100000000;
            //����C������������ÿ����points[j]���ҵ����Min�����ĵ����ڵĴغ�cl
            for (int j = 0; j < t.K; j++)
                if (Min > dist(points[i], centroids[j])) {
                    Min = dist(points[i], centroids[j]);
                    cl = j;
                }
            //��points[i]�����cl��������
            if (cl != -1) {
                pthread_mutex_lock(&Mutex[cl]);
                clusters[cl].push_back(points[i]);
                pthread_mutex_unlock(&Mutex[cl]);
            }
        }

        pthread_barrier_wait(&barrier1);

        if (t.cluster_left != -1) {
            //����C_cluster���أ�����ÿ�����е����ɸ��۲⣬��������������ľ�ֵ����Ϊ��һ�ε��������ĵ�
            for (int j = t.cluster_left; j < t.cluster_right; j++) {
                double meanx = 0, meany = 0;
                //�������е��������������ñ���������������ֵ�ĺ�
                for (int l = 0; l < clusters[j].size(); l++) {
                    meanx += clusters[j][l].first;
                    meany += clusters[j][l].second;
                }
                //������������ֵ��Ȼ���ֵ��Ϊ�µĵ�������
                if (clusters[j].size() != 0) {
                    centroids[j].first = meanx / clusters[j].size();
                    centroids[j].second = meany / clusters[j].size();
                }
                //�ذ������������ܸı䣬��Ҫ�ѴصĴ�С����
                clusters[j].clear();
            }
        }
        pthread_barrier_wait(&barrier2);
    }

    return NULL;
}
//������溯��
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
    //����K��������N��������
    fin >> K;
    fin >> N;
    //�������ݼ�
    for (int i = 0; i < N; i++) {
        double a, b;
        fin >> a >> b;
        points.push_back(make_pair(a, b));
    }
    //���ѡȡK������Ϊ��ʼ���ĵ�
    for (int i = 0; i < K; i++)
        centroids.push_back(points[rand() % N]);
    //���ݲ�������thr_str
    my_thread thr_str[nthreads];
    //ÿ���̴߳���C������������C_cluster����
    int C = N / nthreads;
    int C_cluster = K / nthreads;
    //��ÿ���̵߳Ĳ������и�ֵ
    for (int i = 0; i < nthreads; i++) {
        //�߳�id�ʹ���K
        thr_str[i].tid = i;
        thr_str[i].K = K;
        //���������������յ�
        thr_str[i].left = i * C;
        thr_str[i].right = (i + 1) * C;
        //����ص������յ�
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
    //������������յ� = ������N
    thr_str[nthreads - 1].right = N;
    //�����ص��յ� = ����K
    if (C_cluster > 0)
        thr_str[nthreads - 1].cluster_right = K;
    //�߳�����t
    pthread_t t[nthreads];
    //barrier��mutex��ʼ��
    pthread_barrier_init(&barrier1, NULL, nthreads);
    pthread_barrier_init(&barrier2, NULL, nthreads);
    for (int i = 0; i < K; i++)
        pthread_mutex_init(&Mutex[i], NULL);
    //������������Ĵ�С = ��������N
    final_result.resize(N);
    //kmeans�㷨��ͬʱwindows�¾�׼��ʱ
    long long result=0;
    for(int i=0;i<R;i++){
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    //kmeans����
    for (int i = 0; i < nthreads; i++)
        pthread_create(&(t[i]), NULL, kFunc, &(thr_str[i]));
    for (int i = 0; i < nthreads; i++)
        pthread_join(t[i], NULL);
    //���ս������
    for (int i = 0; i < nthreads; i++)
        pthread_create(&(t[i]), NULL, fFunc, &(thr_str[i]));
    for (int i = 0; i < nthreads; i++)
        pthread_join(t[i], NULL);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
     result+=(tail - head) * 1000;
    }
    cout<<"Phread kmeans: "<<(double) result / (freq*R) <<" ms\n";
    //������浽output�ļ�
    for (int i = 0; i < N; i++)
        fout << final_result[i] << '\n';
    fout.close();
    //barrier��mutex������
    for (int i = 0; i < K; i++)
        pthread_mutex_destroy(&Mutex[i]);
    pthread_barrier_destroy(&barrier1);
    pthread_barrier_destroy(&barrier2);

    return 0;
}
