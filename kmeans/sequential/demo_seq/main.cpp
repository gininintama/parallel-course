#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<ctime>
#include<algorithm>
#include<math.h>
#include <windows.h>
using namespace std;
#define R 100
//input�ļ��������ݼ���output�ļ��洢���
ifstream fin("C:\\Users\\�վ��ĵ���\\Desktop\\kmeans\\demo\\input1.txt");
ofstream fout("C:\\Users\\�վ��ĵ���\\Desktop\\kmeans\\demo\\output1.txt");
//�ֱ��ʾ�ء����ĵ㡢������������
vector< pair<double, double> > clusters[100];
vector< pair<double, double> > centroids;
vector< pair<double, double> > points;
vector<int> final_result;
//��������
int iters = 10000000;
//��������������ŷʽ����
double dist(pair<double, double> pa1, pair<double, double> pa2) {
    return (pa1.first - pa2.first) * (pa1.first - pa2.first) +
        (pa1.second - pa2.second) * (pa1.second - pa2.second);
}
//N��������K����
void compute(int N, int K) {
    int i, j, cl, k, l;
    double meanx, meany;
    double Min;
    //����һ�ε���
    for (i = 0; i < iters; i++) {
        //����N�����������վ������ĵ���С��ԭ�򣬰�ÿ����ָ�����������ĵ����ڵĴ�
        for (j = 0; j < N; j++) {
            cl = -1;
            Min = 100000000;
            //����N������������ÿ����points[j]���ҵ����Min�����ĵ����ڵĴغ�cl
            for (k = 0; k < K; k++) {
                if (Min > dist(points[j], centroids[k])) {
                    Min = dist(points[j], centroids[k]);
                    cl = k;
                }
            }
            //��points[j]�����cl��������
            if (cl != -1) {
                clusters[cl].push_back(points[j]);
            }
        }
        //����K���أ�����ÿ�����е����ɸ��۲⣬��������������ľ�ֵ����Ϊ��һ�ε��������ĵ�
        for (j = 0; j < K; j++) {
            meanx = 0;
            meany = 0;
            //�������е��������������ñ���������������ֵ�ĺ�
            for (l = 0; l < clusters[j].size(); l++) {
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
        //������һ�ε������ظ����ߵĲ�����ֱ���ﵽָ���ĵ�������iters
    }
    //���������������Ѿ�����������ĵ����ڵĴ���Ϊ���ս�����
    for (i = 0; i < N; i++) {
        cl = -1;
        Min = 100000000;
        //����N������������ÿ����points[i]���ҵ����Min�����ĵ����ڵĴغ�cl
        for (j = 0; j < K; j++)
            if (Min > dist(points[i], centroids[j])) {
                Min = dist(points[i], centroids[j]);
                cl = j;
            }
        //���ս��������final_result��
        final_result[i] = cl;
    }
}

int main() {
    int N, K;
    srand(time(NULL));
    long long head, tail, freq;
    //����K��������N��������
    fin >> K;
    fin >> N;
    int i;
    //�������ݼ�
    for (i = 0; i < N; i++) {
        double a, b;
        fin >> a >> b;
        points.push_back(make_pair(a, b));
    }
    //���ѡȡK������Ϊ��ʼ���ĵ�
    for (i = 0; i < K; i++)
        centroids.push_back(points[rand() % N]);
    //������������Ĵ�С = ��������N
    final_result.resize(N);
    //kmeans�㷨���壬ͬʱwindows�¾�׼��ʱ
    long long result=0;
    for(int i=0;i<R;i++){
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    compute(N, K);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    result+=(tail - head) * 1000;
    }
    cout<<"Sequential kmeans: "<<(double) result / (freq*R) <<" ms\n";
    //������浽output�ļ�
    for (int i = 0; i < N; i++)
        fout << final_result[i] << '\n';
    fout.close();
    return 0;
}
