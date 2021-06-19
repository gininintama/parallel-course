#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<ctime>
#include<algorithm>
#include<math.h>
#include <windows.h>
#include<mpi.h>
#include<time.h>
using namespace std;

const int iters = 100;

double dist(pair<double, double> pa1, pair<double, double> pa2) {
    return (pa1.first - pa2.first) * (pa1.first - pa2.first) +
        (pa1.second - pa2.second) * (pa1.second - pa2.second);
}

int main(int argc, char **argv) {
    long long head, tail, freq;
    MPI_Init(&argc, &argv);

    int rank, nrprocs;
    MPI_Status status;

    int N, K, rc;
    srand(time(NULL));

    rc = MPI_Comm_size(MPI_COMM_WORLD, &nrprocs);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector< pair<double, double> > centroids;
    vector< pair<double, double> > points;
    vector<int> final_result;
    double time;
    time = MPI_Wtime();
    //QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    //QueryPerformanceCounter((LARGE_INTEGER*)&head);
    if (rank == 0) {
        ifstream fin("C:\\Users\\颜君的电脑\\Desktop\\kmeans\\demo\\input.txt");
        ofstream fout("C:\\Users\\颜君的电脑\\Desktop\\kmeans\\demo\\output.txt");
        //输入K：簇数、N：样本数
        fin >> K;
        fin >> N;
        for (int i = 0; i < N; i++) {
            double a, b;
            fin >> a >> b;
            points.push_back(make_pair(a, b));
        }
        for (int i = 0; i < K; i++)
            centroids.push_back(points[rand() % N]);
        fin.close();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        // Send initial data
        for (int i = 1; i < nrprocs; i++) {
            MPI_Send(&N, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&K, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        int C = N / nrprocs;

        int size = C;
        if (size < K)
            size = K;

        if (size < N - (nrprocs - 1) * C) {
            size = N - (nrprocs - 1) * C;
        }

        double *ref = (double*)malloc(size * 2 * sizeof(double));

        for (int i = 1; i < nrprocs-1; i++) {
            int l = 0;
            for (int j = i * C; j < (i+1) * C; j++) {
                ref[2 * l] = points[j].first;
                ref[2 * l + 1] = points[j].second;
                l++;
            }
            MPI_Send(ref, 2 * C, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        int l = 0;
        for (int i = (nrprocs-1) * C; i < N; i++) {
            ref[2 * l] = points[i].first;
            ref[2 * l + 1] = points[i].second;
            l++;
        }

        if (nrprocs > 1)
            MPI_Send(ref, 2*(N - (nrprocs - 1) * C), MPI_DOUBLE, nrprocs-1, 0, MPI_COMM_WORLD);
        for (int i = 0; i < K; i++) {
            ref[2 * i] = centroids[i].first;
            ref[2 * i + 1] = centroids[i].second;
        }

        for (int i = 1; i < nrprocs; i++) {
            MPI_Send(ref, 2 * K, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        int cl;
        double Min, meanx[K], meany[K], counts[K], tosend[3 * K];

        // Process data
        for (int i = 0; i < iters; i++) {
            for (int j = 0; j < K; j++) {
                meanx[j] = 0;
                meany[j] = 0;
                counts[j] = 0;
            }
            for (int j = 0; j < C; j++) {
                cl = -1;
                Min = 100000000;

                for (int l = 0; l < K; l++)
                    if (Min > dist(centroids[l], points[j])) {
                        Min = dist(centroids[l], points[j]);
                        cl = l;
                    }

                if (cl != -1) {
                    meanx[cl] += points[j].first;
                    meany[cl] += points[j].second;
                    counts[cl]++;
                }
            }

            for (int j = 1; j < nrprocs; j++) {
                MPI_Recv(tosend, 3 * K, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int l = 0; l < K; l++)
                    if (tosend[l * 3 + 2] != 0) {
                        meanx[l] += tosend[l * 3];
                        meany[l] += tosend[l * 3 + 1];
                        counts[l] += tosend[l * 3 + 2];
                    }
            }

            for (int j = 0; j < K; j++)
                if (counts[j] != 0) {
                    centroids[j] = make_pair(meanx[j] / counts[j], meany[j] / counts[j]);
                }

            for (int j = 0; j < K; j++) {
                ref[2 * j] = centroids[j].first;
                ref[2 * j + 1] = centroids[j].second;
            }

            for (int j = 1; j < nrprocs; j++) {
                MPI_Send(ref, 2 * K, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
            }
        }

        // Establish final clusters
        //for (int i = 0; i < K; i++)
        //    cout << centroids[i].first << ' ' << centroids[i].second << '\n';

        for (int j = 0; j < K; j++) {
            ref[2 * j] = centroids[j].first;
            ref[2 * j + 1] = centroids[j].second;
        }

        for (int j = 1; j < nrprocs; j++) {
            MPI_Send(ref, 2 * K, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
        }
        for (int j = 0; j < C; j++) {
            cl = -1;
            Min = 100000000;

            for (int l = 0; l < K; l++)
                if (Min > dist(centroids[l], points[j])) {
                    Min = dist(centroids[l], points[j]);
                    cl = l;
                }
            fout << cl << '\n';
        }
        for (int j = 1; j < nrprocs-1; j++) {
            MPI_Recv(ref, C, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < C; i++)
                fout << (int)ref[i] << '\n';
        }

        if (nrprocs > 1) {
            MPI_Recv(ref, N - (nrprocs - 1) * C, MPI_DOUBLE, nrprocs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < N - (nrprocs - 1) * C; i++)
                fout << (int)ref[i] << '\n';
        }
        free(ref);
        fout.close();
        //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //cout<<"Mpi kmeans: "<<(double) (tail-head)*1000 / freq <<" ms\n";
    } else {
        // Receive initial data
        MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&K, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int C = N / nrprocs;

        int size = C;
        if (size < K)
            size = K;

        if (rank == nrprocs - 1) {
            size = N - (nrprocs - 1) * C;
        }

        double *ref = (double*)malloc(size * 2 * sizeof(double));
        int recv_size = C;

        if (rank == nrprocs - 1)
            recv_size = N - (nrprocs - 1) * C;
        MPI_Recv(ref, 2 * recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < recv_size; i++) {
            points.push_back(make_pair(ref[2 * i], ref[2 * i + 1]));
        }

        MPI_Recv(ref, 2 * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < K; i++) {
            centroids.push_back(make_pair(ref[2 * i], ref[2 * i + 1]));
        }

        int cl;
        double Min, meanx[K], meany[K], counts[K], tosend[3 * K];

        // Process data
        for (int i = 0; i < iters; i++) {
            for (int j = 0; j < K; j++) {
                meanx[j] = 0;
                meany[j] = 0;
                counts[j] = 0;
            }
            for (int j = 0; j < recv_size; j++) {
                cl = -1;
                Min = 100000000;

                for (int l = 0; l < K; l++)
                    if (Min > dist(centroids[l], points[j])) {
                        Min = dist(centroids[l], points[j]);
                        cl = l;
                    }

                if (cl != -1) {
                    meanx[cl] += points[j].first;
                    meany[cl] += points[j].second;
                    counts[cl]++;
                }
            }

            for (int j = 0; j < K; j++) {
                tosend[3*j+2] = counts[j];
                tosend[3*j] = meanx[j];
                tosend[3*j+1] = meany[j];
            }

            MPI_Send(tosend, 3 * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

            MPI_Recv(ref, 2 * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            centroids.clear();
            for (int j = 0; j < K; j++) {
                centroids.push_back(make_pair(ref[2 * j], ref[2 * j + 1]));
            }
        }

        MPI_Recv(ref, 2 * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        centroids.clear();
        for (int j = 0; j < K; j++) {
            centroids.push_back(make_pair(ref[2 * j], ref[2 * j + 1]));
        }

        // Final clusters
        for (int j = 0; j < recv_size; j++) {
            cl = -1;
            Min = 100000000;

            for (int l = 0; l < K; l++)
                if (Min > dist(centroids[l], points[j])) {
                    Min = dist(centroids[l], points[j]);
                    cl = l;
                }

            ref[j] = cl;
        }
        MPI_Send(ref, recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        free(ref);

    }
    //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //cout<<"Mpi kmeans: "<<(double) (tail-head)*1000 / freq <<" ms\n";
    time = MPI_Wtime() - time;
    cout<<"Mpi kmeans: "<<(double)time*1000 <<" ms\n";
    rc = MPI_Finalize();
    //system("pause");
    return 0;
}
