#include<iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include<cmath>
#include<sys/time.h>
#include<string>
using namespace std;

const int N = 5000;
const int global_count = 2;
//变量初始化
__global__ void initial(double* A, double* r, double* d, double* x, double* b, double* Ad, double* result, double* global_variable, int* C) {
    int THREAD_data = C[1];
    int BLOCK_data = C[0];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[N * i + j] = 2;
            }
            else if (abs(i - j) == 1) {
                A[N * i + j] = -1;
            }
            else {
                A[N * i + j] = 0;
            }
        }
        r[i] = 2;
        d[i] = 0;
        x[i] = 0;
        b[i] = 1;
        Ad[i] = 0;
    }

    for (int i = 0; i < THREAD_data * BLOCK_data; i++) {
        result[i] = 0;
    }
    for (int i = 0; i < global_count; i++) {
        result[i] = 0;
    }
}
//验证结果
void print_b(double* b) {
    for (int i = 0; i < N; i++) {
        cout << b[i] << " ";
    }
    cout << endl;
}
//计算内积
__global__
static void compute_neiji(double* r, double* b, double* result, int* C) {
    int THREAD_data = C[1];
    int BLOCK_data = C[0];
    // 取得线程号
    const int thread = threadIdx.x;
    // 获得块号
    const int block = blockIdx.x;
    result[block * THREAD_data + thread] = 0;
    for (int i = block * THREAD_data + thread; i < N; i += BLOCK_data * THREAD_data) {
        result[block * THREAD_data + thread] += r[i] * b[i];
    }
}
//更新残差
__global__
static void update_cancha(double* r, double* A, double* x, double* b, int* C) {
    int THREAD_data = C[1];
    int BLOCK_data = C[0];
    double temp = 0;
    // 取得线程号
    const int thread = threadIdx.x;
    // 获得块号
    const int block = blockIdx.x;
    for (int i = block * THREAD_data + thread; i < N; i += BLOCK_data * THREAD_data) {
        temp = 0;
        for (int j = 0; j < N; j++) {
            temp += A[i * N + j] * x[j];
        }
        r[i] = temp - b[i];
    }
}
//计算中间变量
//d = -r +now_r/before_r*d
__global__
static void update_d(double* d, double* r, double* global_variable, int* C) {
    int THREAD_data = C[1];
    int BLOCK_data = C[0];
    // 取得线程号
    const int thread = threadIdx.x;
    // 获得块号
    const int block = blockIdx.x;
    for (int i = block * THREAD_data + thread; i < N; i += BLOCK_data * THREAD_data) {
        d[i] = -r[i] + global_variable[0] * d[i];
    }
}
//A*d
__global__
static void A_multiply_d(double* A, double* d, double* Ad, int* C) {
    int THREAD_data = C[1];
    int BLOCK_data = C[0];
    // 取得线程号
    const int thread = threadIdx.x;
    // 获得块号
    const int block = blockIdx.x;
    for (int i = block * THREAD_data + thread; i < N; i += BLOCK_data * THREAD_data) {
        Ad[i] = 0;,
        for (int j = 0; j < N; j++) {
            Ad[i] += d[j] * A[i * N + j];
        }
    }
}
//更新解向量
__global__
static void update_x(double* x, double* d, double* global_variable, int* C) {
    int THREAD_data = C[1];
    int BLOCK_data = C[0];
    // 取得线程号
    const int thread = threadIdx.x;
    // 获得块号
    const int block = blockIdx.x;
    for (int i = block * THREAD_data + thread; i < N; i += BLOCK_data * THREAD_data) {
        x[i] = x[i] + global_variable[1] * d[i];
    }
}
int main(int argc, char** argv)
{
    string s1 = argv[1];
    string s2 = argv[2];
    int BLOCK_data = stoi(s1);
    int THREAD_data = stoi(s2);
    //申请显存并进行初始化，统一寻址，cpu可以直接访问显存
    double* A_gpu;
    cudaMallocManaged(&A_gpu, N * N * sizeof(double)); 
    double* r_gpu;
    cudaMallocManaged(&r_gpu, N * sizeof(double));
    double* d_gpu;
    cudaMallocManaged(&d_gpu, N * sizeof(double));
    double* x_gpu;
    cudaMallocManaged(&x_gpu, N * sizeof(double));
    double* b_gpu;
    cudaMallocManaged(&b_gpu, N * sizeof(double));
    double* Ad_gpu;
    cudaMallocManaged(&Ad_gpu, N * sizeof(double));
    double* result_gpu;
    cudaMallocManaged(&result_gpu, BLOCK_data * THREAD_data * sizeof(double));
    double* global_variable_gpu;
    cudaMallocManaged(&global_variable_gpu, 2 * sizeof(double));
    int* COUNT;
    cudaMallocManaged(&COUNT, 2 * sizeof(int));
    COUNT[0] = BLOCK_data;
    COUNT[1] = THREAD_data;
    initial << <1, 1 >> > (A_gpu, r_gpu, d_gpu, x_gpu, b_gpu, Ad_gpu, result_gpu, global_variable_gpu, COUNT);
    cudaDeviceSynchronize();
    //gpu上进行计算
    timeval  start, end;
    gettimeofday(&start, NULL);
    double before_r = 0;
    double now_r = 0;
    double num = 0;
    double temp1 = 0;
    double length = 0;
    int count = 0;//迭代次数
    for (int i = 0; i < N; i++) {
        count++;
        //计算残差向量r
        before_r = 0;
        compute_neiji << <BLOCK_data, THREAD_data >> > (r_gpu, r_gpu, result_gpu, COUNT);
        cudaDeviceSynchronize();
        for (int i = 0; i < THREAD_data * BLOCK_data; i++) {
            before_r += result_gpu[i];
        }
        update_cancha << <BLOCK_data, THREAD_data >> > (r_gpu, A_gpu, x_gpu, b_gpu, COUNT);
        cudaDeviceSynchronize();
        // 计算方向向量d
        //计算now_r
        now_r = 0;
        compute_neiji << <BLOCK_data, THREAD_data >> > (r_gpu, r_gpu, result_gpu, COUNT);
        cudaDeviceSynchronize();
        for (int i = 0; i < THREAD_data * BLOCK_data; i++) {
            now_r += result_gpu[i];
        }
        if (now_r < 0.00001) {
            break;
        }
        global_variable_gpu[0] = now_r / before_r;
        update_d << <BLOCK_data, THREAD_data >> > (d_gpu, r_gpu, global_variable_gpu, COUNT);
        cudaDeviceSynchronize();
        //计算步长length
        num = 0;
        compute_neiji << <BLOCK_data, THREAD_data >> > (r_gpu, d_gpu, result_gpu, COUNT);
        cudaDeviceSynchronize();
        for (int i = 0; i < THREAD_data * BLOCK_data; i++) {
            num += result_gpu[i];
        }
        temp1 = 0;
        A_multiply_d << <BLOCK_data, THREAD_data >> > (A_gpu, d_gpu, Ad_gpu, COUNT);
        compute_neiji << <BLOCK_data, THREAD_data >> > (d_gpu, Ad_gpu, result_gpu, COUNT);
        cudaDeviceSynchronize();
        for (int i = 0; i < THREAD_data * BLOCK_data; i++) {
            temp1 += result_gpu[i];
        }
        length = -num / temp1;
        global_variable_gpu[1] = length;
        //更新解向量x
        update_x << <BLOCK_data, THREAD_data >> > (x_gpu, d_gpu, global_variable_gpu, COUNT);
        cudaDeviceSynchronize();
    }
    //结束计算
    gettimeofday(&end, NULL);
    double run_time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    run_time = run_time / 1000000;
    //cout<<"结果："<<endl;
    //print_b(x_gpu);
    cout << "迭代次数：" << count << endl;
    cout << "线程数：" << BLOCK_data << "*" << THREAD_data << "=" << THREAD_data * BLOCK_data << endl;
    cout << "时间: " << run_time << " s" << endl;
    //释放申请的显存
    cudaFree(A_gpu);
    cudaFree(r_gpu);
    cudaFree(d_gpu);
    cudaFree(x_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);
    cudaFree(global_variable_gpu);
    return 0;
}