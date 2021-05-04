#include <iostream>
#include <stdlib.h>
#include <windows.h>
using namespace std;

const int n=1920000;
double sum=0.0;
double a[n];
long long head,tail,freq;
long long line[5000],line2[5000],recursion[5000];

void init()
{
    for(int i=0;i<n;i++)
        a[i]=i;
     sum=0.0;
}

double average(long long a[],double b)
{
    double tmp=0;
    for(int i=0;i<b;i++)
        tmp+=a[i];
    return tmp/b;
}

void f1()
{
    for(int i=0;i<n;i++){
            sum+=a[i];
    }
}
void f2()
{
   for(int i=0;i<n;i+=2){
            sum+=a[i]+a[i+1];
    }
}
void f3(int n)
{
    for(int i=0;i<n;i+=3){
            sum+=a[i]+a[i+1]+a[i+2];
    }
}
int main()
{
    cout<<"N= "<<n<<endl;
    init();
    for(int step=0;step<5000;step++)
    {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f1();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    line[step]=(tail-head)*1000.0/freq;
    }
    init();
    for(int step=0;step<5000;step++)
    {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f2();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    line2[step]=(tail-head)*1000.0/freq;
    }
    init();
    for(int step=0;step<5000;step++)
    {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    f3(n);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    recursion[step]=(tail-head)*1000.0/freq;
    }
    cout<<"f1: "<<average(line,5000.0)<<"ms"<<endl;
    cout<<"f2: "<<average(line2,5000.0)<<"ms"<<endl;
    cout<<"f3: "<<average(recursion,5000.0)<<"ms"<<endl;
    return 0;
}
