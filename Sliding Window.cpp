/**
Input:
The first line contains two integers n and k
which are the lengths of the array and the sliding window.
There are n integers in the second line.

Output:
There are two lines in the output.
The first line gives the minimum values in the window at each position, from left to right, respectively.
The second line gives the maximum values.

*/

#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<vector>
#include<algorithm>
typedef long long ll;
#define mem(a,b) memset(a,b,sizeof(a))
#define clr(a) a.clear()
#define mp make_pair
#define pb push_back
#define SZ(a) (int)a.size()
using namespace std;

const int INF=(1<<29);
const int MAX=1000009;

int store[MAX];
int Q[MAX],head,tail;
vector<int>MinValu,MaxValu;

void SlidingWindowMin(int N,int K)
{
    //N=number of elements.
    //K=windows size.
    MinValu.clear();
    head=0;tail=0;
    for(int i=1;i<=N;i++)
    {
        while(head!=tail && Q[head]<=i-K) head++;// Cheack window size.
        while(head!=tail && store[Q[tail-1]]>=store[i]) tail--; // truncate the previous bigger value.
        Q[tail++]=i; // insert new elements.
        if(i>=K) MinValu.pb(store[Q[head]]); // store valu aeach posirion of window.
    }
    return;
}

void SlidingWindowMax(int N,int K)
{
    //N=number of elements.
    //K=windows size.
    MaxValu.clear();
    head=0;tail=0;
    for(int i=1;i<=N;i++)
    {
        while(head!=tail && Q[head]<=i-K) head++;// Cheack window size.
        while(head!=tail && store[Q[tail-1]]<=store[i]) tail--; // truncate the previous smaller value.
        Q[tail++]=i; // insert new elements.
        if(i>=K) MaxValu.pb(store[Q[head]]); // store valu aeach posirion of window.
    }
    return;
}

int main()
{
    int N,K;
    //N=number of elements.
    //K=windows size.
    while(scanf("%d %d",&N,&K)==2)
    {
        for(int i=1;i<=N;i++) scanf("%d",&store[i]);

        SlidingWindowMin(N,K); // Min value in given Window.
        SlidingWindowMax(N,K); // Max value in given Window.

        printf("%d",MinValu[0]);
        for(int i=1;i<MinValu.size();i++) printf(" %d",MinValu[i]);
        puts("");

        printf("%d",MaxValu[0]);
        for(int i=1;i<MaxValu.size();i++) printf(" %d",MaxValu[i]);
        puts("");
    }
    return 0;
}

