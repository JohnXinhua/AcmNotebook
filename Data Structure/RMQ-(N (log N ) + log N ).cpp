#include<stdio.h>
#include<algorithm>
#include<iostream>
#include<vector>
#include<math.h>

#define MAX 1200

using namespace std;

int N,value[MAX],M[MAX][MAX];

void build_RMQ()
{
    for(int i=0;i<N;i++) M[i][0]=i;
    for(int j=1;(1<<j)<=N;j++)
    {
        for(int i=0;i+(1<<j)-1<N;i++)
        {
            if(value[M[i][j-1]]<value[M[i+(1<<(j-1))-1][j-1]]) M[i][j]=M[i][j-1];
            else M[i][j]=M[i+(1<<(j-1))-1][j-1];
        }
    }
    return;
}

int RMQ_quary(int i,int j)
{
    int k=log2(j-i+1);
    if(value[M[i][k]]<=value[M[j-(1<<k)+1][k]]) return M[i][k];
    return M[j-(1<<k)+1][k];
}

int main()
{
    while(cin>>N and N)
    {
        for(int i=0;i<N;i++ ) cin>>value[i];
        build_RMQ();

        int q,x,y;
        cin>>q;

        while(q--)
        {
            cin>>x>>y;
            int cnt=RMQ_quary(x,y);
            cout<<"Ans: "<<cnt<<endl;
        }
    }
    return 0;
}


/***
10
2 4 3 1 6 7 8 9 1 7

***/
