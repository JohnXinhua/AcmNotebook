#include<stdio.h>
#include<vector>
#include<iostream>

using namespace std;

int dp[120],res[120],n;
bool col[120];

void print()
{
    for(int i=0;i<n;i++) cout<<res[i]<<"  ";
    cout<<endl;
}

void permutation(int cur)
{
    if(cur==n){print();return;}

    for(int i=0;i<n;i++)
    {
        if(!col[i])
        {
            if(i>0) if(dp[i]==dp[i-1] && col[i-1]==0) continue;
            col[i]=1;
            res[cur]=dp[i];
            permutation(cur+1);
            col[i]=0;
        }
    }
    return;
}

int main()
{
    int i,k;
    while(scanf("%d",&n)==1)
    {
        for(i=0;i<n;i++)scanf("%d",&dp[i]);
        permutation(0);
    }
    return 0;
}

