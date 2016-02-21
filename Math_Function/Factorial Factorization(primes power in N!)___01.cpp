#include<stdio.h>
#include<math.h>
#include<vector>
#include<iostream>
#include<string>

using namespace std;

#define MAX 120000

bool flag[MAX];
int prime[MAX];

void sive()
{
    int i,j,k,r,total=1;
    flag[0]=flag[1]=1;
    prime[total]=2;

    for(i=3;i<MAX;i+=2)
    {
        if(!flag[i])
        {
            prime[++total]=i;
            r=i*2;
            if(MAX/i>=i)for(j=i*i;j<MAX;j+=r)flag[j]=1;
        }
    }
    return;
}

void Factorial_Factorization(vector<int> &fac,int n)  /// Find the prime power in N!
{
    int i,j,k=n;
    for(i=1;prime[i]<=k;i++)
    {
        j=0;
        while(n)
        {
            j+=(n/prime[i]);
            n/=prime[i];
        }
        fac.push_back(prime[i]);
        n=k;
    }
    return;
}

int main()
{
    sive();
    int i,j,k,n;
    vector<int>fac;
    while(true)
    {
        scanf("%d",&n);
        Factorial_Factorization(fac,n);
        printf("Factor: \n");
        for(i=0;i<fac.size();i++)
        {
            printf("%d  ",fac[i]);
        }
        puts("\n");
        fac.clear();
    }
    return 0;
}
