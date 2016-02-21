#include<stdio.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include <math.h>

#define mp Make_pair
#define pb push_back
#define MAX 100000

using namespace std;

bool flag[MAX];
int prime[MAX];
vector<pair<int ,int> > prim_fre;
vector<int>v;

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

void Factorise(int n) /// workes for Factorise number N
{
    int i,j,k;
    for(i=1;prime[i]<=sqrt(n);i++)
    {
        if(n%prime[i]==0)
        {
            k=0;
            while(n%prime[i]==0)
            {
                k++;
                n/=prime[i];
            }
            prim_fre.push_back(pair<int,int>(prime[i],k));
        }
    }
    if(n>1){prim_fre.push_back(pair<int,int >(n,1));}
    return;
}

void Divisor_generate(int cur,int num) /// generate N's Divisor
{
    int i,val;
    if(cur==prim_fre.size()){v.push_back(num);return;}
    val=1;
    for(i=0;i<=prim_fre[cur].second;i++)
    {
        Divisor_generate(cur+1,val*num);
        val=val*prim_fre[cur].first;
    }
    return;
}

void view(int n)
{
    int i;
    printf("All Divisor Of %d: ",n);
    for(i=0;i<v.size();i++)
    {
        printf("%d   ",v[i]);
    }
    puts("\n");
    return;
}

int main()
{
    sive();
    int n,t,cas=0;
    scanf("%d",&t);
    while(t--)
    {
        scanf("%d",&n);
        Factorise(n);
        Divisor_generate(0,1);
        sort(v.begin(),v.end());
        view(n);
        v.clear();
        prim_fre.clear();
    }
    return 0;
}

