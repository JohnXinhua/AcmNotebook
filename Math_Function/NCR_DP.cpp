#include<stdio.h>
#include<string.h>
#include<iostream>

using namespace std;

#define MAX_N 10009
#define MAX_R 7
#define MOD 1000000009
#define type long long

type pd[MAX_N][MAX_R];
type pdVis[MAX_N][MAX_R];
int gura=1;
type ncr(type n,type r)
{
    if(n<r) return 0;
    if(n==0 || r==0 || n==r) return 1;
    if(pdVis[n][r]==gura) return pd[n][r];
    pdVis[n][r]=gura;
    pd[n][r]=(ncr(n-1,r)+ncr(n-1,r-1) ) %MOD;
    return pd[n][r];
}

int main()
{
    return 0;
}
