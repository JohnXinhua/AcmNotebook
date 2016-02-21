#include<stdio.h>
#include<string.h>
#include<iostream>
using namespace std;

#define MAX_N 10099
#define MAX_R 7
#define type long long
#define MOD 1000000007
type ncr[MAX_N][MAX_R];

void NCR(type n, type r)
{
    for (int i=0; i<=n; i++)
    {
        for (int k=0; k<=r && k<=i; k++)
        {
            if (k==0 || k==i) ncr[i][k]=1;
                //ncr[i&1][k] = 1; /*for ncr[2][MAX_R] */
            else ncr[i][k]=( ncr[i-1][k-1] + ncr[i-1][k])%MOD;
                //C[i&1][k] = (C[(i-1)&1][k-1] + C[(i-1)&1][k])%MOD; /*for ncr[2][MAX_R] */
        }
    }
    return;
}

int main()
{
    return 0;
}
