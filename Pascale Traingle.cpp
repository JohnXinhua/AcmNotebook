#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#define MAX 80

int nCr[MAX][MAX];
long long findNcR(int n,int r)
{
    nCr[0][0]=1;
    for(int i=1;i<MAX;i++)
    {
        nCr[i][0]=1;
        for(int j=1;j<MAX;j++) nCr[i][j]=nCr[i-1][j]+nCR[i-1][j-1];
    }
    return;
}

int main()
{
    findNcR();
    return 0;
}

/*
 (n,k) = (n-1 ,k-1) + (n-1 , k);
 (n,0)+(n,1)+(n,2)+.......(n,n)=2^n;
*/
