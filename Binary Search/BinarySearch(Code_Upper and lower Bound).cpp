/*Bismillahir Rahmanir Rahim*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;
#define MAX 100

int arr[MAX];
int N;

// return the lower indx.
// if the elements not present then return the next lower indx.
// ex 2 2 3 3 7 .if we find indx for 5 it return the indx of 7(0 base).

int lower(int x)
{
    int left=0;int right=N-1;
    while(left<right)
    {
        int mid=(left+right)/2;
        if(arr[mid]<x) left=mid+1;
        else right=mid;
    }
    return left;
}

// return the upper indx.
// if the elements not present then return the lower indx of the next large value.
// ex 2 2 5 9 23 .if we find indx for 6 it return the indx of the value 5 (0 base) that is 2.
int upper(int x)
{
    int left=0;int right=N-1;
    while(left<right)
    {
        int mid=(left+right+1)>>1;
        if(arr[mid]>x) right=mid-1;
        else left=mid;
    }
    return left;
}

int main()
{
    while(scanf("%d",&N)==1)
    {
        for(int i=0;i<N;i++)
        {
            scanf("%d",arr+i);
        }
        int x;
        for(int i=0;i<N;i++)
        {
            printf("%d %3d\n",i,arr[i]);
        }
        puts("\n");

        while(scanf("%d",&x)==1 and x)
        {
            int lo=lower(x);
            int up=upper(x);
            cout<<"Lo: "<<lo<<"  up: "<<up<<endl;
        }
    }
    return 0;
}


/**

16
2 2 2 2 3 3 3 4 4 6 6 6 6 8 8 10

10
3 3 3 3 3 3 3 3 3 10

**/
