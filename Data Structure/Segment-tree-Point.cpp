#include<stdio.h>
#include<string.h>
#include<iostream>
using namespace std;
#define MAX 100003

int arr[MAX],tree[4*MAX];
int N,Q;

int build(int node,int left,int right) //sum
{
    if(left==right)
    {
        tree[node]=arr[left];
        return tree[node];
    }
    int mid=(right+left)/2;

    int p=build(node*2,left,mid);
    int q=build(node*2+1,mid+1,right);
    tree[node]=p+q; // sum
    return tree[node];
}

int update(int node,int left,int right,int idx,int val)
{
    if(left==right)
    {
        tree[node]=val;
        return tree[node];
    }
    int mid=(left+right)/2;
    int p,q;
    if(mid>=idx)
    {
        p=update(node*2,left,mid,idx,val);
        q=tree[node*2+1];
    }
    else
    {
        p=tree[node*2];
        q=update(node*2+1,mid+1,right,idx,val);
    }
    tree[node]=p+q;
    return tree[node];
}

int Query(int node,int left,int right,int i,int j)
{
    if(right<i or left>j) return 0;
    if(left>=i and right<=j) return tree[node];

    int mid=(left+right)/2;
    int p=Query(node*2,left,mid,i,j);
    int q=Query(node*2+1,mid+1,right,i,j);
    return p+q;
}

int main()
{
    int i,j;
    cin>>N>>Q;
    ///element scan.
    build(1,0,N-1);
    Query(1,0,N-1,i,j);
    return 0;
}



