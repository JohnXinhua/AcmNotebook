#include<stdio.h>
#include<string.h>
#define mem(a,b)  memset(a,b,sizeof(a))
#define MAX 100002
using namespace std;
typedef long long ll;
#define type int

type N,Q;
type tree[4*MAX],col[4*MAX];
type arr[MAX];

type build(int node,int left,int right)
{
    if(left==right)
    {
        tree[node]=arr[left];
        return tree[node];
    }
    int mid=(right+left)/2;

    type p=build(node*2,left,mid);
    type q=build(node*2+1,mid+1,right);
    tree[node]=p+q;
    return tree[node];
}
void refresh(int node,int  left,int right)
{
    /// some change here.
    if(left==right) tree[node]+=col[node];
    else
    {
        col[node*2]+=col[node];
        col[node*2+1]+=col[node];
        tree[node]=tree[node]+(right-left+1)*col[node];
    }
    col[node]=0;
    return;
}
type quary(int node,int left,int right,int i,int j)
{
    if(col[node]) refresh(node,left,right);
    if(right<i or left>j) return 0;
    if(left>=i and right<=j) return tree[node];

    int mid=(left+right)/2;
    type p=quary(node*2,left,mid,i,j);
    type q=quary(node*2+1,mid+1,right,i,j);
    return p+q;
}

void update(int node,int left,int right,int i,int j,type val)
{
    if(right<i or left>j) return;
    if(left>=i and right<=j)
    {
        col[node]+=val;
        refresh(node,left,right);
        return;
    }
    int mid=(left+right)/2;

    update(node*2,left,mid,i,j,val);
    update(node*2+1,mid+1,right,i,j,val);

    if(col[node*2]) refresh(node*2,left,mid);
    if(col[node*2+1]) refresh(node*2+1,mid+1,right);
    tree[node]=tree[node*2]+tree[node*2+1];
    return;
}

int main()
{
    scanf("%lld %lld",&N,&Q);
    memset(tree,0,sizeof(tree));
    memset(col,0,sizeof(col));
    /// scan element.
    build(1,0,N-1);
    int x,y,z;type val;
    ll ans=quary(1,0,N-1,y,z);
    update(1,0,N-1,y,z,val);

    return 0;
}




