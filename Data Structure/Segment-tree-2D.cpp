/*
A rectangular matrix a[N][M], and enter the amount of search queries (or minimum / maximum)
at some small matrix [x_1....x_2, y_1...y_2] and requests modification of individual elements
of the matrix (ie, queries of the form a [x][y] = p).
So, we will build a two-dimensional tree segments:
the first tree segments in the first coordinate ( x),then - on the second ( y).

*/
#include<stdio.h>
#include<algorithm>
using namespace std;
typedef long long ll;
#define MAX 505
#define type int

type tree[4*MAX][4*MAX];
int N,grid[MAX][MAX];

void build_y(int vx,int lx,int rx,int vy,int ly,int ry)
{
    if(ly==ry)
    {
        if(lx==rx) tree[vx][vy]=grid[lx][ly];
        else tree[vx][vy]=max(tree[vx*2][vy],tree[vx*2+1][vy]);
        return;
    }

    int my=(ly+ry)/2;
    build_y(vx,lx,rx,vy*2,ly,my);
    build_y(vx,lx,rx,vy*2+1,my+1,ry);
    tree[vx][vy]=max(tree[vx][vy*2],tree[vx][vy*2+1]);
}

void build_x(int vx,int lx,int rx)
{
    if(lx!=rx)
    {
        int mx=(lx+rx)/2;
        build_x(vx*2,lx,mx);
        build_x(vx*2+1,mx+1,rx);
    }
    build_y(vx,lx,rx,1,1,N);
}

type max_y(int vx,int vy,int tly,int _try,int ly,int ry)
{
    if(ry<tly or ly>_try) return 0;
    if(tly>=ly and _try<=ry)
    {
        int t=tree[vx][vy];
        return t;
    }

    int tmy=(tly+_try)/2;
    int m1=max_y(vx,vy*2,tly,tmy,ly,ry);
    int m2=max_y(vx,vy*2+1,tmy+1,_try,ly,ry);
    return max(m1,m2);
}

type max_x(int vx,int tlx,int trx,int lx,int rx,int ly,int ry)
{
    if(rx<tlx or lx>trx) return 0;
    if(tlx>=lx and trx<=rx);
    {
        int t=max_y(vx,1,1,N,ly,ry);
        return t;
    }
    int tmx=(tlx+trx)/2;
    int m1=max_x(vx*2,tlx,tmx,lx,rx,ly,ry);
    int m2=max_x(vx*2+1,tmx+1,trx,lx,rx,ly,ry);
    return max(m1,m2);
}


int main()
{
    int N,s1,s2,s,cnt; //s is the length.
    cin>>N>>s>>s1>>s2;
    build_x(1,1,N);
    cnt=max_x(1,1,N,s1,s1+s-1,s2,s2+s-1);
    return 0;
}


