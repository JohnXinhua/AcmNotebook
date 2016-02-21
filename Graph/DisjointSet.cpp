#include<stdio.h>
#include<iostream>
#define MAX 1000009

using namespace std;

struct DisjointSet
{
    int Node,Rank[MAX],Par[MAX];
    DisjointSet(){;}
    DisjointSet(int _N){Node=_N;}   /// 0 based index.
    void Makeset(){ for(int i=0;i<Node;i++) {Par[i]=i;Rank[i]=0;}}
    void Makeset(int n){Node=n;for(int i=0;i<Node;i++) {Par[i]=i;Rank[i]=0;}}
    int Findset(int x){ if(Par[x]==x) return x;else return Par[x]=Findset(Par[x]);}
    void Link(int x,int y){ doLink(Findset(x),Findset(y));}
    void doLink(int x,int y)
    {
        if(Rank[x]>Rank[y]) Par[y]=x;
        else Par[x]=y;
        if(Rank[x]==Rank[y])Rank[y]++;
    }
};

int main()
{
    int N=100;
    DisjointSet mst(N);
    return 0;
}
