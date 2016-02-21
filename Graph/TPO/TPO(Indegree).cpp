#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<vector>
using namespace std;

vector<int>res,adj[200];
int N,indeg[200];

void Topological_sort()
{
    int i,j,u,v;

    for(i=1;i<=N;i++)
        if(indeg[i]==0) res.push_back(i);

    for(i=0;i<res.size();i++)
    {
        u=res[i];
        for(j=0;j<adj[u].size();j++)
        {
            v=adj[u][j];
            indeg[v]--;
            if(indeg[v]==0) res.push_back(v);
        }
    }
    return;
}

int main()
{
    int test,Case=1,i,j,u,v,M;

    scanf("%d",&test);

    while(test--)
    {
        scanf("%d %d",&N,&M); //N->no. of node, M->No. of edge.

        for(i=0;i<N;i++) adj[i].clear();
        memset(indeg,0,sizeof(indeg));

        for(i=0;i<M;i++)
        {
            scanf("%d %d",&u,&v);
            adj[u].push_back(v);
            indeg[v]++;
        }

        res.clear();
        Topological_sort();

        for(i=0;i<res.size();i++)  //printing the sorted list.
            printf("%d ",res[i]);
        printf("\n");
    }
    return 0;
}
