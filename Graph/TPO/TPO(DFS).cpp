#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

vector<int>res,adj[200];
int N;
bool color[200];

void dfs(int u)
{
    color[u]=1;

    for(int i=0;i<adj[u].size();i++)
    {
        int v=adj[u][i];
        if(color[v]==0) dfs(v);
    }
    res.push_back(u);
}

void Topological_sort()
{
    res.clear();
    memset(color,0,sizeof(color));

    for(int i=1;i<=N;i++)
        if(color[i]==0) dfs(i);

    reverse(res.begin(),res.end());
}



int main()
{
    int test,Case=1,i,j,u,v,M;

    scanf("%d",&test);

    while(test--)
    {
        scanf("%d %d",&N,&M); //N->no. of node, M->No. of edge.

        for(i=0;i<N;i++) adj[i].clear();
        for(i=0;i<M;i++)
        {
            scanf("%d %d",&u,&v);
            adj[u].push_back(v);
        }
        Topological_sort();
        for(i=0;i<res.size();i++)  //printing the sorted list.
            printf("%d ",res[i]);
        printf("\n");
    }
    return 0;
}


