/*
    Algorithm: LCA with Sparse Table
    Comment: sparse[i][j]=k means 2^j-th parent of node-i is k.
             parent[i]=j means j is the parent of node-i.
             level[i]=j means j is the level of node-i.
    Explanation: 1)http://www.shafaetsplanet.com/planetcoding/?p=1831
                 2)https://www.topcoder.com/tc?module=Static&d1=tutorials&d2=lowestCommonAncestor
    Sample Input:
                9
                0 8
                0 3
                0 2
                0 1
                2 4
                4 5
                5 7
                5 6

                4
                3 2
                8 1
                7 6
                4 6
*/

#include<bits/stdc++.h>
using namespace std;
const int MAX=200009;
int node;
int level[MAX];
int sparse[MAX][22]; //Sparse-Table
int parent[MAX];
vector<int>adj[MAX];
bool visited[MAX];

void dfs(int x,int lev)
{
    if(visited[x]) return;
    visited[x]=1;
    level[x]=lev;
    for(int i=0;i<adj[x].size();i++)
    {
        int y=adj[x][i];
        parent[y]=x;
        if(visited[y]==0) dfs(y,lev+1);
    }
}

void lca_init()
{
    memset(sparse,-1,sizeof(sparse));
    for(int i=0;i<node;i++)
        sparse[i][0]=parent[i];

    for(int j=1;(1<<j)<node;j++)
    {
        for(int i=0;i<node;i++)
        {
            if(sparse[i][j-1]!=-1)
                sparse[i][j]=sparse[sparse[i][j-1]][j-1];
        }
    }
}

int lca_query(int x,int y)
{
    //if x is situated on a higher level than y then we swap them
    if(level[x]<level[y]) swap(x,y);

    int log;
    for(log =1;(1<<log)<=level[x];log++);

    log--;
    //taking x and y in same level
    for(int i=log;i>=0;i--)
    {
        if(level[x]-(1<<i)>=level[y])
            x=sparse[x][i];
    }
    if(x==y) return x;

    //we compute LCA(x, y) using the values in x
    for(int i=log;i>=0;i--)
    {
        if(sparse[x][i]!=-1&&sparse[x][i]!=sparse[y][i])
            x=sparse[x][i], y=sparse[y][i];
    }
    return parent[x];
}

int main()
{
    int i,j,k,n,m,d,test,x,y,q;
    while(scanf("%d",&node)==1)
    {
        for(i=0;i<node-1;i++)
        {
            scanf("%d%d",&x,&y);
            adj[x].push_back(y);
        }
        memset(visited,0,sizeof(visited));
        dfs(0,0);
        lca_init();
        scanf("%d",&q);
        while(q--)
        {
            scanf("%d%d",&x,&y);
            int lca=lca_query(x,y);
            printf("The Lowest Common Ancestor of node %d and node %d is %d\n",x,y,lca);
        }
    }
    return 0;
}


