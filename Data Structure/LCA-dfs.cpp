/*
    You are given a rooted tree, where each node contains an integer value.
    And the value of a node is strictly greater than the value of its parent.
    Now you are given a node and an integer query. You have to find the greatest possible parent of this node
    (may include the node itself), whose value if greater than or equal to the given query integer.
*/

#include <cstdio>
#include <cstring>
#define MAXN 100100
#define LOGMAXN 18

int color[MAXN],parent[MAXN][LOGMAXN],maxValue[MAXN],store[MAXN];
int start[MAXN],finish[MAXN],T,step,nodes;
vector<int> adj[MAXN];

void dfs(int u,int par)
{
    int i,v;color[u]=1;start[u]=T++;
    parent[u][0]=par;
    for(i=1;i<=step;i++)  parent[u][i]=parent[parent[u][i-1]][i-1];

    maxValue[u]=store[u];
    maxValue[u]=max(maxValue[u],maxValue[par]); // change here
    REP(i,SZ(adj[u]))
    {
        v=adj[u][i];if(color[v]) continue;
        dfs(v,u);
    }
    finish[u]=T++;
}
bool IsAnchestor(int u,int v)
{
    if(start[u]<=start[v] && finish[u]>=finish[v]) return true;
    return false;
}
int lca_query(int v,int val)
{
    int i,j,p;
    for(i=step;i>=0;i--)
    {
        for(j=step;j>=0;j--)
        {
            p=parent[v][j];
            if(maxValue[p]>=val){v=p;break;}
        }
    }
    return v;
}

int main()
{
    int test,Case=1,i,j,Q,u,v,val,ans;scanf("%d",&test);
    while(test--)
    {
        scanf("%d %d",&nodes,&Q);
        for(int i=0;i<=nodes;i++) adj[i].clear();
        store[0]=1;
        for(int i=0;i<nodes-1;i++)
        {
            u=i+1;scanf("%d %d",&v,&val);
            store[u]=val;
            adj[u].pb(v);
            adj[v].pb(u);
        }
        for(step=0;(1<<step)<=nodes;step++);
        memset(color,0,sizeof(color));
        T=0;dfs(0,0);
        printf("Case %d:\n",Case++);
        while(Q--)
        {
            scanf("%d %d",&v,&val);
            ans=lca_query(v,val);
            printf("%d\n",ans);
        }
    }
    return 0;
}


