#include<stdio.h>
#include<iostream>
#include<vector>
#include<string.h>
#define pb push_back
using namespace std;

vector<int>adj[105];
int arr[105];
bool col[105];
int a[105], b[105];

bool dfs(int u)
{
    if(col[u]) return false;
    int i, v; col[u]=true;

    for(i=0;i<(adj[u].size());i++)
    {
        v = adj[u][i];
        if(arr[v]==-1 || dfs(arr[v]))
        {
            arr[v] = u;
            return true;
        }
    }
    return false;
}

int match(int n)
{
    int i, ret=0;
    memset(arr, -1, sizeof arr);
    for(i=0;i<n;i++)
    {
        memset(col, false, sizeof col);
        if(dfs(i)) ret++;
    }
    return ret;
}

int main()
{
    int t, cas=1;
    scanf("%d", &t);
    while(t--)
    {
        int i, j, u, v, n, m;
        scanf("%d", &n); for(i=0;i<n;i++) adj[i].clear();
        for(i=0;i<n;i++) scanf("%d", &a[i]);  /// A is a set.
        scanf("%d", &m);
        for(i=0;i<m;i++) scanf("%d", &b[i]); /// B is another set.

        for(i=0;i<n;i++)
        {
            for(j=0;j<m;j++)
            {
                if((b[j]%a[i])==0)
                    adj[i].pb(j); /// make edge.
            }
        }
        int ans = match(n); ///find matching

        printf("Case %d: %d\n", cas++, ans);
    }
    return 0;
}

