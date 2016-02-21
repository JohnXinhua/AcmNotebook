/*
You are given a tree (an acyclic undirected connected graph) with N nodes,
and edges numbered 1, 2, 3...N-1.
We will ask you to perfrom some instructions of the following form:
    CHANGE i ti : change the cost of the i-th edge to ti
    or
    QUERY a b : ask for the maximum edge cost on the path from node a to node b.
*/

#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <vector>
#define MAX 10010
using namespace std;

struct Edge
{
    int u, v, w;
    Edge() {};
    Edge(int a, int b, int c){u = a, v = b, w = c;}
};
vector<Edge> edge[MAX];
Edge alledge[MAX];

int t, n, idx;
int dep[MAX], size[MAX];
int son[MAX], tid[MAX];
int pre[MAX], top[MAX];
bool vis[MAX];

/// heavy light Decomposition....
void find_heavy_edge(int x, int father, int depth)
{
    vis[x] = true;pre[x] = father;dep[x] = depth;
    size[x] = 1;son[x] = -1; int maxsize = 0;
    int sz = edge[x].size();
    for (int i = 0; i < sz; ++i)
    {
        int child = edge[x][i].v;
        if (!vis[child])
        {
            find_heavy_edge(child, x, depth + 1);
            size[x] += size[child];
            if (size[child] > maxsize)
            {
                maxsize = size[child];
                son[x] = child;
            }
        }
    }
    return;
}

void connect_heavy_edge(int x, int ance)
{
    
    vis[x] = true; tid[x] = ++idx; top[x] = ance;
    if (son[x] != -1){connect_heavy_edge(son[x], ance);}
    int sz = edge[x].size();
    for (int i = 0; i < sz; ++i)
    {
        int child = edge[x][i].v;
        if (!vis[child]){connect_heavy_edge(child, child);}
    }
    return;
}

/// segment tree
int Max[MAX<<2],val[MAX];
void build(int rt,int l, int r)
{
    if (l == r){Max[rt] = val[l];return ;}
    int m = (l + r) >> 1;
    build(rt << 1,l,m);
    build((rt << 1) | 1,m + 1, r);
    Max[rt] = max(Max[rt<<1], Max[rt<<1|1]);
}
void update(int rt,int l, int r, int p, int c)
{
    if (l == r){Max[rt] = c;return ;}
    int m = (l + r) >> 1;
    if (p <= m){update( rt << 1,l, m, p, c); }
    else {update((rt << 1) | 1,m + 1, r,  p, c);}
    Max[rt] = max(Max[rt<<1], Max[rt<<1|1]);
}

int query(int rt,int l, int r,  int L, int R)
{
    if (L <= l && R >= r){return Max[rt];}
    int m = (l + r) >> 1;
    int tmp = -(1 << 30);
    if (L <= m){tmp = max(tmp, query(rt << 1,l, m,  L, R));}
    if (R > m){tmp = max(tmp, query((rt << 1) | 1,m + 1, r,  L, R));}
    return tmp;
}

void CHANGE(int x, int val)
{
    if (dep[alledge[x].u] > dep[alledge[x].v]){
        update(1,2, n,tid[alledge[x].u], val);
    }
    else{
        update(1,2, n,tid[alledge[x].v], val);
    }
}

int QUERY(int a, int b)
{
    int ans = -(1 << 30);
    while (top[a] != top[b])
    {
        if (dep[top[a]] < dep[top[b]]) swap(a, b);
        ans = max(ans, query(1,2, n,tid[top[a]], tid[a]));
        a = pre[top[a]];
    }
    if (dep[a] > dep[b]) swap(a, b);
    if (a != b) ans = max(ans, query(1,2, n,tid[a] + 1, tid[b]));
    return ans;
}

int main()
{
    scanf("%d", &t);
    while (t--)
    {
        scanf("%d", &n);
        for (int i = 0; i <= n; ++i){edge[i].clear();}
        int a, b, c;
        for (int i = 1; i <= n - 1; ++i)
        {
            scanf("%d%d%d", &a, &b, &c);
            alledge[i].u = a;
            alledge[i].v = b;
            alledge[i].w = c;
            edge[a].push_back(Edge(a, b, c));
            edge[b].push_back(Edge(b, a, c));
        }
        memset(vis, false, sizeof(vis));
        find_heavy_edge(1, 1, 1);
        idx = 0;
        memset(vis, false, sizeof(vis));
        connect_heavy_edge(1, 1);

        for (int i = 1; i <= n - 1; ++i)
        {
            if (dep[alledge[i].u] > dep[alledge[i].v])
                val[tid[alledge[i].u]] = alledge[i].w;
            else
                val[tid[alledge[i].v]] = alledge[i].w;
        }

        build(1,2, n);char op[10];
        while (scanf("%s", op))
        {
            if (op[0] == 'D') break;
            scanf("%d%d", &a, &b);
            if (op[0] == 'Q')
                printf("%d\n", QUERY(a, b));
            else
                CHANGE(a, b);
        }
    }
    return 0;
}
