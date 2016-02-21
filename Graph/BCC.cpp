#include <cstdio>
#include <iostream>
#include <vector>
#include <stack>
using namespace std;
#define mp make_pair
#define SZ(c) (int)c.size()
#define MAX 10000
const double EPS = 1e-9;
const int INF = 0x7f7f7f7f;

vector< int > G[MAX];
stack<pair<int,int> > S;
int dfstime;
int low[MAX], vis[MAX], used[MAX];
int N,M;

void dfs(int u, int par)
{
    int v, i, sz = G[u].size();pair<int,int> e, curr;
    used[u] = 1;vis[u] = low[u] = ++dfstime;
    for(i = 0; i < sz; i++)
    {
        v = G[u][i];
        if(v == par) continue;
        if(!used[v])
        {
            S.push(mp(u, v));
            dfs(v, u);
            if(low[v] >= vis[u])
            {
                // new component
                curr = mp(u, v);int cnt=0;
                do
                {
                    e = S.top();
                    S.pop();
                    cnt++;
                    // e is an edge in current bcc
                    //if(e==curr and cnt==1) break;     //// single edge cheak.
                }
                while(e != curr);
            }
            low[u] = min(low[u], low[v]);
        }
        else if(vis[v] < vis[u])
        {
            S.push(mp(u, v));
            low[u] = min(low[u], vis[v]);
        }
    }
    return;
}

void BCC()
{
    for(int i=0;i<=N;i++){low[i]=vis[i]=used[i]=0;}
    while(!S.empty()) S.pop();
    dfstime=0;
    for(int i=0;i<N;i++)
    {
        if(used[i]==0) dfs(i,-1);
    }
    /// if low is differt between two nodes. Then we can say that...
    /// they are in two different bcc.
    return;
}

int main()
{
    while(cin>>N and N)
    {
        cin>>M;int a,b;
        for(int i=0;i<=N;i++) G[i].clear();
        for(int i=0;i<M;i++)
        {
            cin>>a>>b; // 0 base
            G[a].push_back(b);
            G[b].push_back(a);
        }
        BCC();
    }
    return 0;
}


/***


                        0------1
                        |     -
                        |   -
                        | -
                        2------3
                        |     -
                        |   -
                        | -
                        4-
                        |
                        |
                        |
             6----------5----------7
             |          |   |       |
                |       |       |   |
                    |   |           |
                        8------------9



10 14
0 1
0 2
1 2
2 3
2 4
3 4
4 5
5 6
5 8
5 9
5 7
6 8
8 9
9 7


5 4
0 4
4 1
1 2
1 3


7 8
0 1
1 2
1 3
2 4
4 5
5 0
3 4
0 6


4 5
0 1
1 2
1 3
2 3
0 3



*/
