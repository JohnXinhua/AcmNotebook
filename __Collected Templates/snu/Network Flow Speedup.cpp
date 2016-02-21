#include <cstring>
#include <vector>
#include <queue>
using namespace std;
struct edge
{
    int target;
    int capacity; // cap_t
};

namespace netflow
{
typedef int cap_t; // capacity type
const int SIZE = 5000;
const cap_t CAP_INF = 0x7fFFffFF;
int n;
vector<pair<edge, int> > g;
int p[SIZE];
int dist[SIZE];
cap_t maxcap;
void init(const vector<edge> graph[], int size)
{
    int i, j;
    n = size;
    memset(p, -1, sizeof(p));
    maxcap = 0;
    g.clear();
    for (i = 0 ; i < size ; i++)
    {
        for (j = 0 ; j < graph[i].size() ; j++)
        {
            int next = graph[i][j].target;
            edge tmp = graph[i][j];
            maxcap = max(maxcap, tmp.capacity);
            g.push_back(make_pair(tmp, p[i]));
            p[i] = g.size() - 1;
            tmp.target = i;
            tmp.capacity = 0;
            g.push_back(make_pair(tmp, p[next]));
            p[next] = g.size() - 1;
        }
    }
}
bool bfs(int s,int t,int delta)
{
    for (int i = 0 ; i < n ; i++)
        dist[i] = n + 1;
    queue<int> q;
    dist[s] = 0;
    q.push(s);
    while (!q.empty())
    {
        int now = q.front();
        q.pop();
        for (int i = p[now] ; i != -1 ; i = g[i].second)
        {
            int next = g[i].first.target;
            if (g[i].first.capacity < delta) continue;
            if (dist[next] == n + 1)
            {
                dist[next] = dist[now] + 1;
                q.push(next);
            }
        }
    }
    return dist[t] != n + 1;
}

cap_t dfs(int now, int t, int delta, cap_t minv = CAP_INF)
{
    if (now == t) return minv;
    for (int i = p[now] ; i != -1 ; i = g[i].second)
    {
        if (g[i].first.capacity < delta) continue;
        int next = g[i].first.target;
        if (dist[next] == dist[now] + 1)
        {
            cap_t flow = dfs(next, t, delta, min(minv, g[i].first.capacity));
            if (flow)
            {
                g[i].first.capacity -= flow;
                g[i ^ 1].first.capacity += flow;
                return flow;
            }
        }
    }
    return 0;
}
cap_t maxflow(int s, int t)
{
    cap_t delta = 1, totalflow = 0;
    while (delta <= maxcap) delta <<= 1;
    while (delta >>= 1)
    {
        while (bfs(s, t, delta))
        {
            cap_t flow;
            while (flow = dfs(s, t, delta)) // not ==
                totalflow += flow;
        }
    }
    return totalflow;
}

} // namespace netflow
