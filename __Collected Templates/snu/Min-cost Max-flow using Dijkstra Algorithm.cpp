#include <cstring>
#include <queue>
#include <vector>
#include <algorithm>
#include <functional>
using namespace std;
struct edge
{
    int target;
    int capacity; // cap_t
    int cost; // cost_t
};
namespace mcmf
{
typedef int cap_t; // capacity type
typedef int cost_t; // cost type
const int SIZE = 5000;
const cap_t CAP_INF = 0x7fFFffFF;
const cost_t COST_INF = 0x7fFFffFF;
int n;
vector<pair<edge, int> > g;
int p[SIZE];
cost_t dist[SIZE];
cap_t mincap[SIZE];
Seoul National University  11
cost_t pi[SIZE];
int pth[SIZE];
int from[SIZE];
bool v[SIZE];

void init(const vector<edge> graph[], int size)
{
    int i, j;
    n = size;
    memset(p, -1, sizeof(p));
    g.clear();
    for (i = 0 ; i < size ; i++)
    {
        for (j = 0 ; j < graph[i].size() ; j++)
        {
            int next = graph[i][j].target;
            edge tmp = graph[i][j];
            g.push_back(make_pair(tmp, p[i]));
            p[i] = g.size() - 1;
            tmp.target = i;
            tmp.capacity = 0;
            tmp.cost = -tmp.cost;
            g.push_back(make_pair(tmp, p[next]));
            p[next] = g.size() - 1;
        }
    }
}
int dijkstra(int s, int t)
{
    typedef pair<cost_t, int> pq_t;
    priority_queue<pq_t, vector<pq_t>, greater<pq_t> > pq;
    int i;
    for (i = 0 ; i < n ; i++)
    {
        dist[i] = COST_INF;
        mincap[i] = 0;
        v[i] = false;
    }
    dist[s] = 0;
    mincap[s] = CAP_INF;
    pq.push(make_pair(0, s));
    while (!pq.empty())
    {
        int now = pq.top().second;
        pq.pop();
        if (v[now]) continue;
        v[now] = true;
        for (i = p[now] ; i != -1 ; i = g[i].second)
        {
            int next = g[i].first.target;
            if (v[next]) continue;
            if (g[i].first.capacity == 0) continue;
            cost_t pot = dist[now] + pi[now] - pi[next] + g[i].first.cost;
            if (dist[next] > pot)
            {
                dist[next] = pot;
                mincap[next] = min(mincap[now], g[i].first.capacity);
                pth[next] = i;
                from[next] = now;
                pq.push(make_pair(dist[next], next));
            }
        }
    }
    for (i = 0 ; i < n ; i++) pi[i] += dist[i];
    return dist[t] != COST_INF;
}
pair<cap_t, cost_t> maximum_flow(int source, int sink)
{
    memset(pi, 0, sizeof(pi));
    cap_t total_flow = 0;
    cost_t total_cost = 0;
    while (dijkstra(source, sink))
    {
        cap_t f = mincap[sink];
        total_flow += f;
        for (int i = sink ; i != source ; i = from[i])
        {
            g[pth[i]].first.capacity -= f;
            g[pth[i] ^ 1].first.capacity += f;
            total_cost += g[pth[i]].first.cost * f;
        }
    }
    return make_pair(total_flow, total_cost);
}

} // namespace mcmf
