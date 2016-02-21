#include <cstring>
#include <vector>
#include <algorithm>
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
const int SIZE = 300;
const cap_t CAP_INF = 0x7fFFffFF;
const cost_t COST_INF = 0x7fFFffFF;
int n;
vector<pair<pair<int, edge>, int> > g;
int p[SIZE];
cost_t dist[SIZE];
cap_t mincap[SIZE];
int pth[SIZE];
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
            g.push_back(make_pair(make_pair(i, tmp), p[i]));
            Seoul National University  10
            p[i] = g.size() - 1;
            tmp.target = i;
            tmp.capacity = 0;
            tmp.cost = -tmp.cost;
            g.push_back(make_pair(make_pair(next, tmp), p[next]));
            p[next] = g.size() - 1;
        }
    }
}
int bellman(int s, int t)
{
    int i, j;
    for (i = 0 ; i < n ; i++)
    {
        dist[i] = COST_INF;
        mincap[i] = 0;
    }
    dist[s] = 0;
    mincap[s] = CAP_INF;
    bool flg = false;
    for (i = 0 ; i < n ; i++)
    {
        flg = false;
        for (j = 0 ; j < g.size() ; j++)
        {
            int now, next;
            if (g[j].first.second.capacity == 0) continue;
            now = g[j].first.first;
            next = g[j].first.second.target;
            if (dist[now] == COST_INF) continue;
            if (dist[now] + g[j].first.second.cost < dist[next])
            {
                dist[next] = dist[now] + g[j].first.second.cost;
                pth[next] = j;
                mincap[next] = min(mincap[now], g[j].first.second.capacity);
                flg = true;
            }
        }
        if (!flg) break;
    }
    if (flg) return -1;
    return dist[t] != COST_INF ? 1 : 0;
}
pair<cap_t, cost_t> maximum_flow(int source, int sink)
{
    cap_t total_flow = 0;
    cost_t total_cost = 0;
    int state;
    while ((state = bellman(source,sink)) > 0)
    {
        cap_t f = mincap[sink];
        total_flow += f;
        total_cost += f * dist[sink];
        for (int i = sink ; i != source; i = g[pth[i]].first.first)
        {
            g[pth[i]].first.second.capacity -= f;
            g[pth[i] ^ 1].first.second.capacity += f;
        }
    }
    if (state == -1) while (true); // it's NP-Hard
    return make_pair(total_flow, total_cost);
}

} // namespace mcmf
