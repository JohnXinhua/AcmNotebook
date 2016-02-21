#include <cstring>
#include <queue>
using namespace std;

namespace netflow
{
    typedef int val_t;
    const int SIZE = 1000;
    const val_t INF = 0x7fFFffFF;
    int n;
    val_t capacity[SIZE][SIZE];
    val_t total_flow;
    val_t flow[SIZE][SIZE];
    int back[SIZE];
    inline val_t res(int a, int b)
    {
        return capacity[a][b] - flow[a][b];
    }
    val_t push_flow(int source, int sink)
    {
        memset(back, -1, sizeof(back));
        queue<int> q;
        q.push(source);
        back[source] = source;
        while (!q.empty() && back[sink] == -1)
        {
            int now = q.front();
            q.pop();
            for (int i = 0 ; i < n ; i++)
            {
                if (res(now, i) > 0 && back[i] == -1)
                {
                    back[i] = now;
                    q.push(i);
                }
            }
        }
        if (back[sink] == -1) return 0;
        int now, bef;
        val_t f = INF;
        for (now = sink ; back[now] != -1 ; now = back[now])
            f = min(f, res(back[now], now));
        for (now = sink ; back[now] != -1 ; now = back[now])
        {
            bef = back[now];
            flow[bef][now] += f;
            flow[now][bef] = -flow[bef][now];
        }
        total_flow += f;
        return f;
    }

    val_t maximum_flow(int source, int sink)
    {
        memset(flow, 0, sizeof(flow));
        total_flow = 0;
        while (push_flow(source, sink));
        return total_flow;
    }

} // namespace netflow


int main()
{
    return 0;
}
