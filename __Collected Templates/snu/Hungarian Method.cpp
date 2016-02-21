#include <cstring>
#include <queue>
#include <algorithm>
#include <limits>
using namespace std;
namespace hungarian
{
typedef double val_t;
const int SIZE = 100;
const val_t INF = numeric_limits<double>::infinity();
inline bool eq(val_t a, val_t b)
{
    static const double eps = 1e-9;
    return (a - eps < b && b < a + eps);
}

int n;
val_t cost[SIZE][SIZE];
int xy[SIZE], yx[SIZE];
int match_num;
val_t lx[SIZE], ly[SIZE];
bool s[SIZE], t[SIZE];
int prev[SIZE];

val_t hungarian()
{
    memset(xy, -1, sizeof(xy));
    memset(yx, -1, sizeof(yx));
    memset(ly, 0, sizeof(ly));
    match_num = 0;
    int x, y;
    for (x = 0 ; x < n ; x++)
    {
        lx[x] = cost[x][0];
        for (y = 1 ; y < n ; y++)
            lx[x] = max(lx[x], cost[x][y]);
    }
    for (x = 0 ; x < n ; x++)
        for (y = 0 ; y < n ; y++)
            if (eq(cost[x][y], lx[x] + ly[y]) && yx[y] == -1)
            {
                xy[x] = y;
                yx[y] = x;
                match_num++;
                break;
            }
    while (match_num < n)
    {
        memset(s, false, sizeof(s));
        memset(t, false, sizeof(t));
        memset(prev, -1, sizeof(prev));
        queue<int> q;
        for (x = 0 ; x < n ; x++)
        {
            if (xy[x] == -1)
            {
                q.push(x);
                s[x] = true;
                break;
            }
        }
        bool flg = false;
        while (!q.empty() && !flg)
        {
            x = q.front();
            q.pop();
            for (y = 0 ; y < n ; y++)
            {
                if (eq(cost[x][y], lx[x] + ly[y]))
                {
                    t[y] = true;
                    if (yx[y] == -1)
                    {
                        flg = true;
                        break;
                    }
                    if (!s[yx[y]])
                    {
                        s[yx[y]] = true;
                        q.push(yx[y]);
                        prev[yx[y]] = x;
                    }
                }
            }
        }
        if (flg)
        {
            int t1, t2;
            while (x != -1)
            {
                t1 = prev[x];
                t2 = xy[x];
                xy[x] = y;
                yx[y] = x;
                x = t1;
                y = t2;
            }
            match_num++;
        }
        else
        {
            val_t alpha = INF;
            for (x = 0 ; x < n ; x++) if (s[x])
                    for (y = 0 ; y < n ; y++) if (!t[y])
                            alpha = min(alpha, lx[x] + ly[y] - cost[x][y]);
            for (x = 0 ; x < n ; x++) if (s[x]) lx[x] -= alpha;
            for (y = 0 ; y < n ; y++) if (t[y]) ly[y] += alpha;
        }
    }
    val_t ret = 0;
    for (x = 0 ; x < n ; x++)
        ret += cost[x][xy[x]];
    return ret;
}

} // namespace hungarian
