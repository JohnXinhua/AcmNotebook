#include <algorithm>
using namespace std;
namespace itree
{
typedef int val_t;
const int size = 1024;
const val_t init_value = 0;

inline val_t weight(val_t a, int len)
{
    return a * len;
    Seoul National University  22
}

val_t sum(val_t a, val_t b)
{
    return a + b;
}

val_t update_a(val_t a, val_t b)
{
    return a + b;
}

val_t update_b(val_t b1, val_t b2)
{
    return b1 + b2;
}
pair<val_t, val_t> itree[size * 2];
pair<int, int> ptree[size * 2];
void init()
{
    int i;
    for (i = 1 ; i < size * 2 ; i++)
        itree[i] = make_pair(init_value, init_value);
    for (i = size ; i < size * 2 ; i++)
        ptree[i] = make_pair(i, i);
    for (i = size - 1 ; i >= 1 ; i--)
        ptree[i] = make_pair(ptree[i << 1].first, ptree[i << 1 | 1].second);
}
void update(int s, int e, val_t val)   // [s, e]
{
    int s1, e1;
    int d = 0;
    s |= size;
    e |= size;
    s1 = s >> 1;
    e1 = e >> 1;
    while (s <= e)
    {
        if (s & 1)
        {
            itree[s].second = update_b(itree[s].second, val);
            val_t child_sum = sum(itree[s << 1].first, itree[s << 1 | 1].first);
            itree[s].first = update_a(
                                 (s >= size) ? init_value : child_sum,
                                 weight(itree[s].second, 1 << d));
        }
        if ((e & 1) == 0)
        {
            itree[e].second = update_b(itree[e].second, val);
            val_t child_sum = sum(itree[e << 1].first, itree[e << 1 | 1].first);
            itree[e].first = update_a(
                                 (e >= size) ? init_value : child_sum,
                                 weight(itree[e].second, 1 << d));
        }
        s = (s + 1) >> 1;
        e = (e - 1) >> 1;
        d++;
    }
    d = 1;
    while(s1)
    {
        itree[s1].first = update_a(
                              sum(itree[s1 << 1].first, itree[s1 << 1 | 1].first),
                              weight(itree[s1].second, 1 << d));
        itree[e1].first = update_a(
                              sum(itree[e1 << 1].first, itree[e1 << 1 | 1].first),
                              weight(itree[e1].second, 1 << d));
        s1 >>= 1;
        e1 >>= 1;
        d++;
    }
}
val_t _getrange2(int s, int e, int node)
{
    if (node >= size)
        return itree[node].first;
    if (s <= ptree[node].first && e >= ptree[node].second)
        return itree[node].first;
    val_t cur = weight(itree[node].second,
                       min(e, ptree[node].second) - max(s, ptree[node].first) + 1);
    int left = node << 1;
    int right = node << 1 | 1;
    if(s >= ptree[right].first)
        return update_a(_getrange2(s, e, right), cur);
    else if (e <= ptree[left].second)
        Seoul National University  23
        return update_a(_getrange2(s, e, left), cur);
    else
        return update_a(
                   sum(_getrange2(s, e, left), _getrange2(s, e, right)),
                   cur);
}
val_t getrange(int s, int e)   // [s, e]
{
    if(s > e) return init_value;
    return _getrange2(s | size, e | size, 1);
}

} // namespace itree
