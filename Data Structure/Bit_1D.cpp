
#define type long long
const int NN=MAX-2;
void update(int x, type v)
{
    while(x <= NN)
    {
        tree[x]+= v;
        x += (x & -x);
    }
}
type get(int x)
{
    type res = 0;
    while(x)
    {
        res += tree[x];
        x -= (x & -x);
    }
    return res;
}
