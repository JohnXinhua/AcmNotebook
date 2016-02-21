struct BIT
{
    typedef int T;
    vector<T>tree;
    BIT(int n):tree(n,0){}

    void update(int idx,T val)
    {
        while(idx<(int)tree.size())
        {
            tree[idx]+=val;
            idx+= idx & -idx;
        }
    }
    T read(int idx)
    {
        T res=0;
        while(idx>=1)
        {
            res+=tree[idx];
            idx-= idx & -idx;
        }
        return res;
    }
};
