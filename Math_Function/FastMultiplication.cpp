/*If we want to do an operation like 10^18 * 10^18 % MOD,
where MOD is also of the order of 10^18,
direct multiplication will result in overflow of even unsigned long long.

Now if we want to multiply a by b and b is say 13, then a * b = a * 13 can be written

a * 13 = a + 2 * ( 13/2 * a)
       = a + 2 * ( 6 * a)
       = a + 2 * ( 2 * (6/2 * a))
       = a + 2 * (2 * (3 * a))
       = a + 2 * (2 * (2 * (a + (3/2 * a))))
       = a + 2 * (2 * (2 * (a + ( 1 * a))))
*/

#include<stdio.h>
#include<iostream>
#define ll long long

using namespace std;


// recursive
#if 0
ll FastMultiplication(ll a,ll b,ll m)
{
    if(b==0) return 0;
    ll ret=FastMultiplication(a,b/2,m);
    ret=(ret+ret)%m;
    if(b&1) ret=(ret+a)%m;
    return ret;
}
#endif // 0

// iterative
ll FastMultiplication(ll a,ll b,ll m)
{
    a %= m; b %= m;
    ll ret=0;

    while(b)
    {
        if(b&1)
        {
            ret += a;
            if(ret >= m) ret -= m;
        }
        a = a<<1;
        if(a >= m) a -= m;
        b = b>>1;
    }
    return ret;
}

int main()
{
    ll a,b,c; /// (a<=10^18 & b<=10^18 & c<=10^18)

    cin>>a>>b>>c;
    ll res=FastMultiplication(a,b,c);  // (a*b)%c
    cout<<res<<endl;
    return 0;
}

