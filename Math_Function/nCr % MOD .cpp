#include<stdio.h>
#include<math.h>
#include<vector>
#include<iostream>
#include<string>

typedef long long ll;
using namespace std;

#define MAX 1000009

ll arr[MAX];
ll MOD = 1000003; // A Prime Number

ll Sqr(ll n) {return n*n;}

ll BigMod(ll B,ll P,ll M)
{
    if(P==0) return 1;
    else if(P%2==0) return Sqr(BigMod(B,P/2,M))%M;
    else return (B%MOD) * (BigMod(B,P-1,M))%M;
}
void Store_N_factorial_By_MOD()
{
    ll i,j,k; arr[0] =1;      /// 0! = 1
    for(i=1;i<MAX;i++) arr[i] = (arr[i-1]*i)%MOD;
    return;
}

ll nCr_MOD(ll n ,ll r)
{
    ll num,denom,ret;
    num=arr[n];
    denom=(arr[r]*arr[n-r])%MOD;
    ret = (num * BigMod(denom,MOD-2,MOD))%MOD; /// inverse Mod: (x/y)%M = ((x%M)*(y^M-2)%M)%M  if M=Prime Number
    return ret;
}

int main()
{
    Store_N_factorial_By_MOD();

    ll i,j,n,r;int cas=0;

    while(scanf("%lld %lld",&n,&r)==2)
    {
        ll ans=nCr_MOD(n,r);
        printf("Case %d: %lld\n",++cas,ans);
    }
}
