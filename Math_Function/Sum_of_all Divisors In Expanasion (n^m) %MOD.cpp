#include<stdio.h>
#include<math.h>
#include<vector>
#include<iostream>
#include<string>

typedef long long ll;
using namespace std;

#define MAX 1000009

ll square(ll n) {return n*n;}

bool flag[MAX];
int prime[MAX];

ll MOD=1000000007;

void sive()
{
    ll i,j,k,r,total=1;
    flag[0]=flag[1]=1;
    prime[total]=2;

    for(i=3;i<MAX;i+=2)
    {
        if(!flag[i])
        {
            prime[++total]=i;
            r=i*2;
            if(MAX/i>=i)for(j=i*i;j<MAX;j+=r)flag[j]=1;
        }
    }
    return;
}


ll BigMod(ll b,ll p, ll m)            /// b^p%m
{
    if(p==0)return 1;
    else if(p%2==0) return square(BigMod(b,p/2,m))%m;
    else return (b%m)*(BigMod(b,p-1,m))%m;
}

ll pseudo_code(ll n,ll m)
{
    ll i,c,p,res=1;ll ans=1;
    for(i=1;prime[i]<=n/prime[i];i++)
    {
        if(n%prime[i]==0)
        {
            c=0;
            while(n%prime[i]==0)
            {
                c++;
                n/=prime[i];
            }
            ll po=(c*m)+1;
            ll upmod=(((BigMod(prime[i],po,MOD))+MOD)-1)%MOD;
            ll downmod=BigMod(prime[i]-1,MOD-2,MOD);
            res =(upmod * downmod)%MOD;
            ans=(ans*res)%MOD;
        }
    }
    if(n>1)
    {
        ll po=m+1;
        ll upmod =((BigMod(n,po,MOD)+MOD)-1)%MOD;
        ll downmod =BigMod(n-1,MOD-2,MOD);
        res =(upmod * downmod)%MOD;
        ans=(ans*(res%MOD))%MOD;
    }
    return ans;
}


int main()
{
    sive();
    ll i,j,k,n,m,t;
    scanf("%lld",&t);
    for(k=1;k<=t;k++)
    {
        scanf("%lld %lld",&n,&m);
        ll ans=pseudo_code(n,m);
        printf("Case %lld: %lld\n",k,ans);
    }
    return 0;
}
