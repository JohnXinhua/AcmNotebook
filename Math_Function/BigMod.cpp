#include<stdio.h>
#include<iostream>
using namespace std;

template<class T>inline T sqr(T n){return (n*n);}
template<class T> T BigMod(T B,T P,T M)   /// (B^p)%M
{
    if(P==0)return 1;
    else if(P%2==0)return sqr(BigMod(B,P/2,M))%M;
    else return(B%M)*(BigMod(B,P-1,M))%M;
}

int main()
{
    int i,j,k;
    ll b,p,m;
    while(scanf("%lld %lld %lld",&b,&p,&m)==3)
    {
        ll ans=bigmod(b,p,m);
        printf("BigMod value: %lld\n",ans);
    }
    return 0;
}

