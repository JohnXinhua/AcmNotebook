#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<sstream>

using namespace std;
typedef long long ll;
typedef vector<int> vi;

#define mp          make_pair
#define pb          push_back
#define Clr(a)       a.clear()
#define SZ(a)       (int)a.size()
#define mem(a,b)    memset(a,b,sizeof(a))
#define MAX 119


/*

you have to output the numbers(count) of N digits.......
that considering that M(a number) is not present in N-digits numbers.
Simply:--->>> .... M patarn can not present in N-digits Numbers.

*/

ll dp[MAX][MAX],N,len,M=10000007;
ll b[MAX]; /// use for kmp.
string str;

void KMP()
{
    mem(b,0);
    ll i,k=0;
    b[k]=0;
    len=SZ(str);

    for(i=1;i<len;i++)
    {
        while(k>0 and str[i]!=str[k]) k=b[k-1];
        if(str[k]==str[i]) k++;
        b[i]=k;
    }
    return;
}

ll rec(ll cur, ll match)
{
    if(match>=len) return 0;
    if(cur==N) return 1;
    ll &ret=dp[cur][match];
    if(ret!=-1) return ret;

    ret=0;
    for(ll i=0;i<=9;i++)
    {
        ll k=match;
        while(k>0 and (str[k]-'0')!=i) k=b[k-1];
        if((str[k]-'0')==i) k++;
        ret=(rec(cur+1,k)+ret)%M;
    }
    return ret%M;
}

int  main()
{
    int t;
    scanf("%d",&t);
    while(t--)
    {
        scanf("%d ",&N);
        cin>>str;

        ll ans=0;
        KMP();
        mem(dp,-1);

        for(ll i=1;i<=9;i++)
        {
            if(str[0]-'0'==i)ans=(rec(1,1)+ans)%M;
            else ans=(rec(1,0)+ans)%M;
        }
        if(N==1) ans++;
        printf("%lld\n",(ans%M));
    }
    return 0;
}
