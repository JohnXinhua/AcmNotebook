#include<stdio.h>
#include<math.h>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>

typedef long long ll;
using namespace std;

#define MAX 1000009

long long BigMod(long long B,long long P,long long M)
{
    long long R=1;
    while(P>0)
    {
        if(P%2==1)
        {
            R=(R*B)%M;
        }
        P/=2;
        B=(B*B)%M;
    }
    return R;
}
int least_digite(int n,int k)
{
    long long  ans=BigMod((long long)n,(long long )k,(long long)1000);  /// M=1000 case: we find last 3 Digite
    return (int)ans;
}

int Most_digite(int n,int k)
{
    int m=3;
    long double a;
    int ans;

    a =(long double)k*log10(n);
    a = a - (long long)a;
    ans=(long long )(pow(10,a) * pow(10,m-1.00));
    /**
        log(n^k) = k*log(n) = x.fractional_part=y
        so  10^y=(n^k)=10^x*10^fractional_part
        so (10^fractional_part) =  First 3 Digite in this expresion
    **/
    return ans;
}

int main()
{
    int i,t,cas=0;
    int n ,k;
    scanf("%d",&t);
    while(t--)
    {
        scanf("%d %d",&n,&k);
        int ans1=least_digite(n,k);
        int ans2=Most_digite(n,k);
        printf("Case %d: %03d %03d\n",++cas,ans2,ans1);
    }
    return 0;
}
