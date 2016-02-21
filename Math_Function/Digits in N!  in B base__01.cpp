#include<stdio.h>
#include<math.h>
#include<vector>
#include<iostream>
#include<string>

using namespace std;

#define MAX 1000009

double arr[MAX];

void Digits_in_N_factorial_with_10base()///in 10 base Find the Digts in N! <><> take N! in B base Do ans=arr[n]/log10(b)
{
    int i;
    double ans=0.0;arr[0]=0.0;
    for(i=1;i<MAX;i++)
    {
        ans+=log10(i);
        arr[i]=ans;
    }
    return;
}

int main()
{
    Digits_in_N_factorial_with_10base();

    int i,k,n,cas=0,t;

    scanf("%d",&t);
    while(t--)
    {
        scanf("%d %d",&n,&k);
        double ans=arr[n];
        ans/=log10(k);         /// take in B base;
        ans+=1.0;
        ans=floor(ans);
        printf("Case %d: %.0lf\n",++cas, ans);

    }
    return 0;
}
