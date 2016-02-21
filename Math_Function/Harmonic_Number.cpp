#include<stdio.h>
#include<math.h>
#include<vector>
#include<iostream>
#include<string>

using namespace std;

#define MAX 1000009

int Harmonic_Number(int n)
{
    int start,tmp,end=n,val,ans=0,i;
    val = sqrt(n);
    for(i=1;i<=val;i++)
    {
        ans+=n/i;
        tmp=n/(i+1);
        if(tmp<i)tmp=i;
        ans+=((end-tmp)*i);
        end=tmp;
    }
    return ans;
}

int main()
{
    int n;
    scanf("%d",&n);
    int ans =Harmonic_Number(n);
    printf("Ans: %d\n",ans);
    return 0;
}
