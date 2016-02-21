#include<stdio.h>
#include<iostream>
#include<algorithm>

using namespace std;
char dp[100];

int main()
{
    int i,j,k,t,cas=1,n;
    scanf("%d",&t);

    while(t--)
    {
        for(i=0;i<26;i++) dp[i] = i+'A';
        int cnt=0;
        scanf("%d %d",&n,&k);
        cout<<"Case: "<<cas<<endl;cas++;


        do
        {
            for(i=0;i<n;i++) cout<<dp[i];
            cout<<endl;
            cnt++;
            if(cnt==k) break;
        }while(next_permutation(&dp[0],&dp[0]+n));
    }
    return 0;
}
