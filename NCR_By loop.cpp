#include<stdio.h>
#include<math.h>
#include<string.h>
#include<iostream>
using namespace std;

long long findNcR ( int n , int r )
{
    long long res=1LL;
    r = min (r ,n-r );

    int i , j;
    for ( i=1,j=n ; i<=r ; i++,j-- )
    {
        res *= j;
        res /= i;
    }
    return res;
}

int main()
{
    int n,r;
    while(true)
    {
        cin>>n>>r;
        cout<<"NCR: "<<findNcR(n,r)<<endl;
    }
    return 0;
}
