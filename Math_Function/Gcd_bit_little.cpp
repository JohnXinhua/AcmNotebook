#include<stdio.h>
#include<iostream>

using namespace std;

unsigned int gcd(unsigned int u, unsigned int v)
{
    while(v) v^=u^=v^=u%=v;
    return u;
}

int  main()
{
    unsigned int a,b;
    while(cin>>a>>b)
    {
        cout<<gcd(a,b)<<endl;
    }
    return 0;
}
