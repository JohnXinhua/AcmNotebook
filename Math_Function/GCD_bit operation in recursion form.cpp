#include<stdio.h>
#include<iostream>

using namespace std;

unsigned int gcd(unsigned int u, unsigned int v)
{
    // simple cases (termination)
    if (u == v) return u;
    if (u == 0) return v;
    if (v == 0) return u;

    // look for factors of 2

    if (~u & 1) // u is even
    {
        if (v & 1) return gcd(u >> 1, v); // v is odd
        else  return gcd(u >> 1, v >> 1) << 1; // both u and v are even
    }
    if (~v & 1) return gcd(u, v >> 1);  // u is odd, v is even

    // reduce larger argument
    if (u > v) return gcd((u - v) >> 1, v);
    else return gcd((v - u) >> 1, u);
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

