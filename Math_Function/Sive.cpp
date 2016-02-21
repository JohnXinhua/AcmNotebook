#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;


#define MAX_N 100002
vector<int>prime;
bool flag[MAX_N];
/*MAX_N.  Be Careful.*/
int Sive()
{
    flag[0]=flag[1]=1;
    prime.push_back(2);
    int i;long long r;
    for(i=3; i<MAX_N; i+=2)
    {
        if(!flag[i])
        {
            prime.push_back(i);
            r=i*2;
            if(MAX_N/i>=i) for(int j=i*i; j<MAX_N; j+=r) flag[j]=1;
        }
    }
    return (int)prime.size();
}

bool isPrime(int n)
{
    if(n<=1) return false;
    if(n==2)return 1;
    if(n<MAX_N){ if((n%2) && !flag[n]) return 1;else return 0;}
    for(int i=0; prime[i]<=sqrt(n); i++){ if(n%prime[i]==0) return 0;}
    return 1;
}

int main()
{
    cout<<Sive()<<endl;
    int x;
    while(cin>>x)
    {
        cout<<isPrime(x)<<endl;
    }
    return 0;
}
