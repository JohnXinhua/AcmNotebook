#include<stdio.h>
#include<iostream>
#include<algorithm>
#include<vector>

using namespace std;
#define SZ(s) (int)s.size()

vector<int>v;
int id1,id2;

int findMaxsum()
{
    int maxsum=-(1<<30);
    int currentstartindx=0,maxstartindex,maxendindx,currentsum=0;
    for(int i=0;i<SZ(v);i++)
    {
        currentsum+=v[i];
        if(currentsum>maxsum)
        {
            maxsum=currentsum;
            maxstartindex=currentstartindx;
            maxendindx=i;
        }
        if(currentsum<0)
        {
            currentsum=0;
            currentstartindx=i+1;
        }
    }
    id1=maxstartindex;
    id2=maxendindx;
    return maxsum;
}


int main()
{
    int N;
    int t;
    cin>>t;
    while(t--)
    {
        scanf("%d",&N);
        v.clear(); int x;
        bool fl=0;

        for(int i=0;i<N;i++)
        {
            scanf("%d",&x);
            v.push_back(x);
            if(x>0) fl=1;
        }
        int maxsum=findMaxsum();
        printf("%d %d %d\n",maxsum,id1+1,id2+1);
    }
    return 0;
}

/**

3
5
-22 2 -1 2 -1

*/



