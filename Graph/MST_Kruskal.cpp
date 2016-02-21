#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#define MAX 100

int N;int E_count;
vector<pair<int,pair<int,int> > >node;

/// Copy below code for MST_Kruskal.

int rank[MAX],par[MAX];
void Make_set(int x) { par[x]=x; rank[x]=0; return; }
int Find_set(int x) { if(x!=par[x]) par[x]=Find_set(par[x]); return par[x]; }

void Union_set(int x,int y)
{
    if(rank[x]>rank[y]) par[y]=x;
    else par[x]=y;
    if(rank[x]==rank[y]) rank[x]++;
    return;
}

void Link(int x,int y) {  Union_set(Find_set(x),Find_set(y)); return; }

int MST()
{
    int i,u,v;int cnt=0;
    for(i=1;i<=N;i++) Make_set(i);
    for(i=0;i<SZ(node);i++)
    {
        u=node[i].sc.fs;
        v=node[i].sc.sc;
        if(Find_set(u)!=Find_set(v))
        {
            Link(u,v);
            E_count++;
            cnt+=node[i].fs;
        }
    }
    return cnt;
}

int main()
{
    int i,j,k,cas=0,t;
    scanf("%d",&t);
    while(t--)
    {
        scanf("%d",&N);
        int cnt=0; node.clear();

        for(i=1;i<=N;i++)
        {
            for(j=1;j<=N;j++)
            {
                scanf("%d",&k);
                if(k)
                {
                    cnt+=k;
                    node.pb(mp(k,mp(i,j)));node.pb(mp(k,mp(j,i)));
                }
            }
        }
        sort(all(node));
        E_count=0;
        int ans=MST();
        ans=cnt-ans;
        if(E_count<N-1) printf("Case %d: %d\n",++cas,-1);
        else printf("Case %d: %d\n",++cas,ans);
    }
    return 0;
}

