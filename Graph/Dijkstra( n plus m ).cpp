#include<stdio.h>
#include<iostream>
#include<vector>
#include<queue>
#define sz 200
#define oo -1u/2
using namespace std;
struct edge {int n,w;};
int n,dis[sz];
vector<edge>v[sz];

//priority_queue<pair<int,int> ,vector<pair<int,int> >,greater<pair<int,int> > > Q; // min priority
//priority_queue<pair<int,int> > Q; // max priority


struct qnode
{
    int qn, qw;
    qnode(){;}
    qnode(int _qn,int _qw){ qn=_qn;qw=_qw;}
    bool operator<(const qnode &x)const {return x.qw<qw;}
};

bool man(int u,int v,int w)
{
    if(dis[v]>dis[u]+w)
    {
        dis[v]=dis[u]+w;
        return true;
    }
    return false;
}

void dijkstra(int s,int t)
{
    int i,j,k;
    struct qnode tmp;
    priority_queue<qnode>Q;
    for(i=0;i<=n;i++)dis[i]=oo;

    dis[s]=0;
    tmp.qn=s;
    tmp.qw=0;
    Q.push(tmp);

    while(!Q.empty())
    {
        tmp=Q.top();Q.pop();
        k=tmp.qn;
        if(k==t) break; /// t=targate and if you want to evalute only source to targate then add this.
        if(tmp.qw>dis[k]) continue;

        for(i=0;i<v[k].size();i++)
        {
            if(man(k,v[k][i].n,v[k][i].w))
            {
                tmp.qn=v[k][i].n;
                tmp.qw=dis[v[k][i].n];
                Q.push(tmp);
            }
        }
    }
}


/*********************************************************************************************************/


int main()
{
    int i,j,k,x,y,w,loop,c=0;
    struct edge tmp;
    scanf("%d",&loop);

    while(loop--)
    {
        int s,t,m;
        scanf("%d %d %d %d",&n, &m,&s,&t); ///s=start node; t=end node; n=num of node; m=Num of edge;
        for(i=0;i<n;i++)v[i].clear();

        for(i=0;i<m;i++)
        {
            scanf("%d %d %d",&x,&y,&w);   /// x=node; y=node; w=weihht;
            tmp.n=y;
            tmp.w=w;
            v[x].push_back(tmp);
            tmp.n=x;
            v[y].push_back(tmp);
        }
        dijkstra(s,t);

        if(dis[t]==oo)printf("No Entry\n");
        else printf("Distance= %d\n",dis[t]);
    }
    return 0;
}
