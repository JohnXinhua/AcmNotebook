#include<stdio.h>
#include<iostream>
#include<algorithm>
#define MAX 101
#define oo 1<<29
using namespace std;

int n,e;
int dis[sz][sz];

void fw(void)
{
    int i,j,k;
    for(k=1;k<=n;k++)
    {
        for(i=1;i<=n;i++)
        {
            for(j=1;j<=n;j++)
            {
                dis[i][j]=min(dis[i][j],(dis[i][k]+dis[k][j]));
            }
        }
    }
    return;
}

int main()
{
    int i,j,k,loop,d,x,y;
    scanf("%d",&loop);
    while(loop--)
    {
        scanf("%d %d",&n,&e);
        for(int i=0;i<=n;i++) for(int j=0;j<=n;j++)
        {
            if(i==j)dis[i][j]=0;
            else dis[i][j]=oo;
        }

        for(i=1;i<=e;i++)
        {
            scanf("%d %d %d",&x,&y,&d);
            dis[x][y]=d;
        }
        fw();
    }
    return 0;
}
