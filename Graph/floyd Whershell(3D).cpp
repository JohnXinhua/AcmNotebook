/// three dimensional
#include<stdio.h>
#include<iostream>
#include<algorithm>
#define sz 100
#define oo 1<<29
using namespace std;
int n,e;
int dis[sz][sz][sz];

void reset()
{
    for(int k=0; k<=n; k++)
    {
        for(int i=0; i<=n; i++)
        {
            for(int j=0; j<=n; j++)
            {
                if(i==j)dis[0][i][j]=0;
                else dis[0][i][j]=oo;
            }
        }
    }
}

void view()
{
    puts("\n\n\n");
    for(int i=1; i<=n; i++)
    {
        for(int j=1; j<=n; j++)
        {

            printf("%10d   ",dis[n][i][j]);
        }
        printf("\n");
    }

    puts("\n\n");
}

void fw()
{
    int i,j,k,p;
    for(k=1; k<=n; k++)
    {
        for(i=1; i<=n; i++)
        {
            for(j=1; j<=n; j++)
            {
                dis[k][i][j]=min(dis[k-1][i][j],(dis[k-1][i][k]+dis[k-1][k][j]));
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
        reset();
        for(i=1; i<=e; i++)
        {
            scanf("%d %d %d",&x,&y,&d);
            dis[0][x][y]=d;
        }
        fw();
        view();
    }
}
