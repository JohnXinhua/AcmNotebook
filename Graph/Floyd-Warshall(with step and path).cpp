#include<stdio.h>
#include<iostream>
#define MAX 109
using namespace std;
double grid[MAX][MAX][MAX];
int path[MAX][MAX][MAX];
int N;

void output(int start,int end,int step)
{
    if(step>=2)
    {
        output(start,path[start][end][step],step-1);
        cout<<" "<<path[start][end][step];
    }
    return;
}

void Floydwarshall(void)
{
    bool flag=true;int start,steps,end;

    for(int step=2;step<=N and flag;step++)
    {
        for(int k=1;k<=N and flag ;k++)
            for(int i=1;i<=N and flag ; i++)
                for(int j=1;flag and j<=N;j++)
                    {
                        double tmp=grid[i][k][step-1]*grid[k][j][1];  /// change as needed.
                        if(tmp>grid[i][j][step])
                        {
                            grid[i][j][step]=tmp;
                            path[i][j][step]=k;

                            if(i==j  and grid[i][j][step]>1.01)  /// i==j means start and end node same.change as needed.
                            {
                                start=i;
                                steps=step;
                                flag=false;
                            }
                        }
                    }
    }

    /// path print.
    if(!flag)
    {
        cout<<start;
        output(start,start,steps);
        cout<<" "<<start<<endl;
    }
    else printf("no arbitrage sequence exists\n");
}


int main()
{
    while(cin>>N)
    {
        for(int i=1;i<=N;i++)
            for(int j=1;j<=N;j++)
                for(int step=1;step<=N;step++) grid[i][j][step]=0.0; // intialization.

        for(int i=1;i<=N;i++)
        {
            for(int j=1;j<=N;j++)
            {
                if(i==j) grid[i][j][1]=0.0;  /// change as needed.
                else
                {
                    cin>>grid[i][j][1];
                    path[i][j][1]=i;  /// for path frint.
                }
            }
        }

        Floydwarshall();
    }
    return 0;
}

