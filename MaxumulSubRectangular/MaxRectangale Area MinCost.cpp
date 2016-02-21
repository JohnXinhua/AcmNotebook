#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<ctype.h>
#include<string.h>
#include<iostream>
#include<vector>
#include<map>
#include<queue>
#include<stack>
#include<set>
#include<algorithm>
#include<sstream>
#include<time.h>
using namespace std;

#define popcount(a) __builtin_popcount(a)
#define mp          make_pair
#define pb          push_back
#define FOR(i,a,b)  for(int i=(a);i<=b;i++)
#define REV(i,a,b)  for(int i=a;i>=b;i--)
#define PI          (2*acos(0))
#define clr(a)       a.clear()
#define SZ(a)       (int)a.size()
#define all(a)      (a).begin(),(a).end()
#define reall(a)    (a).rbegin(),(a).rend()
#define fs          first
#define sc          second
#define mem(a,b)    memset(a,b,sizeof(a))
#define ERR         1e-7
#define EQ(a,b)     (fabs(a-b)<ERR)

//#define FOREACH(it,x) for(__typeof((x.begin())) it=x.begin();it!=x.end();it++)
//int rrr[]={1,0,-1,0};int ccc[]={0,1,0,-1};                      //4 Direction
//int rrr[]={1,1,0,-1,-1,-1,0,1};int ccc[]={0,1,1,1,0,-1,-1,-1};  //8 direction

typedef long long ll;
typedef unsigned long long ull; /// scanf("%llu",&N);
typedef vector<int>     VI;
typedef vector<string>  VS;
typedef pair<int,int>   PII;
typedef pair<int,PII >  DPII;
typedef vector<pair<int,int> >VPII;
typedef vector<pair<int,pair<int,int> > > VDPII;

#define INF (1<<28)
#define MAX 200009

// Maximal Sub-Rectangle Problem with n^3(using 2 pointer method)
// 11951 - Area(UVA).
// Maximum area with minimum cost
// K<=1e9

#define lim 110
int n,m,K;
int arr[lim][lim];
int cumarr[lim][lim];
int maxarea,mincost;

int update(int area,int cost)
{
    if(area>maxarea)
    {
        maxarea=area;
        mincost=cost;
    }
    else if(area==maxarea)
        mincost=min(mincost,cost);
    return 0;
}


int main()
{
    int t,cas=0;
    cin>>t;
    while(t--)
    {
        cin>>n>>m>>K;
        maxarea=0;
        mincost=0;
        mem(cumarr,0);
        int i,j,k,l;
        for(i=1; i<=n; i++)
            for(j=1; j<=m; j++)
                cin>>arr[i][j];

        for(i=1; i<=n; i++)
            for(j=1; j<=m; j++)
                cumarr[i][j]=cumarr[i][j-1]+arr[i][j];

        //selecting 2 columns
        for(i=1; i<=m; i++)
        {
            for(j=i; j<=m; j++)
            {
                l=1;
                int area=0,cost=0;
                //optimal row chosing by 2 pointer
                for(k=1; k<=n; k++)
                {
                    if(k!=1) area-=(j-i+1); //cancelling previous
                    cost-=(cumarr[k-1][j]-cumarr[k-1][i-1]);

                    while(l<=n)
                    {
                        if(cost+cumarr[l][j]-cumarr[l][i-1]>K)break;
                        area+=(j-i+1);
                        cost+=(cumarr[l][j]-cumarr[l][i-1]);
                        l++;
                    }
                    update(area,cost);
                }
            }
        }
        printf("Case #%d: ",++cas);
        cout<<maxarea<<" "<<mincost<<endl;
    }
    return 0;
}

/*

12
5 5 2000
0  2 3  3  4
5 2  0  3  1
6 8  3  5  2
0  2  3  4  5
3  4  6  7  8

**/
