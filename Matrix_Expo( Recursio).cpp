#include<stdio.h>
#include<string.h>
#include<iostream>
#include<vector>
#include<algorithm>

#define FOR(i,a,b)  for(__typeof(a) i=(a);i<b;i++)
#define mem(a,b)    memset(a,b,sizeof(a))

using namespace std;

#define MAX 200009


/*****************************************************************************************************/


int Mod;

const int ROW=2; /// Change it matrix row and col.
struct matrix
{
    int A[ROW][ROW];
    matrix(){mem(A,0);}
    matrix(int a[ROW][ROW]){FOR(i,0,ROW)FOR(j,0,ROW)A[i][j]=a[i][j];}

    friend matrix operator * (const matrix &a ,const matrix &b)
    {
        matrix res;
//        FOR(i,0,N)FOR(j,0,N)FOR(k,0,N) res.A[i][j]=((a.A[i][k]*b.A[k][j]) + res.A[i][j]);
        FOR(i,0,ROW)FOR(j,0,ROW)FOR(k,0,ROW) res.A[i][j]=((a.A[i][k]*b.A[k][j]) + res.A[i][j])%Mod;
        return res;
    }
    friend matrix operator ^ (const matrix &a, const int k)
    {
        if(k==1) return a;
        matrix tmp=a^(k/2);
        if(k&1) return (tmp*tmp)*a;
        else return tmp*tmp;
    }
    void view()
    {
        FOR(i,0,ROW){FOR(j,0,ROW)cout<<A[i][j]<<" ";puts("");}
    }
};

matrix Base;
void makeBase()
{

    Base.A[0][0]=1;
    Base.A[0][1]=1;
    Base.A[1][0]=1;
    Base.A[1][1]=0;
    return;
}



/************************************************************************************/




int main()
{

//    freopen("input.txt", "r", stdin);
//    freopen("output.txt", "w", stdout);
    makeBase();

    int t,cas=0;
    scanf("%d",&t);
    while(t--)
    {
        int a,b,m,n;
        scanf("%d %d %d %d",&a,&b,&n,&m); Mod=1;
        for(int i=1;i<=m;i++) Mod*=10;

        printf("Case %d: ",++cas);

        if(n==0) cout<<(a)%Mod<<endl;
        else if(n==1) cout<<(b)%Mod<<endl;

        matrix res=Base^(n-1);
        int cnt=0;
        FOR(i,0,2) cnt=(res.A[0][0]*b + res.A[0][1]*a)%Mod;
        cout<<cnt<<endl;
    }

    return 0;
}

