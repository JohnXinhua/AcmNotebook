#include <stdio.h>
#include <string.h>
#include <vector>
#include<iostream>
#include<algorithm>
using namespace std;

struct edge
{
    int a, b, cost ;
};

vector < edge > e ;
int n, m ;
const int INF = 1000000000 ;


void solve (int v,int t)
{
    vector < int > d ( n ,INF) ;
    vector < int > p ( n, - 1 ) ;

    int x=-1 ;
    d[v]=0;

    for ( int i = 0 ; i < n ; ++ i )
    {
        x = - 1 ;
        for (int j = 0 ; j < m ; ++ j )
        {
            if(d [ e[ j ] . a]< INF)
            {
                if ( d [ e [ j ] . b ] > d [ e [ j ] . a ] + e [ j ] . cost )
                {
                    d [ e [ j ] . b ] = max ( - INF, d [ e [ j ] . a ] + e [ j ] . cost ) ;
                    p [ e [ j ] . b ] = e [ j ] . a ;
                    x = e [ j ] . b ;
                }
            }
        }
    }

    if ( x == -1 ) cout <<"No negative cycle found." ;
    else
    {
        int y = x ;
        for (int i = 0 ; i < n ; ++ i )    y = p [ y ] ;

        vector < int > path ;
        for ( int cur = y ; ; cur = p [ cur ] )
        {
            path. push_back ( cur ) ;
            if ( cur == y && path. size ( ) > 1 )  break ;
        }
        reverse ( path. begin ( ) , path. end ( ) ) ;

        cout << "Negative cycle: " ;
        for ( size_t i = 0 ; i < path. size ( ) ; ++ i ) cout << path [ i ] << ' ' ;
    }
    return;
}


int main()
{
    int v,t;
    solve(v,t);
}

