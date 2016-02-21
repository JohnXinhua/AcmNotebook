#include<stdio.h>
#include<iostream>
#include<vector>
#include<queue>
#define sz 200
#define oo -1u/2
using namespace std;

#define MAX  1000



/*****************************************************************/


const int INF = 1000000000 ;
vector<pair<int,int> >g[MAX];

void dijkstra()
{
    int n,s,e; /// node,start,edge.
    cin>>n>>s;

	vector < int > d(n,INF), p(n);
	d[s]=0 ;
	vector<char>col(n);

	for (int i = 0 ; i < n ; ++ i )
    {
		int v = - 1 ;
		for (int j = 0 ; j < n ; ++ j )
		{
			if ( ! col[j] && (v==-1 || d[j]<d[ v ]) ) v = j ;
		}

		if ( d[ v ]==INF ) break ;
		col[v] = true ;

		for ( size_t j = 0 ; j < g[ v ] . size ( ) ; ++ j )
        {
			int to = g[v][j].first , len =g[ v ][j].second ;

			if ( d[ v ] + len < d[ to ] )
            {
				d[ to ] = d[ v ] + len ;
				p[ to ] = v ;
			}
		}
	}

	/*****
        vector < int > path ;
        for ( int v = t ; v ! = s ; v = p [ v ] ) path. push_back ( v ) ;

        path. push_back ( s ) ;
        reverse ( path. begin ( ) , path. end ( ) ) ;

    ****/

    return ;
}


int main()
{
    dijkstra();
	return 0;
}


