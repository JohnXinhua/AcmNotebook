#include<stdio.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string.h>
using namespace std;

typedef struct
{
	int v; // end vertex
	int w; // weight of edge
} Edge;

int N; // No of vertices
int n_edges; // No of edges

vector< vector< Edge > > adjlist; // Graph represented as an adjacency list
vector< int > isolated_vert; // List of isolated vertices having indegree 0

int indegree[N]; 	// Populate with indegree of every vertex
int cur_vert, sz;

void TPO()
{
    while(isolated_vert.size() != 0)
    {
        cur_vert = isolated_vert[0];
        sz = adjlist[cur_vert].size();

        for(int e = 0; e < sz; e++)
        {
            /* Remove all outgoing edges */
            indegree[ adjlist[cur_vert][e].v ] -= 1;
            n_edges -= 1;

            /* If vertex has indegree 0 add to list of isolated vertices */
            if(indegree[adjlist[cur_vert][e].v] == 0)
            {
                isolated_vert.push_back( adjlist[cur_vert][e].v );
            }
        }
        isolated_vert.erase(isolated_vert.begin());
    }

    /* Graph still has edges */
    if(n_edges > 0)
    {
        cout<< "Cycle found" << endl;
    }
    else cout<< "No Cycle found" << endl;
    return;
}

int main()
{
    // make graph;
    return 0;
}

