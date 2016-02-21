#include<stdio.h>
#include<string.h>
#include<iostream>
#include<vector>
using namespace std;

vector<int>g[100],gr[100] ;
vector<char>used;
vector<int>order,component ;
int n,e;

void dfs1(int v)
{
	used[v]=true ;
	for(size_t i=0; i<g[v].size(); ++i)
    {
		if(!used[g[v][i]]) dfs1(g[v][i]);
    }
	order.push_back(v);
}

void dfs2(int v)
{
	used[v]=true ;
	component.push_back(v);
	for(size_t i=0; i<gr[v].size(); ++i)
    {
        if(!used[gr[v][i]]) dfs2(gr[v][i]) ;
    }
    return;
}

void SCC()
{
    used.assign(n,false) ;
	for(int i = 0 ; i<n; ++i) if (!used[i]) dfs1(i);
    //for(int i=0;i<order.size();i++) cout<<order[i]<<"  "; puts("");

	used.assign(n,false) ;
	int compo=0;
	for (int i=0; i<n; ++i)
    {
		int v = order[n-1-i] ;
		if (!used[v])
        {
			dfs2(v) ;
			compo++;
			cout<<"Components num : "<<compo<<endl;
			for(int i=0;i<component.size();i++)
            {
                cout<<" "<<component[i];
            }
            puts("\n");
			component.clear();
		}
	}
	return;
}

int main ()
{
    cin>>n>>e;
	for (int i=0;i<e;i++)
    {
		int a, b ;
		cin>>a>>b;
		g[a].push_back(b) ;
		gr[b].push_back(a) ; /// Transpose G
	}
    SCC();
	return 0;
}

