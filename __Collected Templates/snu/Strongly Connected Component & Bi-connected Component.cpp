#include <cstdlib>
#include <vector>
using namespace std;

namespace cc
{
    const int SIZE = 10000;
    vector<int> graph[SIZE];
    int connected[SIZE];
    int cut_vertex_num;
    bool cut_vertex[SIZE];
    int cut_edge_num, cut_edge[SIZE][2];
    int order[SIZE];
    int visit_time[SIZE], finish[SIZE], back[SIZE];
    int stack[SIZE], seen[SIZE];

    #define MIN(a,b) (a) = ((a)<(b))?(a):(b)
    int dfs(int size)
    {
        int top, cnt, cnt2, cnt3;
        int i;
        cnt = cnt2 = cnt3 = 0;
        stack[0] = 0;
        for (i = 0 ; i < size ; i++) visit_time[i] = -1;
        for (i = 0 ; i < size ; i++) cut_vertex[i] = false; // CUT VERTEX
        cut_edge_num = 0; // CUT_EDGE
        for (i = 0 ; i < size ; i++)
        {
            if (visit_time[order[i]] == -1)
            {
                top = 1;
                stack[top] = order[i];
                seen[top] = 0;
                visit_time[order[i]] = cnt++;
                connected[order[i]] = cnt3++;
                int root_child = 0; // CUT VERTEX
                while (top > 0)
                {
                    int j, now = stack[top];
                    if (seen[top] == 0) back[now] = visit_time[now]; // NOT FOR SCC
                    for (j = seen[top] ; j < graph[now].size() ; j++)
                    {
                        int next = graph[now][j];
                        if (visit_time[next] == -1)
                        {
                            if (top == 1) root_child++; // CUT VERTEX
                            seen[top] = j + 1;
                            stack[++top] = next;
                            seen[top] = 0;
                            visit_time[next] = cnt++;
                            connected[next] = connected[now];
                            break;
                        }
                        else if (top == 1 || next != stack[top - 1]) // NOT FOR SCC
                            MIN(back[now], visit_time[next]); // NOT FOR SCC
                    }
                    if (j == graph[now].size())
                    {
                        finish[cnt2++] = now; // NOT FOR BCC
                        top--;
                        if (top > 1)
                        {
                            MIN(back[stack[top]], back[now]); // NOT FOR SCC
                            if (back[now] >= visit_time[stack[top]])   // CUT VERTEX
                            {
                                cut_vertex[stack[top]] = true;
                                cut_vertex_num++;
                            }
                        }
                        // CUT EDGE
                        if (top > 0 && visit_time[stack[top]] < back[now])
                        {
                            cut_edge[cut_edge_num][0] = stack[top];
                            cut_edge[cut_edge_num][1] = now;
                            cut_edge_num++;
                        }
                    }
                }
                if (root_child > 1)   // CUT VERTEX
                {
                    cut_vertex[order[i]] = true;
                    cut_vertex_num++;
                }
            }
        }
        return cnt3; // number of connected component
    }
    #undef MIN

    vector<int> graph_rev[SIZE];
    void graph_reverse(int size)
    {
        for (int i = 0 ; i < size ; i++) graph_rev[i].clear();
        for (int i = 0 ; i < size ; i++)
            for (int j = 0 ; j < graph[i].size() ; j++)
                graph_rev[graph[i][j]].push_back(i);
        for (int i = 0 ; i < size ; i++) graph[i] = graph_rev[i];
    }

    int scc(int size)
    {
        int n;
        for (int i = 0 ; i < size ; i++) order[i] = i;
        dfs(size);
        graph_reverse(size);
        for (int i = 0 ; i < size ; i++) order[i] = finish[size - i - 1];
        n = dfs(size);
        graph_reverse(size);
        return n;
    }
    void bcc(int size)
    {
        for (int i = 0 ; i < size ; i++) order [ i ] = i;
        dfs(size);
        cut_vertex_num = 0;
        for (int i = 0 ; i < size ; i++)
            if (cut_vertex[i])
                cut_vertex_num++;
    }
} // namespace cc


