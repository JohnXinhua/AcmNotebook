/*
    You are given a rooted tree, where each node contains an integer value.
    And the value of a node is strictly greater than the value of its parent.
    Now you are given a node and an integer query. You have to find the greatest possible parent of this node
    (may include the node itself), whose value if greater than or equal to the given query integer.
*/

#include <cstdio>
#include <cstring>
const int MAX = 100000;
const int LOG = 20;

int T[MAX], V[MAX];
int P[MAX][LOG];

int main()
{
    int test, cs, n, u, m, i, j, q, lg;
    V[0] = 1; T[0] = -1;
    scanf("%d", &test);
    for(cs = 1; cs <= test; cs++)
    {
        scanf("%d %d", &n, &q);
        for(i = 1; i < n; i++)
        {
            scanf("%d %d", &u, &m);  // ith node's parent and ith node's value.
            T[i] = u, V[i] = m;
        }

        for(i = 0; i < n; i++) P[i][0] = T[i];
        for(j = 1; (1 << j) < n; j++)
        {
            for(i = 0; i < n; i++)
            {
                if(P[i][j-1] != -1) P[i][j] = P[P[i][j-1]][j-1];
                else P[i][j] = -1;
            }
        }
        for(lg = 1; (1 << lg) < n; lg++);
        lg--;
        printf("Case %d:\n", cs);
        while(q--)
        {
            scanf("%d %d", &u, &m);
            for(j = lg; j >= 0; j--)
            {
                if(P[u][j] != -1 && V[P[u][j]] >= m)  u = P[u][j];
            }
            printf("%d\n", u);
        }
    }
    return 0;
}

