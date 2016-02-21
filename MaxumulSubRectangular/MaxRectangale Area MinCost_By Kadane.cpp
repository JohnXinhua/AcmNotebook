#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <iostream>
using namespace std;
#define MAX 100
#define INF (1<<28)

// Implementation of Kadane's algorithm for 1D array. The function returns the
// maximum area  smaller than K and stores starting and ending indexes with mincost
// at addresses pointed by start and finish pointers respectively.

int n,m,K; // K<=1e9
int ROW,COL;
int M[MAX][MAX];

int kadane(int* arr, int* start, int* finish, int n)
{
    // initialize sum, maxSum and
    int sum = 0, TotalSum = INT_MIN, i;
    // Just some initial value to check for all negative values case
    *finish = -1;
    // local variable
    int local_start = 0;
    for (i = 0; i < n; ++i)
    {
        sum += arr[i];
        if (sum<=K)
        {
            TotalSum = sum;
            *start = local_start;
            *finish = i;
        }
    }
    if(TotalSum==INT_MIN) return (INF);
    return TotalSum;
}

// The main function that finds maximum sum rectangle in M[][]
void findMaxSum(int M[][MAX])
{
    // Variables to store the final output
    int finalLeft, finalRight, finalTop, finalBottom;
    int left, right, i;
    int temp[ROW], sum, start, finish;

    int MaxArea=0;
    int minSum =(1<<28);
    // Set the left column
    for (left = 0; left < COL; ++left)
    {
        // Initialize all elements of temp as 0
        memset(temp, 0, sizeof(temp));

        // Set the right column for the left column set by outer loop
        for (right = left; right < COL; ++right)
        {
            // Calculate sum between current left and right for every row 'i'
            for (i = 0; i < ROW; ++i) temp[i] += M[i][right];

            sum = kadane(temp, &start, &finish, ROW);
            if(sum==INF) continue;

            int area=(finish-start+1)*(right-left+1);

            if(area>MaxArea)
            {
                MaxArea=area;
                finalLeft = left;
                finalRight = right;
                finalTop = start;
                finalBottom = finish;
                minSum=sum;
            }
            else if(area==MaxArea)
            {
                if(sum<minSum)
                {
                    finalLeft = left;
                    finalRight = right;
                    finalTop = start;
                    finalBottom = finish;
                    minSum=sum;
                }
            }
        }
    }

    cout<<MaxArea<<" "<<minSum<<endl;
    return;

    if(MaxArea==0)
    {
        cout<<"No Solution Exits!!"<<endl;
        return ;
    }

    // Print final values
    printf("( Top, Left )   (%d, %d)\n", finalTop+1, finalLeft+1); // (+1) for convert to 1 base idx.
    printf("( Bottom, Right) (%d, %d)\n", finalBottom+1, finalRight+1);
    printf("( MaxArea, MinCost )  (%d %d)\n", MaxArea,minSum);
}

void view()
{
    puts("\n");
    for(int i=0;i<ROW;i++)
    {
        for(int j=0;j<COL;j++) printf("%3d",M[i][j]);
        puts("");
    }
    puts("----- Finished -------\n\n");
}

// Driver program to test above functions
int main()
{
    while(cin>>n>>m>>K)
    {
        ROW=n;COL=m;
        memset(M,0,sizeof(M));

        for(int i=0;i<ROW;i++)for(int j=0;j<COL;j++) cin>>M[i][j];

        findMaxSum(M);
    }
    return 0;
}

/**

5 5 30
19  2 3  3  4
5 2  0  3  1
6 8  3  5  2
0  2  3  4  5
3  4  6  7  8

5 5 3
-19  -2 -3  -3  -4
-5 -2  -0  -3  -1
-6 -8  -3  -5  -2
-0  -2  -3  -4  -5
-3  -4  -6  -7  -8

3 3 -222
19  2  3
-5  2  0
6  8  3

3  3  3
19  2  3
-5 -1  0
6   8  3

*/


