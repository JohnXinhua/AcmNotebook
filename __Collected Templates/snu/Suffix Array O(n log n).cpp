#include <cstdio>
#include <algorithm>
using namespace std;
int n, K;
int dat[20003];
int ians[20003];
int ans[20003];
int tmpans[20003];
int bucket[20003];
int bucketcnt[20003];
int cntbucket;
int bucketmark[20003];
int bucketupdate[20003];
inline int sf(const int& a, const int& b)
{
    return dat[a] < dat[b];
}
int main()
{
    int i, H;
    scanf("%d%d", &n, &K);
    for (i = 0 ; i < n ; i++)
    {
        scanf("%d", &dat[i]);
        dat[i]++;
        ans[i] = i;
        ians[i] = i;
        Seoul National University  24
    }
    // constructing suffix array by doubling method
    // phase 1: init
    sort(ans, ans + n, sf);
    for (i = 0 ; i < n ; i++)
    {
        if (i == 0 || dat[ans[i]] != dat[ans[i - 1]])
        {
            bucket[cntbucket] = i;
            bucketcnt[cntbucket] = 0;
            cntbucket++;
        }
        bucketmark[ans[i]] = cntbucket - 1;
    }

    // phase 2: doubling
    for (H = 1 ; ; H *= 2)
    {
        // phase 2-1: rearrangement
        for (i = 0 ; i < n ; i++)
        {
            if (ans[i] >= n - H)
            {
                int tbuck = bucketmark[ans[i]];
                bucketupdate[ans[i]] = -1;
                tmpans[bucket[tbuck] + bucketcnt[tbuck]] = ans[i];
                bucketcnt[tbuck]++;
            }
        }
        for (i = 0 ; i < n ; i++)
        {
            if (ans[i] >= H)
            {
                int tbuck = bucketmark[ans[i] - H];
                bucketupdate[ans[i] - H] = bucketmark[ans[i]];
                tmpans[bucket[tbuck] + bucketcnt[tbuck]] = ans[i] - H;
                bucketcnt[tbuck]++;
            }
        }

        // phase 2-2: identify new buckets
        int lastbucket = bucketmark[tmpans[0]];
        for (i = 1 ; i < n ; i++)
        {
            if (bucket[bucketmark[tmpans[i]]] != i)
            {
                if (bucketupdate[tmpans[i]] != bucketupdate[tmpans[i - 1]])
                {
                    // found new bucket
                    bucket[cntbucket] = i;
                    lastbucket = cntbucket;
                    cntbucket++;
                }
            }
            else
            {
                lastbucket = bucketmark[tmpans[i]];
            }
            bucketmark[tmpans[i]] = lastbucket;
        }

        // phase 2-3: copy ans and calculate ians
        int flg = 0;
        bucketmark[n] = -1;
        for (i = 0 ; i < n ; i++)
        {
            if(bucketmark[tmpans[i]] == bucketmark[tmpans[i + 1]]) flg = 1;
            ans[i] = tmpans[i];
            ians[ans[i]] = i;
            bucketcnt[bucketmark[ans[i]]] = 0;
        }
        if (flg == 0) break;
    }
    return 0;
}


