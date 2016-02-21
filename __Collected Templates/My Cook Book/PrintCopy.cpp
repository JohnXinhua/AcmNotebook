Eulear Tour And Circuit
Sliding Window
Kadane Algorithm 1D
Kadane Algorithm 2D
Kadane Algorithm 3D
SuffixArray (nlogn)

Eulear Tour And Circuit
=============================
void dfs(int u)
{
    int i,v;
    for(i=0; i<SZ(adj[u]); i++)
    {
        v = adj[u][i];
        if(v!=-1)
        {
            adj[u][i] = -1;
            dfs(v);
        }
    }
    order.pb(u);
}
bool possible()
{
    int i,start,end,c=0;
    start = end = -1;
    for(i=0; i<nodes; i++)
    {
        if(indeg[i]==outdeg[i]) continue;
        else if(indeg[i]-outdeg[i]==1)
        {
            end = i;
            c++;
        }
        else if(outdeg[i]-indeg[i]==1)
        {
            start = i;
            c++;
        }
        else return false;
    }
    if(c>2) return false;
    if(start == -1)   //circuit probably
    {
        for(i=0; i<nodes; i++)
            if(outdeg[i])
            {
                start = i;
                break;
            }
    }
    order.clear(); //Here Finding the
    dfs(start);//Eulear tour orderings.
    Reverse(order);
    if(SZ(order)!=nodes) return false;
//could be disconnected.
    return true;
}



Sliding Window
=================
/*
given two integers n and k which are the lengths
of the array and the sliding window.Output the minimum values
in the window at each position, from left to right, respectively.
*/
const int MAX=100009;
int store[MAX];
int Q[MAX],head,tail;
vector<int>MinValu;

void SlidingWindowMin(int N,int K)
{
    //N=number of elements.
    //K=windows size.
    MinValu.clear();
    head=0;tail=0;
    for(int i=1;i<=N;i++)
    {
        while(head!=tail && Q[head]<=i-K) head++;// Cheack window size.
        while(head!=tail && store[Q[tail-1]]>=store[i]) tail--; // truncate the previous bigger value.
        Q[tail++]=i; // insert new elements.
        if(i>=K) MinValu.pb(store[Q[head]]);
    }
    return;
}

int main()
{
    int N,K;
    scanf("%d %d",&N,&K);
    for(int i=1;i<=N;i++) scanf("%d",&store[i]);
    SlidingWindowMin(N,K); // Min value in given Window.
}

Kadane Algorithm 1D
====================
vector<int>v;int N;
int id1,id2;
int findMaxsum()
{
    int maxsum=-(1<<30);
    int currentstartindx=0,maxstartindex,maxendindx,currentsum=0;
    for(int i=0;i<SZ(v);i++)
    {
        currentsum+=v[i];
        if(currentsum>maxsum)
        {
            maxsum=currentsum;
            maxstartindex=currentstartindx;
            maxendindx=i;
        }
        if(currentsum<0)
        {
            currentsum=0;
            currentstartindx=i+1;
        }
    }
    id1=maxstartindex;
    id2=maxendindx;
    return maxsum;
}
int main()
{
    scanf("%d",&N);
    for(int i=0,x;i<N;i++) {
        v.push_back(x);
    }
    int maxsum=findMaxsum();
}

Kadane Algorithm 2D
=========================================
const int MAX=109;
int grid[MAX][MAX],sum[MAX][MAX],N;

int main()
{
    while(scanf("%d",&N)==1)
    {
        sum[1][0]=0;
        for(int i=1;i<=N;i++)
            for(int j=1;j<=N;j++)
            {
                scanf("%d",&grid[i][j]);
                sum[i][j]=sum[i][j-1]+grid[i][j];
            }

        // Khadane 2D.
        int ans=-INF;
        for(int colum1=1;colum1<=N;colum1++) // select first colume.
            for(int colum2=colum1;colum2<=N;colum2++) // select second colume.
            {
                //khadane 1-D
                int val=0;
                for(int row=1;row<=N;row++)
                {
                    val+=sum[row][colum2]-sum[row][colum1-1];
                    ans=max(ans,val);
                    if(val<0) val=0;
                }
            }
        printf("%d\n",ans);
    }
    return 0;
}

Kadane Algorithm 3D
==========================================
const int MAX=100;
int X,Y,Z;
ll grid[MAX][MAX][MAX];
ll sum[MAX][MAX][MAX];

int main()
{
    int t;cin>>t;
    while(t--)
    {
        cin>>X>>Y>>Z;X++;Y++;Z++;
        FOR(i,1,X)FOR(j,1,Y)FOR(k,1,Z) scanf("%lld",&grid[i][j][k]);
        mem(sum,0);
        FOR(i,1,X)FOR(j,1,Y)FOR(k,1,Z) sum[i][j][k]=sum[i][j][k-1]+grid[i][j][k];
        FOR(i,1,X)FOR(j,1,Y)FOR(k,1,Z) sum[i][j][k]+=sum[i-1][j][k];

        // khadane 3d
        ll ans=-INF;
        for(int z1=1;z1<Z;z1++)
            for(int z2=z1;z2<Z;z2++)
                for(int x1=1;x1<X;x1++)
                    for(int x2=x1;x2<X;x2++)
                    {
                        ll val=0;
                        for(int y=1;y<Y;y++)
                        {
                            ll tmp=sum[x2][y][z2]-(sum[x2][y][z1-1]+sum[x1-1][y][z2])+sum[x1-1][y][z1-1];
                            val+=tmp;
                            ans=max(val,ans);
                            if(val<0) val=0;
                        }
                    }
        printf("%lld\n",ans);
    }

    return 0;
}

SuffixArray (nlogn)
===================================
const int MAX=110009;
int SA[MAX],rank[MAX],lcp[MAX];
int rank2[MAX],*X,*Y,cnt[MAX];

bool cmp(int* r,int a,int b,int l,int n)
{
    if(r[a]==r[b]&&a+l<n&&b+l<n&&r[a+l]==r[b+l]) return true;
    return false;
}

void radix_sort(int n,int alphabet)
{
    for(int i=0; i<alphabet; i++) cnt[i]=0;
    for(int i=0; i<n; i++) cnt[X[Y[i]]]++;
    for(int i=1; i<alphabet; i++) cnt[i]+=cnt[i-1];
    for(int i=n-1; i>=0; i--) SA[--cnt[X[Y[i]]]]=Y[i];
}
void buildsa(const string &str,int n,int alphabet)
{
    X=rank,Y=rank2;
    for(int i=0; i<n; i++) X[i]=str[i],Y[i]=i;
    radix_sort(n,alphabet);
    for(int len=1; len<n; len<<=1)
    {
        int yid=0;
        for(int i=n-len; i<n; i++) Y[yid++]=i;
        for(int i=0; i<n; i++) if(SA[i]>=len) Y[yid++]=SA[i]-len;
        radix_sort(n,alphabet);
        swap(X,Y);
        X[SA[0]]=yid=0;
        for(int i=1; i<n; i++)
        {
            X[SA[i]]=cmp(Y,SA[i],SA[i-1],len,n)?yid:++yid;
        }
        alphabet=yid+1;
        if(alphabet>=n) break;
    }
    for(int i=0; i<n; i++) rank[i]=X[i];
}
void buildlcp(const string &str,int n)
{
    int k=0;
    lcp[0]=0;
    for(int i=0; i<n; i++)
    {
        if(rank[i]==0) continue;
        k=max(k-1,0);
        int j=SA[rank[i]-1];
        while(i+k<n && j+k<n && str[i+k]==str[j+k] ) k++;
        lcp[rank[i]]=k;
    }
    return;
}

int in,n,limit;
int idx[MAX],col[200];
int ok(const string &str,int mid)
{
    int i,j,k;
    for(int i=1;i<n;i=j+1)
    {
        while(lcp[i]<mid and i<n) i++;
        if(i>=n) break;
        mem(col,0);int ans=0;
        col[idx[SA[i-1]]]++; ans++;

        for(j=i;j<n and lcp[j]>=mid;j++)
        {
            int id=idx[SA[j]];
            if(!col[id])
            {
                col[id]++;
                ans++;
            }
        }
        if(ans>in/2) return true;
    }
    return false;
}

void solve(string &str,int n)
{
    int lo=1,hi=limit;
    while(lo<=hi)
    {
        int mid=(lo+hi)/2;
        if(ok(str,mid))
        {
            ans=mid;
            lo=mid+1;
        }
        else hi=mid-1;
    }
    return ans;
}

int main()
{
    string str;
    cin>>str;
    n=SZ(str);
    buildsa(str,n,250);
    buildlcp(str,n);
    cout<<solve(str,n)<<endl;
}
