
#define MAX 1000009

bool flag[MAX];
int prime[MAX];

void sive()
{
    int i,j,k,r,total=1;
    flag[0]=flag[1]=1;
    prime[total]=2;

    for(i=3;i<MAX;i+=2)
    {
        if(!flag[i])
        {
            prime[++total]=i;
            r=i*2;
            if(MAX/i>=i)for(j=i*i;j<MAX;j+=r)flag[j]=1;
        }
    }
    printf("Last Prime (Mind it): %d\n\n",prime[total]);
    return;
}

bool isPrime(int n)
{
    if(n==2)return 1;
    if(n<MAX){if((n%2) && !flag[n]) return 1;}
    for(int i=1;i<=sqrt(n);i++)
    {
        if(n%prime[i]==0) return 0;
    }
    return 1;
}
