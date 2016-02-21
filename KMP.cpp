#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace std;

#define MAX 2000006
int b[MAX];

int prefix(string s)
{
    int i=1;int j=0;b[j]=0;
    while(i<(int)(s.size()))
    {
        while(j>0 and s[i]!=s[j]) j=b[j-1];
        if(s[i]==s[j])j++;
        b[i]=j;i++;
    }
    return 0;
}


int main()
{
    int t;
    scanf("%d",&t);
    while(t--)
    {
        string s;
        cin>>s;
        int ans=prefix(s);
        puts("\n----- Kmp complete ----- ");
    }
    return 0;
}

