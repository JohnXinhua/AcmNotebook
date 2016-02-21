#include<stdio.h>
#include<iostream>
#include<algorithm>
#include<string>

using namespace std;

bool com(const char x,const char y) {return x<y;} /// next_permutation.
//bool com(const char x,const char y) {return x>y;} /// prev_permutation.

int main()
{
    string str;
    while(cin>>str)
    {
        sort(str.begin(),str.end());      /// ignore sort Function for com added permutation.
        cout<<str<<endl;
        while(next_permutation(str.begin(),str.end(),com))
        {
            cout<<str<<endl;
        }
    }
    return 0;
}

