#include<stdio.h>
#include<iostream>
#include<algorithm>
#include<vector>

#define pb push_back

using namespace std;

int  main()
{
    vector<int>v;
    v.pb(10); v.pb(20);v.pb(20);v.pb(30);v.pb(30);v.pb(30);v.pb(40);v.pb(50);v.pb(60);v.pb(60);

    for(int i=0;i<v.size();i++) cout<<i<<"  :::: "<<v[i]<<endl;

    cout<<"Size:--->> "<<v.size()<<endl;
    /*

        normaly return iterator.
        By doing this you access index. if geter value from last value return the size of vector.
        if smaller then first value return 0.
    */

    int N;
    while(cin>>N)
    {
        int low=(lower_bound(v.begin(),v.end(),N) - v.begin());

        cout<<"Lower:---->>> "<<low<<endl;
    }
    return 0;
}
