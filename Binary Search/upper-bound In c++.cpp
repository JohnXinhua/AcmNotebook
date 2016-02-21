#include<stdio.h>
#include<iostream>
#include<algorithm>
#include<vector>

#define pb push_back

using namespace std;

int  main()
{
    vector<int>v;
    v.pb(10); v.pb(20);v.pb(20);v.pb(30);v.pb(30);v.pb(30);v.pb(40);v.pb(50);v.pb(60);

    for(int i=0;i<v.size();i++) cout<<i<<" "<<v[i]<<endl;
    cout<<"Size:--->> "<<v.size()<<endl;

    /*

        normaly return iterator.
        By doing this you can access index. if geter value from last value return the size of vector.
        if smaller then first value return 0.

        must cheack again if return 0.
    */
    while(1)
    {
        int x;
        cin>>x;
        int upper=(upper_bound(v.begin(),v.end(),x) - v.begin());

        cout<<"Upper:---->>> "<<upper<<endl;
    }
    return 0;
}


