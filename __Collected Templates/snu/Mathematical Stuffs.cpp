#include <cmath>
#include <climits>
#include <vector>
#include <algorithm>
using namespace std;

//Modular Power:  n^k mod m
long long power(long long n, long long k, long long m = LLONG_MAX)
{
    long long ret = 1;
    while (k)
    {
        if (k & 1) ret = (ret * n) % m;
        n = (n * n) % m;
        k >>= 1;
    }
    return ret;
}
// GCD
long long gcd(long long a, long long b)
{
    if (b == 0) return a;
    return gcd(b, a % b);
}
//Extended GCD: ac + bd = gcd(a, b)
pair<long long, long long> extended_gcd(long long a, long long b)
{
    if (b == 0) return make_pair(1, 0);
    pair<long long, long long> t = extended_gcd(b, a % b);
    return make_pair(t.second, t.first - t.second * (a / b));
}

// Modular Inverse: ax = gcd(a, m) (mod m)
long long modinverse(long long a, long long m)
{
    return (extended_gcd(a, m).first % m + m) % m;
}

// Chinese Remainder Theorem: x = a (mod n)
// Dependencies: gcd(a, b), modinverse(a, m)

    long long chinese_remainder(long long *a, long long *n, int size)
{
    if (size == 1) return *a;
    long long tmp = modinverse(n[0], n[1]);
    long long tmp2 = (tmp * (a[1] - a[0]) % n[1] + n[1]) % n[1];
    long long ora = a[1];
    long long tgcd = gcd(n[0], n[1]);
    a[1] = a[0] + n[0] / tgcd * tmp2;
    n[1] *= n[0] / tgcd;
    long long ret = chinese_remainder(a + 1, n + 1, size - 1);
    n[1] /= n[0] / tgcd;
    a[1] = ora;
    return ret;
}

// Binomial Calculation: nCm
long long binomial(int n, int m)
{
    if (n > m || n < 0) return 0;
    long long ans = 1, ans2 = 1;
    for (int i = 0 ; i < m ; i++)
    {
        ans *= n - i;
        ans2 *= i + 1;
    }
    return ans / ans2;
}
//Lucas Theorem: nCm mod p
int lucas_theorem(const char *n, const char *m, int p)
{
    vector<int> np, mp;
    int i;
    for (i = 0 ; n[i] ; i++)
    {
        if (n[i] == '0' && np.empty()) continue;
        np.push_back(n[i] - '0');
    }
    for (i = 0 ; m[i] ; i++)
    {
        if (m[i] == '0' && mp.empty()) continue;
        mp.push_back(m[i] - '0');
        Seoul National University  13
    }
    int ret = 1;
    int ni = 0, mi = 0;
    while (ni < np.size() || mi < mp.size())
    {
        int nmod = 0, mmod = 0;
        for (i = ni ; i < np.size() ; i++)
        {
            if (i + 1 < np.size())
                np[i + 1] += (np[i] % p) * 10;
            else
                nmod = np[i] % p;
            np[i] /= p;
        }
        for (i = mi ; i < mp.size() ; i++)
        {
            if (i + 1 < mp.size())
                mp[i + 1] += (mp[i] % p) * 10;
            else
                mmod = mp[i] % p;
            mp[i] /= p;
        }
        while (ni < np.size() && np[ni] == 0) ni++;
        while (mi < mp.size() && mp[mi] == 0) mi++;
        ret = (ret * binomial(nmod, mmod)) % p;
    }
    return ret;
}
//Catalan Number:
long long catalan_number(int n)
{
    return binomial(n * 2, n) / (n + 1);
}
//Euler’s Totient Function: phi(n), n
// phi(n) = (p_1 - 1) * p_1 ^ (k_1 - 1) * (p_2 - 1) * p_2 ^ (k_2-1)

long long euler_totient2(long long n, long long ps)
{
    for (long long i = ps ; i * i <= n ; i++)
    {
        if (n % i == 0)
        {
            long long p = 1;
            while (n % i == 0)
            {
                n /= i;
                p *= i;
            }
            return (p - p / i) * euler_totient2(n, i + 1);
        }
        if (i > 2) i++;
    }
    return n - 1;
}
long long euler_totient(long long n)
{
    return euler_totient2(n, 2);
}
