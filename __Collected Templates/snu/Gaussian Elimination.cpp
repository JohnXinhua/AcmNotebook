#include <algorithm>
using namespace std;

long long gcd(long long a, long long b)
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

struct rational
{
    long long p, q;
    void red()
    {
        if (q < 0)
        {
            p *= -1;
            q *= -1;
        }
        long long t = gcd((p >= 0 ? p : -p), q);
        p /= t;
        q /= t;
    }
    rational() {}
    rational(long long p_): p(p_), q(1) {}
    rational(long long p_, long long q_): p(p_), q(q_)
    {
        red();
    }
    bool operator==(const rational& rhs) const
    {
        return p == rhs.p && q == rhs.q;
    }
    bool operator!=(const rational& rhs) const
    {
        return p != rhs.p || q != rhs.q;
    }
    bool operator<(const rational& rhs) const
    {
        return p * rhs.q < rhs.p * q;
    }
    const rational operator+(const rational& rhs) const
    {
        return rational(p * rhs.q + q * rhs.p, q * rhs.q);
    }
    const rational operator-(const rational& rhs) const
    {
        return rational(p * rhs.q - q * rhs.p, q * rhs.q);
    }
    const rational operator*(const rational& rhs) const
    {
        return rational(p * rhs.p, q * rhs.q);
    }
    const rational operator/(const rational& rhs) const
    {
        return rational(p * rhs.q, q * rhs.p);
    }
};

namespace gaussian
{
typedef rational val_t;
const val_t abs(const val_t& x)
{
    return (x.p >= 0) ? x : rational(-x.p, x.q);
}

#define GET(i, j, n) A[i * n + j]
// return true when solution exists, false o/w.
bool run(int size_eq, int size_var, val_t* A, val_t* B, val_t* C)
{
    int i = 0, j = 0, k, l;
    int maxi;
    val_t temp_r;
    val_t* x;
    val_t* y;
    while (i < size_eq && j < size_var)
    {
        maxi = i;
        for (k = i + 1 ; k < size_eq ; k++)
            if (abs(GET(maxi, j, size_var)) < abs(GET(k, j, size_var)))
                maxi = k;
        if (GET(maxi, j, size_var) != val_t(0))
        {
            x = A + i * size_var;
            y = A + maxi * size_var;
            for (k = 0 ; k < size_var ; k++)
                swap(*(x + k), *(y + k));
            swap(B[i], B[maxi]);
            temp_r = *(x + j);
            for (k = j ; k < size_var ; k++)
                *(x + k) = *(x + k) / temp_r;
            B[i] = B[i] / temp_r;
            for (k = 0 ; k < size_eq ; k++)
            {
                if (k == i) continue;
                temp_r = GET(k, j, size_var);
                for (l = j ; l < size_var ; l++)
                    GET(k, l, size_var) = GET(k, l, size_var)
                                          - temp_r * GET(i, l, size_var);
                B[k] = B[k] - GET(k, j, size_var) * B[i];
            }
            i++;
        }
        j++;
    }
    if (i < size_eq)
        for ( ; i < size_eq ; i++)
            if (B[i] != val_t(0)) return false;
// C[...] := Case by case
    return true;
}
#undef GET
} // namespace gaussian
