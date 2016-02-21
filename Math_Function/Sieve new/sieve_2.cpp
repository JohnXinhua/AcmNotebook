#include <stdio.h>
#include <math.h>
#include <string.h>

#define LIMIT 1000005   // Maximum Limit of input
#define BLOCK 8*sizeof(char)

char prime[LIMIT / BLOCK / 2 + 2];

int is_prime(long n)
{
    if (n < 2)
        return 0;

    if (n == 2)
        return 1;

    if (n % 2 == 0)
        return 0;

    n = (n - 1) / 2;

    return (prime[n / BLOCK] & (1 << (BLOCK - n % BLOCK - 1)));
}

void sieve_2(void)
{
    long i, j, s;

    memset(prime, ~(0), LIMIT / BLOCK / 2 + 1);

    s = (long)(sqrt(LIMIT)) + 1;

    for (i = 3; i < s; i += 2)
        if (is_prime(i))
            for (j = i * i; j < LIMIT; j += i)
                if (j % 2)
                    prime[(j - 1) / 2 / BLOCK] &= (~(1 << (BLOCK - ((j - 1) / 2) % BLOCK - 1)));
}

int main(void)
{
    sieve_2();

    //if (is_prime(n))

    return 0;
}
