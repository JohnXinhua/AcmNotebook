#include <stdio.h>
#include <math.h>

#define SIZE  18000
#define LIMIT 200000000 // maximum input
#define BLOCK sizeof(int)

long prime_num, primes[SIZE];
int temp[LIMIT / BLOCK / 2 + 1];

int is_prime(long n)
{
	n = (n - 1) / 2;
	if (n % BLOCK == 0)
		return (!(temp[n / BLOCK - 1] & 1));
	else
		return (!(temp[n / BLOCK] & (1 << (BLOCK - n % BLOCK))));
}

void sieve_3(void)
{
	long i, j, k, loc, loop;

	prime_num = 0;
	for (i=0, k = LIMIT / BLOCK / 2 + 1; i < k; i++)
		temp[i] = 0;
	for (i=3, loop=(long)(sqrt(LIMIT)) + 1; i < loop; i+=2)
	{
		if (is_prime(i))
		{
			for (j = i, k = LIMIT / i + 1; j < k; j+=2)
			{
				loc = (i * j - 1) / 2;
				if (loc%BLOCK == 0)
					temp[loc / BLOCK -1] |= 1;
				else
					temp[loc / BLOCK] |= (1 << (BLOCK - loc % BLOCK));
			}
		}
	}
	for (i = 3, primes[prime_num++] = 2; i <= LIMIT; i += 2)
		if (is_prime(i))
			primes[prime_num++] = i;

	printf ("%ld\n",prime_num);
}

int main(void)
{
	sieve_3();

	return 0;
}
