#include <stdio.h>
#include <math.h>

#define LIMIT  200000000// (Maximum number + 1) that we need to judge

char prime[LIMIT + 1];

void sieve_1(void)
{
	long i, j, s;

	prime[0] = prime[1] = 0;
	prime[2] = 1;

	for (i = 3; i < LIMIT; i += 2)
	{
		prime[i] = 1;
		prime[i + 1] = 0;
	}

	s = (long)(sqrt(LIMIT));
	for (i = 3; i <= s; i += 2)
		if (prime[i])
			for (j = i * i; j < LIMIT; j += i)
				prime[j] = 0;
}

int main(void)
{
	sieve_1();

	//if (prime[n]);

	return 0;
}
