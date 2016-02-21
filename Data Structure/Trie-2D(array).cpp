/*
    Trie implementation using array, faster and takes less memory.
    Each node can contain arbitrary data as needed for solving the problem.
    The ALPHABET, MAX and scale() may need tuning as necessary.
*/

const int ALPHABET = 26;
const int MAX = 100000;    // Number of string * number of Alphabet
#define scale(x) (x-'a')   // for mapping items form 0 to ALPHABET-1
int next[MAX][ALPHABET];
int data[MAX];             // there can be more data fields

struct Trie
{
    int n, root;
    void init()
    {
        root = 0, n = 1;
        data[root] = 0;
        memset(next[root], -1, sizeof(next[root]));
    }
    void insert(char *s)
    {
        int curr = root, i, k;
        for(i = 0; s[i]; i++)
        {
            k = scale(s[i]);
            if(next[curr][k] == -1)
            {
                next[curr][k] = n;
                memset(next[n], -1, sizeof(next[n]));
                n++;
            }
            curr = next[curr][k];
            data[curr] = s[i];   // optional
        }
        data[curr] = 0;   // sentinel, optional
    }
    bool find(char *s)
    {
        int curr = root, i, k;
        for(i = 0; s[i]; i++)
        {
            k = scale(s[i]);
            if(next[curr][k] == -1) return false;
            curr = next[curr][k];
        }
        return (data[curr] == 0);
    }
};

int main()
{
    Trie T;T.init();
    return 0;
}


