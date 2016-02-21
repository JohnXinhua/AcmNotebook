/*
    Basic trie, don't use it with large alphabet set or long length.
    All operation has complexity O(length).
    MAX is number of different items.
*/
#define MAX 26 /// number of elements to string build.
struct trie
{
    trie *next[MAX+1];
    trie() {for(int i=0; i<=MAX; i++) next[i] = NULL;}
};

void insert(trie *root, int *seq, int len)
{
    trie *curr = root;
    for(int i = 0; i < len; i++)
    {
        if(!curr->next[seq[i]]) curr->next[seq[i]] = new trie;
        curr = curr->next[seq[i]];
    }
    if(!curr->next[MAX]) curr->next[MAX] = new trie;
}

bool found(trie *root, int *seq, int len)
{
    trie *curr = root;
    for(int i = 0; i < len; i++)
    {
        if(!curr->next[seq[i]]) return false;
        curr = curr->next[seq[i]];
    }
    if(!curr->next[MAX]) return false;
    return true;
}

int main()
{
    trie *root=new trie;
    return 0;
}

