struct ufds{
    vector <int> root, sz;
    int n;
 
    void init(int nn){
        n = nn;
        root.resize(n + 1);
        sz.resize(n + 1, 1);
        for (int i = 1; i <= n; i++) root[i] = i;
    }
 
    int find(int x){
        if (root[x] == x) return x;
        return root[x] = find(root[x]);
    }
 
    bool unite(int x, int y){
        x = find(x); y = find(y);
        if (x == y) return false;
 
        if (sz[y] > sz[x]) swap(x, y);
        sz[x] += sz[y];
        root[y] = x;
        return true;
    }
};
