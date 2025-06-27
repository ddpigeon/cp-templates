template<typename T>
struct SegTree {
    using Func = function<T(T, T)>;

    int n;
    vector<T> s;
    T unit;
    Func f;

    // Constructor: n is size, unit is identity value, f is associative function
    // eg. SegTree<ll> st(n, 0, [](ll a, ll b){return a+b;});

    SegTree(int n, T unit, Func f) : n(n), s(n << 1, unit), unit(unit), f(f) {}

    void update(int pos, T val) {
        for (s[pos += n] = val; pos >>= 1;)
            s[pos] = f(s[pos << 1], s[pos << 1 | 1]);
    }

    T query(int b, int e) { // [b, e)
        T ra = unit, rb = unit;
        for (b += n, e += n; b < e; b >>= 1, e >>= 1) {
            if (b & 1) ra = f(ra, s[b++]);
            if (e & 1) rb = f(s[--e], rb);
        }
        return f(ra, rb);
    }
};
