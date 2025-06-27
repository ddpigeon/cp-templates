class Matrix {
public:
    vector<vector<ll>> a;
    ll n;

    Matrix(ll sz) {
        n = sz;
        a.resize(sz);
        for (int i = 0; i < sz; i++) a[i].resize(sz);
    }

    Matrix operator *(Matrix other) {
        Matrix product = Matrix(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    product.a[i][k] += a[i][j] * other.a[j][k];
                    product.a[i][k] %= mod;
                }
            }
        }
        return product;
    }
};

Matrix expo_power(Matrix a, ll n, long long p) {
    // n = size. p = power
    Matrix res = Matrix(n);
    for (int i = 0; i < n; i++) res.a[i][i] = 1;
    while(p) {
        if(p % 2) {
            res = res * a;
        }
        p /= 2;
        a = a * a;
    }
    return res;
}

