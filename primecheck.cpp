bool PrimeCheck(ll n) {
    if (n < 2 || n % 6 % 4 != 1) return (n | 1) == 3;
    ll s = __builtin_ctzll(n-1);
    ll d = n >> s;

    vector<ll> bases = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};

    for (ll a: bases) {
        a %= n;
        if (a == 0) continue;
        ll x = power(a, d, n);
        ll y;
        for (int i = 0; i < s; i++) {
            y = modmul(x, x, n);
            if (y == 1 && x != 1 && x != n-1) return false;
            x = y;
        }
        if (y != 1) return false;
    }
    return true;
}
