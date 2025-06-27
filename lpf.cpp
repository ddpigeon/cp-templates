vector<ll> lpf(ll n) {
    vector<ll> lepf(n+1);
    lepf[0] = 0;
    lepf[1] = 1;
    for (ll i = 2; i * i <= n; i++) {
        if (lepf[i] == 0) {
            lepf[i] = i;
            for (ll j = i; j < n; j+= i) {
                if (lepf[j] == 0) lepf[j] = i;
            }
        }
    }
    for (int i = 2; i <= n; i++) if (lepf[i] == 0) lepf[i] = i;
    return lepf;
}
