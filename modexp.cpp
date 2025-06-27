ll modmul(ll a, ll b, ll mod) {
    return (__int128)a * b % mod;
}

ll power(ll x, ll y, ll p) { 
    ll res = 1;     
    x %= p; 
    if (x == 0) return 0;
 
    while (y > 0) { 
        if (y & 1) res = modmul(res, x, p);  
        y = y>>1;
        x = modmul(x, x, p); 
    } 
    return res; 
} 
