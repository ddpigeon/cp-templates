ll x = -1;
for (ll b = z; b >= 1; b /= 2) {
    while (!ok(x+b)) x += b;
}
ll k = x+1;
