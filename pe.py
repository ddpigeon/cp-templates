import math
import functools
import sys

# returns whether or not n is prime
# Deterministic Miller test
def primecheck(n):

    a_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    # Should be big enough for most n I guess
    if n in a_list:
        return True

    if n == 1:
        return False

    end_idx = 0

    if n < 2047:
        end_idx = 1

    elif n < 1373653:
        end_idx = 2

    elif n < 25326001:
        end_idx = 3

    elif n < 3215031751:
        end_idx = 4

    elif n < 2152302898747:
        end_idx = 5

    elif n < 3474749660383:
        end_idx = 6

    elif n < 341550071728321:
        end_idx = 7

    elif n < 3825123056546413051:
        end_idx = 9

    elif n < 318665857834031151167461:
        end_idx = 12
    elif n < 3317044064679887385961981:
        end_idx = 13

    if end_idx > 0:
        a_list = a_list[:end_idx]


    if (n%2 == 0):
        return False

    s = 0
    d = n-1

    while (d%2 == 0):
        s += 1
        d //= 2

    #print(s, d)

    for a in a_list:
        x = fast_mod_exp(a, d, n)
        for i in range(s):
            y = fast_mod_exp(x, 2, n)
            if y == 1 and x != 1 and x != n-1:
                return False

            x = y
        if y != 1:
            return False

    return True


# Calculates a^b mod m
def fast_mod_exp(a, b, m):
    a %= m;

    if (a == 0):
        return 0

    elif (b == 0):
        return 1

    elif (b&1):
        return (a * fast_mod_exp((a*a)%m, (b-1)//2, m))%m

    else:
        return fast_mod_exp((a*a)%m, b//2, m)%m

# Modulo inverse of a wrt a prime
def modInverse(A, M):
    m0 = M
    y = 0
    x = 1
 
    if (M == 1):
        return 0
 
    while (A > 1): 
        q = A // M
        t = M
        M = A % M
        A = t
        t = y
        y = x - q * y
        x = t
 
    if (x < 0):
        x = x + m0
 
    return x

# Chinese remainder theorem
def crt(bases, rems):
    assert len(bases) == len(rems)
    n = len(bases)

    prd = 1
    for i in bases:
        prd *= i

    res = 0

    for i in range(n):
        pp = prd // bases[i]
        res += (rems[i] * pp * modInverse(pp, bases[i]))
        res %= prd

    return res


# returns a list of primes upto n
def primes(n):
    sieve = [True for i in range(n + 1)]
    sieve[0] = sieve[1] = False
    for i in range(2, int((n ** 0.5) + 1)):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


# returns a dict with {x: sum of proper divisors of x} upto n; they are equal for perfect numbers
def divisor_sum(n):
    sums = {}
    for i in range(1, n + 1):
        divisorsum = 0
        for j in range(1, int(math.sqrt(i)) + 1):
            if (x := i / j) == int(x):
                divisorsum += (x+j)
            if x == j:
                divisorsum -= x
        divisorsum -= i
        print(i)
        sums.update({i: int(divisorsum)})
    return sums


# returns the first n triangular numbers
def triangular_numbers(n):
    a = 0
    trangles = []
    for i in range(1, n+1):
        a += i
        trangles.append(a)
    return trangles


# returns the next lexicographic permutation of a sequence
# this was before I found next_permutation()
def lexicographer(seq):
    seq3 = list(seq)
    seq3.sort(reverse=True)
    if seq3 == seq:
        return seq
    for m in reversed(range(len(seq))):
        try:
            if seq[m] < seq[m + 1]:
                for n in reversed(range(m + 1, len(seq))):
                    if seq[n] > seq[m]:
                        seq[n], seq[m] = seq[m], seq[n]
                        seq[m + 1:] = seq[m + 1:][::-1]
                        return seq

        except IndexError:
            pass


# returns all permutations of a sequence
# just itertools this one lmao
def permuter(seq):
    total_perms = []
    seq2 = seq[::-1]
    while seq2 != seq:
        seq4 = list(lexicographer(seq))
        total_perms.append(seq4)
    return total_perms


# returns prime factors of a number greater than 2
def prime_factoriser(n):
    factors = []
    a = n
    while a != 1:
        factors.append(b := least_pf(int(a)))
        a /= b
    return factors


# returns lowest prime factor of n
def least_pf(n):
    for i in range(2, n + 1):
        if n % i == 0:
            return i

# Returns array of Least Prime factors 2 to n
def lpf(n):

    sievelim = int(n**0.5)+1

    spf = [i for i in range(n+1)]
    spf[1] = 1

    for i in range(2, n+1, 2):
        spf[i] = 2
        
    for i in range(3, sievelim):
        if spf[i] == i:
            for j in range(i * i, n + 1, 2*i):
                if spf[j] == j:
                    spf[j] = i

    return spf

# returns prime factorization of all numbers in a range upto n
def prime_factorise_range(n):
    factors = lpf(n)
    pfrange = [dict() for i in range(n+1)]
    pfrange[0][0] = 1
    pfrange[1][1] = 1
    for i in range(2, n+1):
        cur = i
        while cur > 1:
            if factors[cur] not in pfrange[i]:
                pfrange[i][factors[cur]] = 1
            else:
                pfrange[i][factors[cur]] += 1
            #pflist.append(factors[cur])
            cur //= factors[cur]

    return pfrange

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphadict = {letter: idx for idx, letter in enumerate(alphabet)}
digits = '0123456789'

# Returns all totients till n, starting from 1
def totient(n):
    spf = lpf(n)

    totients = [0] * (n+1)
    totients[1] = 1

    for i in range(2, n+1):
        if spf[i] == i:
            totients[i] = i-1

        else:
            p = spf[i]
            m = i // p
            if spf[m] == p:
                factor = p
            else:
                factor = p-1

            totients[i] = factor * totients[m]

    return totients


# Returns a dict of list of all factors of a number
def all_factors(n):
    factors = {}
    factors[1] = [1]
    for i in range(2, n+1):
        m = i%6
        if m == 0:
            factors[i] = [1, 2, 3]
        elif m == 2 or m == 4:
            factors[i] = [1, 2]
        elif m == 3:
            factors[i] = [1, 3]
        else:
            factors[i] = [1]

    for i in range(4, n+1):
        for j in range(i, n+1, i):
            factors[j].append(i)

    return factors

# FIArray, adapted from following blog to implement lucy's prime counting algorithm easier
# https://gbroxey.github.io/blog/2023/04/09/lucy-fenwick.html
class FIArray:
    def __init__(self, x):
        self.x = x
        self.isqrt = math.isqrt(x)
        self.size = 2*self.isqrt
        if self.isqrt == x // self.isqrt:
            self.size -= 1
        self.arr = [0] * self.size

    def __getitem__(self, v):
        if v <= 0:
            return 0

        elif v <= self.isqrt:
            return self.arr[v-1]

        return self.arr[self.size - (self.x // v)]


    def __setitem__(self, v, z):
        if v <= self.isqrt:
            self.arr[v-1] = z
        else:
            self.arr[self.size - (self.x // v)] = z


    def get(self, v):
        # same as setitem, will remove soon
        if v <= 0:
            return 0

        elif v <= self.isqrt:
            return self.arr[v-1]

        return self.arr[self.size - (self.x // v)]


    def set(self, v, z):
        if v <= self.isqrt:
            #print(self.x, v)
            self.arr[v-1] = z

        else:
            self.arr[self.size - (self.x // v)] = z

    
    def keysInc(self):
        for v in range(1, self.isqrt+1):
            yield v

        if self.isqrt != (self.x // self.isqrt):
            yield self.x // self.isqrt

        for n in range(self.isqrt-1, 0, -1):
            yield self.x // n


    def keysDec(self):
        for n in range(1, self.isqrt):
            yield self.x // n

        if self.isqrt != (self.x // self.isqrt):
            yield self.x // self.isqrt

        for v in range(self.isqrt, 0, -1):
            yield v


def primecount(n):
    fia = FIArray(n)
    for v in fia.keysInc():
        fia.set(v, v-1)
    
    for p in range(2, fia.isqrt+1):
        if fia.get(p) == fia.get(p-1):
            continue

        for v in fia.keysDec():
            if v < p*p:
                break
            fia.set(v, fia.get(v) - (fia.get(v//p) - fia.get(p-1)))

    return fia.get(n)


# Nth fibonacci modulo m
def fib_n(n, m):
    a = q = 1
    b = p = 0

    while n > 0:
        if n%2 == 0:
            qq = (q*q)%m
            q = (2*p*q + qq)%m
            p = (p*p + qq)%m
            n //= 2

        else:
            aq = (a*q)%m
            a = (b*q + aq + a*p)%m
            b = (b*p + aq)%m
            n -= 1

    return b

# Matrix multiplication
def mat_mul(a, b, modulus=-1):
    c = []
    for i in range(len(a)):
        c.append([0]*len(b[0]))
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                c[i][j] += (a[i][k]*b[k][j])
                if modulus != -1:
                    c[i][j] %= modulus;
    return c

# matrix exponentiation
def mat_pow(a, n, modulus=-1):
    if n<0 or len(a) != len(a[0]):
        return None


    if n == 0:
        sz = len(a)
        mat = [[0] * sz for i in range(sz)] 
        for i in range(sz):
            mat[i][i] = 1
        return mat

    if n==1:
        return a
    if n==2:
        return mat_mul(a, a, modulus)
    t1 = mat_pow(a, n//2, modulus)
    if n%2 == 0:
        return mat_mul(t1, t1, modulus)
    return mat_mul(t1, mat_mul(a, t1, modulus), modulus)


# Extended Euclidean algorithm, returns GCD, bezout coefficients
def gcd_extended(a, b):
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = gcd_extended(b%a, a)

    x = y1 - (b//a) * x1
    y = x1

    return gcd, x, y

# Generates list of binary numbers of n bits or less:
def binstrings(n):
    strings = []
    for i in range(2**n, 2**(n+1)):
        strings.append([int(c) for c in bin(i)[3:]])

    return strings


# union find /DSU
class ufds:
    def __init__(self, nn):
        self.n = nn
        self.root = [i for i in range(self.n+1)]
        self.sz = [1 for i in range(self.n+1)]


    def find(self, x):
        if self.root[x] == x:
            return x
        else:
            self.root[x] = self.find(self.root[x])
            return self.root[x]

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if (x == y):
            return False

        if self.sz[x] > self.sz[y]:
            x, y = y, x

        self.sz[x] += self.sz[y]
        self.root[y] = x
        return True


# mobius function till x in nlogn
# 0 if not squarefree, otherwise (-1)^(no. of primes dividing x)
def mobius(x):
    primelist = primes(x)
    mobii = [1] * (x+1)

    for i in primelist:
        ct = 0
        for j in range(i, x+1, i):
            ct += 1
            mobii[j] *= -1
            if ct == i:
                mobii[j] = 0
                ct = 0

    #print(mobii)
    return mobii


# Mertens function of x, i.e sum of mobius(x) till x.
def mertens(x):
    M = FIArray(x)
    mu = mobius(math.isqrt(x) + 1)

    for v in M.keysInc():
        if v == 1:
            M[v] = 1
            continue

        muV = 1
        vsqrt = math.isqrt(v)
        for i in range(1, vsqrt+1):
            muV -= mu[i] * (v//i)
            muV -= M[v//i]

        muV += M[vsqrt]*vsqrt
        M[v] = muV

    #print(M[x])
    return M



# sum from a to b, both inclusive with given step
def sum_ap(a, b, step, mod=None):
    n = (b-a)//step + 1
    if mod is None:
        return n * (a+b) // 2

    else:
        return (n * (a+b) // 2) % mod

def sumN(x, mod=None):
    return sum_ap(1, x, 1, mod)

def TotientSum(x, modulo=None):
    M = mertens(x)
    xsqrt = math.isqrt(x)
    res = 0

    for n in range(1, xsqrt+1):
        res += (M[n] - M[n-1]) * sumN(x//n, modulo)
        res += n * M[x//n]
        if modulo is not None:
            res %= modulo

    res -= sumN(xsqrt, modulo) * M[xsqrt]
    if modulo is not None:
        res = (res+modulo)%modulo

    return res


def EvenTotientSum(x, m=None):

    res = 0
    if x <= 10**5:
        phi = totient(x)
        for i in range(2, x+1, 2):
            res += phi[i]
        return res


    res = TotientSum(x//2, m)
    res += EvenTotientSum(x//2, m)

    if m is not None:
        res = (res+m)%m

    return res

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

# mCn % p in O(p)
def LucasComb(m, n, p):

    factorials = [1]*p
    inv_factorials = [1]*p

    for i in range(2, p):
        factorials[i] = (factorials[i-1] * i)%p
        inv_factorials[i] = modInverse(factorials[i], p)

    def small_lucas(m1, n1, p):
        if m1 < n1:
            return 0
        elif n1 == 0 or n1 == m1:
            return 1

        else:
            return (factorials[m1] * inv_factorials[n1] * inv_factorials[m1-n1])%p 

    m_p = numberToBase(m, p)
    n_p = numberToBase(n, p)
    mlen = len(m_p)
    nlen = len(n_p)

    sz = max(mlen, nlen)

    ans = 1
    for i in range(sz):
        midx = mlen-i-1
        nidx = nlen-i-1

        nd = n_p[nidx] if nidx >= 0 else 0
        md = m_p[midx] if midx >= 0 else 0

        ans *= small_lucas(md, nd, p)
        ans %= p
        if ans == 0:
            break

    return ans


# General polynomial class
class poly:
    def __init__(self, l):
        self.coeffs = list(l)
        self.degree = len(l)-1

    def __eq__(self, other):
        if not isinstance(other, poly):
            return False
        return self.coeffs == other.coeffs

    def __getitem__(self, i):
        if i < 0 or i > self.degree:
            return 0

        else:
            return self.coeffs[i]

    def __add__(self, other):
        newl = []
        for i in range(max(self.degree, other.degree)+1):
            newl.append(self[i] + other[i])

        while (newl[-1] == 0):
            del newl[-1]

        if len(newl) == 0:
            newl.append(0)

        return poly(newl)

    def __iadd__(self, other):
        for i in range(0, self.degree+1):
            self.coeffs[i] += other.coeffs[i]

        if other.degree > self.degree:
            for i in range(self.degree+1, other.degree+1):
                self.coeffs.append(other.coeffs[i])

        while (self.coeffs[-1] == 0):
            del self.coeffs[-1]
            self.degree -= 1

        if len(self.coeffs) == 0:
            self.coeffs.append(0)
            self.degree = 0

        return self

    def lc(self):
        return self.coeffs[self.degree]


    # change this to FFT at some point I guess
    def __mul__(self, other):
        if type(other) is not poly:
            prod = poly(self.coeffs)
            for i in range(0, self.degree+1):
                prod.coeffs[i] = self.coeffs[i] * other

        else:
            prod = poly([0] * (self.degree + other.degree + 1))
            for i in range(self.degree+1):
                for j in range(other.degree+1):
                    prod.coeffs[i+j] += (self.coeffs[i] * other.coeffs[j])

        return prod


    def __rmul__(self, other):
        return self * other


    def euclidean_div(self, other):
        if other == poly([0]):
            raise ZeroDivisionError()

        r = list(self.coeffs)
        q = [0] * max(len(r) - len(other.coeffs) + 1, 1)
        d = other.degree
        c = other.lc()

        while len(r) - 1 >= d:
            k = len(r) - 1
            coeff = r[k] // c
            q[k-d] = coeff
            
            for i in range(d+1):
                r[k-d+i] -= coeff * other.coeffs[i]

            while len(r) > 1 and r[-1] == 0:
                del r[-1]

        return poly(q), poly(r)


    def __mod__(self, other):
        return self.euclidean_div(other)[1]

    def __floordiv__(self, other):
        return self.euclidean_div(other)[0]

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def mod_coeffs(self, mod):
        for i in range(len(self.coeffs)):
            self.coeffs[i] %= mod

        return self

# polynomial A^n in O(ord(A)^2 log(n))
def fast_poly_mod_exp(a, b, mod_pol, mod_coeff):
    a %= mod_pol;
    a.mod_coeffs(mod_coeff)

    if (a == poly([0])):
        return poly([0])

    elif (b == 0):
        return poly([1])

    elif b == 1:
        return a

    elif (b&1):
        return ((a * fast_poly_mod_exp((a*a)%mod_pol, (b-1)//2, mod_pol, mod_coeff))%mod_pol).mod_coeffs(mod_coeff)

    else:
        return (fast_poly_mod_exp((a*a)%mod_pol, b//2, mod_pol, mod_coeff)%mod_pol).mod_coeffs(mod_coeff)


# Nth term of Linear recurrence of order k in k^2 log(N)
def solve_linear_recurrence(init, n, charpoly, mod):
    if len(init)+1 < len(charpoly):
        print("too few initial terms")
        return 0

    elif n < len(init):
        return init[n]

    else:
        po = poly([0, 1])
        mod_poly = poly(charpoly)
        ans_1 = fast_poly_mod_exp(po, n, mod_poly, mod)
        ans = 0
        for i in range(len(ans_1.coeffs)):
            ans += ans_1.coeffs[i] * init[i]
            ans %= mod
        return ans


# returns array of derangements of given number of elements till n
def derangements(n):
    d = [1, 0]
    for i in range(2, n+1):
        d.append((i-1) * (d[i-1] + d[i-2]))

    return d


