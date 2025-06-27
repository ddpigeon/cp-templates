template<typename T>
struct point2d {
    T x, y;
    
    point2d() = default;
    point2d(T x, T y): x(x), y(y) {}

    point2d& operator+=(const point2d &t) {
        x += t.x;
        y += t.y;
        return *this;
    }
    point2d& operator-=(const point2d &t) {
        x -= t.x;
        y -= t.y;
        return *this;
    }
    point2d& operator*=(T t) {
        x *= t;
        y *= t;
        return *this;
    }
    point2d& operator/=(T t) {
        x /= t;
        y /= t;
        return *this;
    }

    point2d operator+(const point2d &t) const {
        return point2d(*this) += t;
    }
    point2d operator-(const point2d &t) const {
        return point2d(*this) -= t;
    }
    point2d operator*(T t) const {
        return point2d(*this) *= t;
    }
    point2d operator/(T t) const {
        return point2d(*this) /= t;
    }
    long double arg() const {
        return atan2(static_cast<long double>(y), static_cast<long double>(x));
    }
};

// Non-member scalar multiplication
template<typename T>
point2d<T> operator*(T a, point2d<T> b) {
    return b * a;
}

using point = point2d<ld>;
