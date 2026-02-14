#ifndef CUDA_MATH_H
#define CUDA_MATH_H
/*
    This header file aims to implement math structs and operations for CUDA API
    as seamlessly as possible with glsl
*/
#include <cmath>
#include <cuda_runtime.h>

struct alignas(16) vec2 {
    float x;
    float y;

    vec2() : x(0.0f), y(0.0f) {};
    vec2(float v) : x(v), y(v) {};
    vec2(float x, float y) : x(x), y(y) {};

    __host__ __device__ inline
    vec2 operator+(const vec2& other) const {
        return vec2(this->x + other.x, this->y + other.y);
    }

    __host__ __device__ inline
    vec2 operator-(const vec2& other) const {
        return vec2(this->x - other.x, this->y - other.y);
    }

    __host__ __device__ inline
    vec2 operator-() const {
        return vec2(-this->x, -this->y);
    }

    __host__ __device__ inline
    vec2 operator*(float s) const {
        return vec2(this->x * s, this->y * s);
    }

    __host__ __device__ inline
    vec2 operator*(const vec2& other) const {
        return vec2(this->x * other.x, this->y * other.y);
    }

    __host__ __device__ inline
    vec2 operator/(float s) const {
        return vec2(this->x, this->y) * (1.0f/s);
    }

    static constexpr int length = 2;

    float& operator[](int i) { return *(&x + i); }
    const float& operator[](int i) const { return *(&x + i); }
};

struct alignas(16) vec3 {
    float x;
    float y;
    float z;

    vec3() : x(0.0f), y(0.0f), z(0.0f) {};
    vec3(float v) : x(v), y(v), z(v) {};
    vec3(float x, float y, float z) : x(x), y(y), z(z) {};

    __host__ __device__ inline
    vec3 operator+(const vec3& other) const {
        return vec3(this->x + other.x, this->y + other.y, this->z + other.z);
    }

    __host__ __device__ inline
    vec3 operator-(const vec3& other) const {
        return vec3(this->x - other.x, this->y - other.y, this->z - other.z);
    }

    __host__ __device__ inline
    vec3 operator-() const {
        return vec3(-this->x, -this->y, -this->z);
    }

    __host__ __device__ inline
    vec3 operator*(float s) const {
        return vec3(this->x * s, this->y * s, this->z * s);
    }

    __host__ __device__ inline
    vec3 operator*(const vec3& other) const {
        return vec3(this->x * other.x, this->y * other.y, this->z * other.z);
    }

    __host__ __device__ inline
    vec3 operator/(float s) const {
        return vec3(this->x, this->y, this->z) * (1.0f/s);
    }

    static constexpr int length = 3;

    float& operator[](int i) { return *(&x + i); }
    const float& operator[](int i) const { return *(&x + i); }
};

struct vec4 {
    float x;
    float y;
    float z;
    float w;

    vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {};
    vec4(float v) : x(v), y(v), z(v), w(v) {};
    vec4(const vec3& v3, float w = 0.0f) : x(v3.x), y(v3.y), z(v3.z), w(w) {};
    vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {};

    __host__ __device__ inline
        vec4 operator+(const vec4& other) const {
        return vec4(this->x + other.x, this->y + other.y, this->z + other.z, this->w + other.w);
    }

    __host__ __device__ inline
        vec4 operator-(const vec4& other) const {
        return vec4(this->x - other.x, this->y - other.y, this->z - other.z, this->w - other.w);
    }

    __host__ __device__ inline
        vec4 operator-() const {
        return vec4(-this->x, -this->y, -this->z, -this->w);
    }

    __host__ __device__ inline
        vec4 operator*(float s) const {
        return vec4(this->x * s, this->y * s, this->z * s, this->w * s);
    }

    __host__ __device__ inline
        vec4 operator*(const vec4& other) const {
        return vec4(this->x * other.x, this->y * other.y, this->z * other.z, this->w * other.w);
    }

    __host__ __device__ inline
    vec4 operator/(float s) const {
        return vec4(this->x, this->y, this->z, this->w) * (1.0f/s);
    }

    static constexpr int length = 4;

    float& operator[](int i) { return *(&x + i); }
    const float& operator[](int i) const { return *(&x + i); }
};

__host__ __device__ inline
float dot(const vec3& v0, const vec3& v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

__host__ __device__ inline
float dot(const vec4& v0, const vec4& v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
}

__host__ __device__ inline
vec3 cross(const vec3& v0, const vec3& v1) {
    return vec3(v0.y * v1.z - v0.z * v1.y, v0.z * v1.x - v0.x * v1.z, v0.x * v1.y - v0.y * v1.x);
}

__host__ __device__ inline
float length(const vec3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline
float length(const vec4& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline
vec3 normalize(const vec3& v) {
    return v / length(v);
}

__host__ __device__ inline
vec4 normalize(const vec4& v) {
    return v / length(v);
}

#endif