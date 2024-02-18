/*

VecUtils: A Vector Utility Library
--------------------------------------

Author: Isak Horvath
Year:   2024

INSPIRATION:
This project draws inspiration from the work of Ingemar Ragnemalm,
author of MicroGLUT and examiner for university computer graphics course TSKB07. 

*/

#ifndef VECUTILS
#define VECUTILS

#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

typedef struct Vector4 Vector4;

typedef struct Vector3 {
    union {float x; float r;};
    union {float y; float g;};
    union {float z; float b;};

    Vector3() {}
    Vector3(float x2, float y2, float z2) : x(x2), y(y2), z(z2) {}
    Vector3(float x2) : x(x2), y(x2), z(x2) {}
	Vector3(Vector4 v);
} Vector3, *Vector3Ptr;

typedef struct Vector4 {
    union {float x; float r;};
    union {float y; float g;};
    union {float z; float b;};
    union {float h; float w; float a;};

    Vector4() {}
    Vector4(float x2, float y2, float z2, float w2) : x(x2), y(y2), z(z2), w(w2) {}
	Vector4(float xyz, float w2) : x(xyz), y(xyz), z(xyz), w(w2) {}
	Vector4(float v, float w2) : x(v.x), y(v.y), z(v.z), w(w2) {}
	Vector4(float v) : x(v.x), y(v.y), z(v.z), w(1) {}
} Vector4, *Vector4Ptr;

typedef struct Vector2 {
	union {float x; float s; float u;};
	union {float y; float t; float v;};
		
    Vector2() {}
	Vector2(float x2, float y2) : x(x2), y(y2) {}
} Vector2, *Vector2Ptr;

typedef struct Matrix3 Matrix3;

typedef struct Matrix4 {
	float m[16];

    Matrix4() {}
    Matrix4(float x2) {
	    m[0]  = x2; m[1]  =  0; m[2]  =  0; m[3]  =  0;
		m[4]  =  0; m[5]  = x2; m[6]  =  0; m[7]  =  0;
		m[8]  =  0; m[9]  =  0; m[10] = x2; m[11] =  0;
		m[12] =  0; m[13] =  0; m[14] =  0; m[15] = x2;
	}
    Matrix4(float p0,  float p1,  float p2,  float p3,
			float p4,  float p5,  float p6,  float p7,
			float p8,  float p9,  float p10, float p11, 
			float p12, float p13, float p14, float p15) {
		m[0]  =  p0; m[1]  =  p1; m[2]  =  p2; m[3]  =  p3;
		m[4]  =  p4; m[5]  =  p5; m[6]  =  p6; m[7]  =  p7;
		m[8]  =  p8; m[9]  =  p9; m[10] = p10; m[11] = p11;
		m[12] = p12; m[13] = p13; m[14] = p14; m[15] = p15;
	}
	Matrix4(Matrix3 x);
} Matrix4;

typedef struct Matrix3 {
	float m[9];
    
    Matrix3() {}
	Matrix3(float x2) {
	    m[0] = x2; m[1] =  0; m[2] =  0;
		m[3] =  0; m[4] = x2; m[5] =  0;
		m[6] =  0; m[7] =  0; m[8] = x2;
	}
	Matrix3(float p0, float p1, float p2,
			float p3, float p4, float p5,
			float p6, float p7, float p8) {
		m[0] = p0; m[1] = p1; m[2] = p2;
		m[3] = p3; m[4] = p4; m[5] = p5;
		m[6] = p6; m[7] = p7; m[8] = p8;
	}
	Matrix3(Matrix4 x) {
		m[0] = x.m[0]; m[1] = x.m[1]; m[2] =  x.m[2];
		m[3] = x.m[4]; m[4] = x.m[5]; m[5] =  x.m[6];
		m[6] = x.m[8]; m[7] = x.m[9]; m[8] = x.m[10];
	}
	Matrix3(Vector3 x1, Vector3 x2, Vector3 x3) {
		m[0] = x1.x; m[1] = x1.y; m[2] = x1.z;
		m[3] = x2.x; m[4] = x2.y; m[5] = x2.z;
		m[6] = x3.x; m[7] = x3.y; m[8] = x3.z;
	}
} Matrix3;

Vector2 Set(float x, float y);
Vector3 Set(float x, float y, float z);
Vector4 Set(float x, float y, float z, float w);

Matrix3 Set(float p0,  float p1,  float p2,  float p3, float p4, float p5, float p6, float p7, float p8);
Matrix4 Set(float p0,  float p1,  float p2,  float p3,
			float p4,  float p5,  float p6,  float p7,
			float p8,  float p9,  float p10, float p11, 
			float p12, float p13, float p14, float p15);

Vector3 Sub(Vector3 a, Vector3 b);
Vector3 Add(Vector3 a, Vector3 b);

float Dot(Vector3 a, Vector3 b);
float Norm(Vector3 a);

Vector3 Cross(Vector3 a, Vector3 b);
Vector3 Scalar(Vector3 a, float s);
Vector3 Normalize(Vector3 a);

Vector3 GetNormal(Vector3 a, Vector3 b, Vector3 c);

void Split(Vector3 v, Vector3 n, Vector3 *vn, Vector3 *vp);

Matrix4 GetIdentityMatrix();

Matrix4 RotateX(float a);
Matrix4 RotateY(float a);
Matrix4 RotateZ(float a);

Matrix4 Transform(float tx, float ty, float tz);
Matrix4 Scale(float sx, float sy, float sz);

Matrix4 Mult(Matrix4 a, Matrix4 b);
Matrix3 Mult(Matrix3 a, Matrix3 b);

Vector3 Mult(Matrix3 a, Vector3 b);
Vector3 Mult(Matrix4 a, Vector3 b);
Vector4 Mult(Matrix4 a, Vector4 b);

void OrthoNormalize(Matrix4 *R);

Matrix4 Transpose(Matrix4 m);
Matrix3 Transpose(Matrix3 m);

Matrix4 ArbritraryRotate(Vector3 axis, float fi);

Matrix4 CrossMatrix(Vector3 a);

Matrix4 Add(Matrix4 a, Matrix4 b);

void SetTransposed(char t);

Matrix4 LookAtVector(Vector3 p, Vector3 l, Vector3 v);
Matrix4 LookAt( float px, float py, float pz, 
			    float lx, float ly, float lz,
			    float vx, float vy, float vz);

Matrix4 Perspective(float fovyInDeg, float aspectRatio, float znear, float zfar);
Matrix4 Frustum(float l, float r, float b, float t, float zn, float zf);
Matrix4 Ortho(float left, float right, float bottom, float top, float near, float far);

Matrix3 Invert(Matrix3 in);
Matrix4 Invert(Matrix4 in);
Matrix3 InverseTranspose(Matrix4 in);

Matrix3 M4toM3(Matrix4 m);
Matrix4 M3toM4(Matrix3 m);

Vector3 V4toV3(Vector4 v);
Vector4 V3toV4(Vector3 v);

void Print(Matrix4 in);
void Print(Vector3 in);

#ifdef __cplusplus

inline Vector3 operator+(const Vector3 &a, const Vector3 &b) {
	return Set(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vector3 operator-(const Vector3 &a, const Vector3 &b) {
	return Set(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vector3 operator-(const vec3 &a) {
	return Set(-a.x, -a.y, -a.z);
}

inline float operator*(const Vector3 &a, const Vector3 &b) {
	return ((a.x * b.x) + (a.y * b.y) + (a.z * b.z));
}

inline Vector3 operator*(const Vector3 &b, double a) {
	return Set(a * b.x, a * b.y, a * b.z);
}

inline Vector3 operator*(double a, const Vector3 &b) {
	return Set(a * b.x, a * b.y, a * b.z);
}

inline Vector3 operator/(const Vector3 &b, double a) {
	return Set(b.x / a, b.y / a, b.z / a);
}

inline void operator+=(Vector3 &a, const Vector3 &b) {
	a = a + b;
}

inline void operator-=(Vector3 &a, const Vector3 &b) {
	a = a - b;
}

inline void operator*=(Vector3 &a, const float &b) {
	a = a * b;
}

inline void operator/=(Vector3 &a, const float &b) {
	a = a / b;
}

inline Matrix4 operator*(const Matrix4 &a, const Matrix4 &b) {
	return Mult(a, b);
}

inline Matrix3 operator*(const Matrix3 &a, const Matrix3 &b) {
	return Mult(a, b);
}

inline Vector3 operator*(const Matrix4 &a, const Vector3 &b) {
	return Mult(a, b);
}

inline Vector4 operator*(const Matrix4 &a, const Vector4 &b) {
	return Mult(a, b);
}

inline Vector3 operator*(const Matrix3 &a, const Vector3 &b) {
	return Mult(a, b);
}

#endif

/*

=== IMPLEMENTATION ===

*/

char TRANSPOSED = 0;

// TODO: Implement all declared functions

#endif