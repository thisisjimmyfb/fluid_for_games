//----------------------------------------------------------------
// FooBar library
// 
// Copyright (C) 2008 Jimmy Lee
//
// This program is not free software: you can not redistribute it 
// nor modify it. In fact, you can not use it.
//
// This program is distributed without the author's consent or
// ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR ANY PURPOSE.
//----------------------------------------------------------------

#pragma once
#ifndef FB_MATRIX_H
#define FB_MATRIX_H

#include "FBVector.h"

struct FbMatrix4x4
{
							operator const float *	() const;

	FbVector4 &				operator []				(int a_nIndex);
	const FbVector4 &		operator []				(int a_nIndex) const;

	const FbMatrix4x4 &		operator =				(const float a[16]);

	friend FbMatrix4x4		operator *				(const FbMatrix4x4 &a, const FbMatrix4x4 &b);

	friend FbVector4		operator *				(const FbMatrix4x4 &a, const FbVector4 &b);
	friend FbVector4		operator *				(const FbVector4 &a, const FbMatrix4x4 &b);

	friend FbVector3		operator *				(const FbMatrix4x4 &a, const FbVector3 &b);
	friend FbVector3		operator *				(const FbVector3 &a, const FbMatrix4x4 &b);

	//-------------------------------------------------------------------------

	//-------------------------------------------------------------------------
	friend FbMatrix4x4		FbScaleMatrix		(float x, float y, float z);
	friend FbMatrix4x4		FbPitchMatrix		(float fPitch);
	friend FbMatrix4x4		FbYawMatrix			(float fYaw);
	friend FbMatrix4x4		FbRollMatrix		(float fRoll);
	friend FbMatrix4x4		FbRotateMatrix		(float fAngle, const FbVector3 &vAxis);
	friend FbMatrix4x4		FbTranslateMatrix	(float x, float y, float z);
	friend FbMatrix4x4		FbDirectionMatrix	(const FbVector3 &z, const FbVector3 &y, const FbVector3 &pos = FbVector3::ZeroVec);
	friend FbMatrix4x4		FbTranspose			(const FbMatrix4x4 &t);
	friend FbMatrix4x4		FbSRTInverse		(const FbMatrix4x4 &i);

	//-------------------------------------------------------------------------
	union
	{
		FbVector4			m[4];

		float				f[16];
	};

	//-------------------------------------------------------------------------
	static FbMatrix4x4		Identity;
};


//-----------------------------------------------------------------------------
inline const FbMatrix4x4 &	FbMatrix4x4::operator =				(const float a[16])		{ m[0] = a, m[1] = a+4, m[2] = a+8, m[3] = a+12; return *this; }

inline FbVector4 &			FbMatrix4x4::operator []			(int a_nIndex)			{ return m[a_nIndex]; }
inline const FbVector4 &	FbMatrix4x4::operator []			(int a_nIndex) const	{ return m[a_nIndex]; }

inline						FbMatrix4x4::operator const float *	() const				{ return m[0].v; }

inline FbMatrix4x4 operator*(const FbMatrix4x4 &a, const FbMatrix4x4 &b)
{
	FbMatrix4x4 temp;

	temp[0][0] = a[0][0]*b[0][0] + a[1][0]*b[0][1] + a[2][0]*b[0][2] + a[3][0]*b[0][3];
	temp[1][0] = a[0][0]*b[1][0] + a[1][0]*b[1][1] + a[2][0]*b[1][2] + a[3][0]*b[1][3];
	temp[2][0] = a[0][0]*b[2][0] + a[1][0]*b[2][1] + a[2][0]*b[2][2] + a[3][0]*b[2][3];
	temp[3][0] = a[0][0]*b[3][0] + a[1][0]*b[3][1] + a[2][0]*b[3][2] + a[3][0]*b[3][3];

	temp[0][1] = a[0][1]*b[0][0] + a[1][1]*b[0][1] + a[2][1]*b[0][2] + a[3][1]*b[0][3];
	temp[1][1] = a[0][1]*b[1][0] + a[1][1]*b[1][1] + a[2][1]*b[1][2] + a[3][1]*b[1][3];
	temp[2][1] = a[0][1]*b[2][0] + a[1][1]*b[2][1] + a[2][1]*b[2][2] + a[3][1]*b[2][3];
	temp[3][1] = a[0][1]*b[3][0] + a[1][1]*b[3][1] + a[2][1]*b[3][2] + a[3][1]*b[3][3];

	temp[0][2] = a[0][2]*b[0][0] + a[1][2]*b[0][1] + a[2][2]*b[0][2] + a[3][2]*b[0][3];
	temp[1][2] = a[0][2]*b[1][0] + a[1][2]*b[1][1] + a[2][2]*b[1][2] + a[3][2]*b[1][3];
	temp[2][2] = a[0][2]*b[2][0] + a[1][2]*b[2][1] + a[2][2]*b[2][2] + a[3][2]*b[2][3];
	temp[3][2] = a[0][2]*b[3][0] + a[1][2]*b[3][1] + a[2][2]*b[3][2] + a[3][2]*b[3][3];

	temp[0][3] = a[0][3]*b[0][0] + a[1][3]*b[0][1] + a[2][3]*b[0][2] + a[3][3]*b[0][3];
	temp[1][3] = a[0][3]*b[1][0] + a[1][3]*b[1][1] + a[2][3]*b[1][2] + a[3][3]*b[1][3];
	temp[2][3] = a[0][3]*b[2][0] + a[1][3]*b[2][1] + a[2][3]*b[2][2] + a[3][3]*b[2][3];
	temp[3][3] = a[0][3]*b[3][0] + a[1][3]*b[3][1] + a[2][3]*b[3][2] + a[3][3]*b[3][3];

	return temp;
}

inline FbVector4 operator*(const FbMatrix4x4 &a, const FbVector4 &b)
{
	FbVector4 dest;
	dest[0] =	a[0][0]*b[0] + a[1][0]*b[1] + a[2][0]*b[2] + a[3][0]*b[3];
	dest[1] =	a[0][1]*b[0] + a[1][1]*b[1] + a[2][1]*b[2] + a[3][1]*b[3];	
	dest[2] =	a[0][2]*b[0] + a[1][2]*b[1] + a[2][2]*b[2] + a[3][2]*b[3];
	dest[3] =	a[0][3]*b[0] + a[1][3]*b[1] + a[2][3]*b[2] + a[3][3]*b[3];
	return dest;
}

inline FbVector4 operator*(const FbVector4 &a, const FbMatrix4x4 &b)
{
	FbVector4 dest;
	dest[0] =	a[0]*b[0][0] + a[1]*b[0][1] + a[2]*b[0][2] + a[3]*b[0][3];
	dest[1] =	a[0]*b[1][0] + a[1]*b[1][1] + a[2]*b[1][2] + a[3]*b[1][3];	
	dest[2] =	a[0]*b[2][0] + a[1]*b[2][1] + a[2]*b[2][2] + a[3]*b[2][3];
	dest[3] =	a[0]*b[3][0] + a[1]*b[3][1] + a[2]*b[3][2] + a[3]*b[3][3];
	return dest;
}

inline FbVector3 operator*(const FbMatrix4x4 &a, const FbVector3 &b)
{
	FbVector3 dest;
	dest[0] =	a[0][0]*b[0] + a[1][0]*b[1] + a[2][0]*b[2];
	dest[1] =	a[0][1]*b[0] + a[1][1]*b[1] + a[2][1]*b[2];	
	dest[2] =	a[0][2]*b[0] + a[1][2]*b[1] + a[2][2]*b[2];
	return dest;
}

inline FbVector3 operator*(const FbVector3 &a, const FbMatrix4x4 &b)
{
	FbVector3 dest;
	dest[0] =	a[0]*b[0][0] + a[1]*b[0][1] + a[2]*b[0][2];
	dest[1] =	a[0]*b[1][0] + a[1]*b[1][1] + a[2]*b[1][2];	
	dest[2] =	a[0]*b[2][0] + a[1]*b[2][1] + a[2]*b[2][2];
	return dest;
}

//-----------------------------------------------------------------------------
inline FbMatrix4x4 FbScaleMatrix(float x, float y, float z)
{
	FbMatrix4x4 mat;
	mat[0] = MakeVector4(	x,		0.0f,	0.0f,	0.0f );
	mat[1] = MakeVector4(	0.0f,	y,		0.0f,	0.0f );
	mat[2] = MakeVector4(	0.0f,	0.0f,	z,		0.0f );
	mat[3] = MakeVector4(	0.0f,	0.0f,	0.0f,	1.0f );
	return mat;
}

inline FbMatrix4x4 FbPitchMatrix(float fPitch)
{
	float c = cosf( fPitch );
	float s = sinf( fPitch );

	float x = FbVector4::UnitVecX[0];

	FbMatrix4x4 temp;
	temp.m[0] = FbVector4::UnitVecX;

	temp.m[1][0] = 0.0f;
	temp.m[1][1] = c;
	temp.m[1][2] = x*s;
	temp.m[1][3] = 0.0f;

	temp.m[2][0] = 0.0f;
	temp.m[2][1] = -x*s;
	temp.m[2][2] = c;
	temp.m[2][3] = 0.0f;

	temp.m[3] = FbVector4::ZeroPoint;
	return temp;
}

inline FbMatrix4x4 FbYawMatrix(float fYaw)
{
	float c = cosf( fYaw );
	float s = sinf( fYaw );

	float y = FbVector4::UnitVecY[1];

	FbMatrix4x4 temp;
	temp.m[0][0] = c;
	temp.m[0][1] = 0.0f;
	temp.m[0][2] = -y*s;
	temp.m[0][3] = 0.0f;

	temp.m[1] = FbVector4::UnitVecY;
	
	temp.m[2][0] = y*s;
	temp.m[2][1] = 0.0f;
	temp.m[2][2] = c;
	temp.m[2][3] = 0.0f;

	temp.m[3] = FbVector4::ZeroPoint;
	return temp;
}

inline FbMatrix4x4 FbRollMatrix(float fRoll)
{
	float c = cosf( fRoll );
	float s = sinf( fRoll );

	float z = FbVector4::UnitVecZ[2];

	FbMatrix4x4 temp;
	temp.m[0][0] = c;
	temp.m[0][1] = z*s;
	temp.m[0][2] = 0.0f;
	temp.m[0][3] = 0.0f;

	temp.m[1][0] = -z*s;
	temp.m[1][1] = c;
	temp.m[1][2] = 0.0f;
	temp.m[1][3] = 0.0f;

	temp.m[2] = FbVector4::UnitVecZ;
	temp.m[3] = FbVector4::ZeroPoint;
	return temp;
}

inline FbMatrix4x4 FbRotateMatrix(float fAngle, const FbVector3 &vAxis)
{
	//http://www.opengl.org/sdk/docs/man/xhtml/glRotate.xml

	float c = cosf( fAngle );
	float s = sinf( fAngle );
	float one_minus_c = 1 - c;

	const float &x = vAxis[0];
	const float &y = vAxis[1];
	const float &z = vAxis[2];

	float xx = x * x;
	float xy = x * y;
	float xz = x * z;
	float yy = y * y;
	float yz = y * z;
	float zz = z * z;

	float xs = x * s;
	float ys = y * s;
	float zs = z * s;

	FbMatrix4x4 temp;

	temp.m[0][0] = xx * one_minus_c + c;
	temp.m[0][1] = xy * one_minus_c + zs;
	temp.m[0][2] = xz * one_minus_c - ys;
	temp.m[0][3] = 0.0f;

	temp.m[1][0] = xy * one_minus_c - zs;
	temp.m[1][1] = yy * one_minus_c + c;
	temp.m[1][2] = yz * one_minus_c + xs;
	temp.m[1][3] = 0.0f;

	temp.m[2][0] = xz * one_minus_c + ys;
	temp.m[2][1] = yz * one_minus_c - xs;
	temp.m[2][2] = zz * one_minus_c + c;
	temp.m[2][3] = 0.0f;

	temp.m[3] = FbVector4::ZeroPoint;

	return temp;
}

inline FbMatrix4x4 FbTranslateMatrix(float x, float y, float z)
{
	FbMatrix4x4 mat = FbMatrix4x4::Identity;
	mat[3] = MakeVector4( x, y, z, 1.0f );
	return mat;
}

inline FbMatrix4x4 FbDirectionMatrix(const FbVector3 &z, const FbVector3 &y, const FbVector3 &pos)
{
	FbVector3 x = Normalize( Cross(y,z) );
	FbVector3 normz = Normalize( z );

	FbMatrix4x4 temp;
	temp.m[0] = MakeVector4( x,								0.0f );
	temp.m[2] = MakeVector4( Normalize( z ),				0.0f );
	temp.m[3] = MakeVector4( pos,							1.0f );
	//why the weird order? to reduce dependency between temp.m[1] and temp.m[0] 
	temp.m[1] = MakeVector4( Normalize( Cross(normz,x) ),	0.0f );
	return temp;
}

inline FbMatrix4x4 FbTranspose(const FbMatrix4x4 &t)
{
	FbMatrix4x4 temp;

	temp.m[0][0] = t.m[0][0];
	temp.m[1][1] = t.m[1][1];
	temp.m[2][2] = t.m[2][2];
	temp.m[3][3] = t.m[3][3];

	temp.m[0][1] = t.m[1][0];
	temp.m[0][2] = t.m[2][0];
	temp.m[0][3] = t.m[3][0];

	temp.m[1][0] = t.m[0][1];
	temp.m[1][2] = t.m[2][1];
	temp.m[1][3] = t.m[3][1];

	temp.m[2][0] = t.m[0][2];
	temp.m[2][1] = t.m[1][2];
	temp.m[2][3] = t.m[3][2];

	temp.m[3][0] = t.m[0][3];
	temp.m[3][1] = t.m[1][3];
	temp.m[3][2] = t.m[2][3];
	return temp;
}

inline FbMatrix4x4 FbSRTInverse(const FbMatrix4x4 &i)
{
	FbMatrix4x4 inverse = i;
	
	inverse[3] = FbVector4::ZeroPoint;

	inverse = FbTranspose( inverse );

	inverse[3][0] =	-Dot( i[0], i[3] );
	inverse[3][1] =	-Dot( i[1], i[3] );
	inverse[3][2] =	-Dot( i[2], i[3] );

	return inverse;
}

#endif