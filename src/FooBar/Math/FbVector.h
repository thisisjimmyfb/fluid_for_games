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
#ifndef FB_VECTOR_H
#define FB_VECTOR_H

#ifndef FB_MATH_H
#include <math.h>
#endif

struct FbVector3;
struct FbVector4;

struct FbVector3
{
							operator const float* () const;

	float &					operator []			(int a_nIndex);
	const float &			operator []			(int a_nIndex) const;
	FbVector3				operator -			() const;

	FbVector3				operator +			(const FbVector3 &a) const;
	FbVector3				operator -			(const FbVector3 &a) const;
	FbVector3				operator *			(float scale) const;
	friend FbVector3		operator *			(float scale, const FbVector3 &v);
	FbVector3				operator /			(float scale) const;

	const FbVector3 &		operator =			(const float a[3]);
	const FbVector3 &		operator +=			(const FbVector3 &a);
	const FbVector3 &		operator -=			(const FbVector3 &a);
	const FbVector3 &		operator *=			(float scale);
	const FbVector3 &		operator /=			(float scale);

	//----------------------------------------------------------------------------------
	bool					Near			( const FbVector3 &a, float epsilon = 0.0f ) const;

	bool					SanityCheck		() const;

	float					Pitch			() const;
	float					Yaw				() const;

	//----------------------------------------------------------------------------------
	friend FbVector3		MakeVector3		( float x, float y, float z );
	friend FbVector3		MakeVector3		( float yaw, float pitch );

	friend float			Dot				( const FbVector3 &a, const FbVector3 &b );
	friend FbVector3		Cross			( const FbVector3 &a, const FbVector3 &b );
	friend float			LengthSqr		( const FbVector3 &a );
	friend float			Length			( const FbVector3 &a );
	friend float			LengthXZSqr		( const FbVector3 &a );
	friend float			LengthXZ		( const FbVector3 &a );
	friend FbVector3		Normalize		( const FbVector3 &a );
	friend float			DistanceSqr		( const FbVector3 &a, const FbVector3 &b );
	friend float			Distance		( const FbVector3 &a, const FbVector3 &b );

	//----------------------------------------------------------------------
	union
	{
		struct { float x; float y; float z; };

		float v[3];
	};

	//----------------------------------------------------------------------
	static FbVector3		UnitVecZ;
	static FbVector3		UnitVecY;
	static FbVector3		UnitVecX;
	static FbVector3		ZeroVec;
};

struct FbVector4
{
							operator const float* () const;
							operator const FbVector3 () const;

	float &					operator []			(int a_nIndex);
	const float &			operator []			(int a_nIndex) const;
	FbVector4				operator -			() const;

	FbVector4				operator +			(const FbVector4 &a) const;
	FbVector4				operator -			(const FbVector4 &a) const;
	FbVector4				operator *			(float scale) const;
	friend FbVector4		operator *			(float scale, const FbVector4 &v);
	FbVector4				operator /			(float scale) const;

	const FbVector4 &		operator =			(const FbVector3 &a);
	const FbVector4 &		operator =			(const float a[4]);
	const FbVector4 &		operator +=			(const FbVector4 &a);
	const FbVector4 &		operator -=			(const FbVector4 &a);
	const FbVector4 &		operator *=			(float scale);
	const FbVector4 &		operator /=			(float scale);

	//----------------------------------------------------------------------------------
	bool					Near			( const FbVector4 &a, float epsilon = 0.0f ) const;

	bool					SanityCheck		() const;

	//----------------------------------------------------------------------------------
	friend FbVector4		MakeVector4		( float x, float y, float z, float w );
	friend FbVector4		MakeVector4		( const FbVector3 &a, float w );

	//----------------------------------------------------------------------
	union
	{
		struct { float x; float y; float z; float w; };

		float v[4];

		FbVector3 v3;
	};

	//----------------------------------------------------------------------
	static FbVector4		UnitVecZ;
	static FbVector4		UnitVecY;
	static FbVector4		UnitVecX;
	static FbVector4		ZeroVec;
	static FbVector4		ZeroPoint;
};

//---------------------------------------------------------------------------------------

inline						FbVector3::operator const float* () const					{ return v; }
inline float &				FbVector3::operator []			(int a_nIndex)				{ return v[a_nIndex]; }
inline const float &		FbVector3::operator []			(int a_nIndex) const		{ return v[a_nIndex]; }
inline FbVector3			FbVector3::operator -			() const					{ FbVector3 temp; temp.x = -x; temp.y = -y; temp.z = -z; return temp; }

inline FbVector3			FbVector3::operator +			(const FbVector3 &a) const	{ FbVector3 temp; temp.x = x + a.x, temp.y = y + a.y, temp.z = z + a.z; return temp; }
inline FbVector3			FbVector3::operator -			(const FbVector3 &a) const	{ FbVector3 temp; temp.x = x - a.x, temp.y = y - a.y, temp.z = z - a.z; return temp; }
inline FbVector3			FbVector3::operator *			(float scale) const			{ FbVector3 temp; temp.x = x * scale, temp.y = y * scale, temp.z = z * scale; return temp; }
inline FbVector3			operator *						(float scale, const FbVector3 &v)		{ return v * scale; }
inline FbVector3			FbVector3::operator /			(float scale) const			{ FbVector3 temp; temp.x = x / scale, temp.y = y / scale, temp.z = z / scale; return temp; }

inline const FbVector3 &	FbVector3::operator =			(const float a[3])			{ x = a[0]; y = a[1]; z = a[2]; return *this; }
inline const FbVector3 &	FbVector3::operator +=			(const FbVector3 &a)		{ x += a.x; y += a.y; z += a.z; return *this; }
inline const FbVector3 &	FbVector3::operator -=			(const FbVector3 &a)		{ x -= a.x; y -= a.y; z -= a.z; return *this; }
inline const FbVector3 &	FbVector3::operator *=			(float scale)				{ x *= scale; y *= scale; z *= scale; return *this; }
inline const FbVector3 &	FbVector3::operator /=			(float scale)				{ x /= scale; y /= scale; z /= scale; return *this; }

//---------------------------------------------------------------------------------------
inline bool FbVector3::Near( const FbVector3 &a, float epsilon ) const
{
	return fabs(x-a.x) <= epsilon && fabs(y-a.y) <= epsilon && fabs(z-a.z) <= epsilon;
}

inline bool FbVector3::SanityCheck() const
{
	return x == x && y == y && z == z;
}

inline float FbVector3::Pitch() const
{
	float h = Length(*this);
	return asinf( y / h );
}

inline float FbVector3::Yaw() const
{
	float h = LengthXZ(*this);
	return atan2f( x / h, z / h );
}

//---------------------------------------------------------------------------------------
inline FbVector3 MakeVector3( float x, float y, float z )
{
	FbVector3 v;
	v[0] = x, v[1] = y, v[2] = z;
	return v;
}

inline FbVector3 MakeVector3( float yaw, float pitch )
{
	float y = sinf(pitch);
	float h = sqrtf( 1.0f - y * y );
	return MakeVector3( h*sin(yaw), y, h*cos(yaw) );
}

inline float Dot( const FbVector3 &a, const FbVector3 &b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline FbVector3 Cross( const FbVector3 &a, const FbVector3 &b)
{
	FbVector3 temp;
	temp.x = a.y * b.z - a.z * b.y;
	temp.y = a.z * b.x - a.x * b.z;
	temp.z = a.x * b.y - a.y * b.x;
	return temp;
}

inline float LengthSqr( const FbVector3 &a )
{
	return Dot( a, a );
}

inline float Length( const FbVector3 &a )
{
	return sqrt( LengthSqr( a ) );
}

inline float LengthXZSqr( const FbVector3 &a )
{
	return a[0]*a[0] + a[2]*a[2];
}

inline float LengthXZ( const FbVector3 &a )
{
	return sqrt( LengthXZSqr( a ) );
}

inline FbVector3 Normalize( const FbVector3 &a )
{
	float fLength = Length(a);

	FbVector3 temp;
	temp.v[0] = a.v[0] / fLength;
	temp.v[1] = a.v[1] / fLength;
	temp.v[2] = a.v[2] / fLength;

	return temp;
}

inline float DistanceSqr( const FbVector3 &a, const FbVector3 &b )
{
	float dx = a.x - b.x;
	float dy = a.y - b.y;
	float dz = a.z - b.z;
	return dx * dx + dy * dy + dz * dz;
}

inline float Distance( const FbVector3 &a, const FbVector3 &b )
{
	return sqrt( DistanceSqr( a, b ) );
}

//---------------------------------------------------------------------------------------
inline						FbVector4::operator const float* () const					{ return v; }
inline						FbVector4::operator const FbVector3 () const				{ return v3; }

inline float &				FbVector4::operator []			(int a_nIndex)				{ return v[a_nIndex]; }
inline const float &		FbVector4::operator []			(int a_nIndex) const		{ return v[a_nIndex]; }
inline FbVector4			FbVector4::operator -			() const					{ FbVector4 temp; temp.x = -x; temp.y = -y; temp.z = -z; temp.w = -w; return temp; }

inline FbVector4			FbVector4::operator +			(const FbVector4 &a) const	{ FbVector4 temp; temp.x = x + a.x, temp.y = y + a.y, temp.z = z + a.z, temp.w = w + a.w; return temp; }
inline FbVector4			FbVector4::operator -			(const FbVector4 &a) const	{ FbVector4 temp; temp.x = x - a.x, temp.y = y - a.y, temp.z = z - a.z, temp.w = w - a.w; return temp; }
inline FbVector4			FbVector4::operator *			(float scale) const			{ FbVector4 temp; temp.x = x * scale, temp.y = y * scale, temp.z = z * scale, temp.w = w * scale; return temp; }
inline FbVector4			operator *						(float scale, const FbVector4 &v)		{ return v * scale; }
inline FbVector4			FbVector4::operator /			(float scale) const			{ FbVector4 temp; temp.x = x / scale, temp.y = y / scale, temp.z = z / scale, temp.w = w / scale; return temp; }

inline const FbVector4 &	FbVector4::operator =			(const FbVector3 &a)		{ x = a[0]; y = a[1]; z = a[2]; w = 0.0f; return *this; }
inline const FbVector4 &	FbVector4::operator =			(const float a[4])			{ x = a[0]; y = a[1]; z = a[2]; w = a[3]; return *this; }
inline const FbVector4 &	FbVector4::operator +=			(const FbVector4 &a)		{ x += a.x; y += a.y; z += a.z; w += a.w; return *this; }
inline const FbVector4 &	FbVector4::operator -=			(const FbVector4 &a)		{ x -= a.x; y -= a.y; z -= a.z; w -= a.w; return *this; }
inline const FbVector4 &	FbVector4::operator *=			(float scale)				{ x *= scale; y *= scale; z *= scale; w *= scale; return *this; }
inline const FbVector4 &	FbVector4::operator /=			(float scale)				{ x /= scale; y /= scale; z /= scale; w /= scale; return *this; }

//---------------------------------------------------------------------------------------
inline bool FbVector4::Near( const FbVector4 &a, float epsilon ) const
{
	return v3.Near(a.v3,epsilon) && fabs(w-a.w) <= epsilon;
}

inline bool FbVector4::SanityCheck() const
{
	return v3.SanityCheck() && w == w;
}

//---------------------------------------------------------------------------------------
inline FbVector4 MakeVector4( float x, float y, float z, float w )
{
	FbVector4 v;
	v[0] = x, v[1] = y, v[2] = z, v[3] = w;
	return v;
}

inline FbVector4 MakeVector4( const FbVector3 &a, float w )
{
	FbVector4 v;
	v.v3 = a, v[3] = w;
	return v;
}

#endif