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

//#include "Math/FbMath.h"
#include "FBCamera.h"

FbMatrix4x4 FbCamera::GetModelViewMatrix() const
{
	FbVector3 vZ = m_vCamPos - m_vLookAt;
	FbVector3 vY = m_vUp;
	FbVector3 vX = Normalize( Cross( vY, vZ ) );

	FbMatrix4x4 camMat;

	camMat[0] = Normalize( vX );
	camMat[2] = Normalize( vZ );
	camMat[1] = Normalize( Cross( vZ, vX ) );
	camMat[3] = MakeVector4( m_vCamPos, 1.0f );

	return FbSRTInverse(camMat);
}

FbMatrix4x4 FbCamera::GetProjectionMatrix() const
{
	float A = ( m_fRight + m_fLeft ) / ( m_fRight - m_fLeft );
	float B = ( m_fTop + m_fBottom ) / ( m_fTop - m_fBottom );
	float C = ( m_fFar + m_fNear ) / ( m_fNear - m_fFar );
	float D = 2.0f * m_fFar * m_fNear / ( m_fNear - m_fFar );

	FbMatrix4x4 projMat;

	projMat[0] = MakeVector4(	2.0f * m_fNear / ( m_fRight - m_fLeft ),
								0.0f,
								0.0f,
								0.0f );

	projMat[1] = MakeVector4(	0.0f,
								2.0f * m_fNear / ( m_fTop - m_fBottom ),
								0.0f,
								0.0f );

	projMat[2] = MakeVector4(	A, B, C, -1.0f );

	projMat[3] = MakeVector4(	0.0f, 0.0f, D, 0.0f );

	return projMat;
}

FbVector3 FbCamera::GetX() const
{
	return Normalize( Cross( m_vUp, m_vCamPos - m_vLookAt ) );
}

FbVector3 FbCamera::GetY() const
{
	return Normalize( Cross( m_vCamPos - m_vLookAt, GetX() ) );
}

FbVector3 FbCamera::GetZ() const
{
	return Normalize( m_vCamPos - m_vLookAt );
}

float FbCamera::GetFOV() const
{
	return 2.0f * atanf( m_fTop / m_fNear );
}

float FbCamera::GetAspect() const
{
	return ( m_fRight - m_fLeft ) / ( m_fTop - m_fBottom );
}

float FbCamera::GetNear() const
{
	return m_fNear;
}

float FbCamera::GetFar() const
{
	return m_fFar;
}

bool FbCamera::SanityCheck() const
{
	bool bQNANCheck =	m_vLookAt.SanityCheck() &&
						m_vCamPos.SanityCheck() &&
						m_vUp.SanityCheck();

	if ( !bQNANCheck ) return false;

	bool bUnitVecCheck = fabs( LengthSqr( m_vUp )-1.0f ) < 0.00001f;

	if ( !bUnitVecCheck ) return false;

	bool bZeroVecCheck = !m_vLookAt.Near( m_vCamPos );

	if ( !bZeroVecCheck ) return false;

	bool bUpDirCheck = Dot( m_vUp, m_vCamPos - m_vLookAt ) < 1.0f;

	if ( !bUpDirCheck ) return false;

	bool bNearPlaneCheck = fabs( m_fNear ) < 0.000001f;

	if ( !bNearPlaneCheck ) return false;

	return true;
}

void FbCamera::SetPerspective(float fovy, float aspect, float zNear, float zFar)
{
	m_fNear = zNear;
	m_fFar = zFar;

	m_fTop = tanf( .5f * fovy ) * zNear;
	m_fBottom = -m_fTop;
	m_fLeft = aspect * m_fBottom;
	m_fRight = aspect * m_fTop;
}
