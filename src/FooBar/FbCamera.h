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
#ifndef FB_CAMERA_H
#define FB_CAMERA_H

#include "Math/FbMatrix.h"

struct FbCamera
{
	FbMatrix4x4		GetModelViewMatrix() const;

	FbMatrix4x4		GetProjectionMatrix() const;

	FbVector3		GetX() const;
	FbVector3		GetY() const;
	FbVector3		GetZ() const;

	float			GetFOV() const;
	float			GetAspect() const;
	float			GetNear() const;
	float			GetFar() const;

	bool			SanityCheck() const;

	void			SetPerspective(	float fovy,
									float aspect,
									float zNear,
									float zFar );

	//--------------------------------------------------------------
	FbVector3		m_vLookAt;
	FbVector3		m_vCamPos;
	FbVector3		m_vUp;


	float			m_fRight;
	float			m_fLeft;
	float			m_fTop;
	float			m_fBottom;
	float			m_fFar;
	float			m_fNear;
};


#endif