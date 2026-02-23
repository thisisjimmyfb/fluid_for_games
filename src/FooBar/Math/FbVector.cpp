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

#include "FBVector.h"

FbVector3	FbVector3::UnitVecX		=	{	1.0f,	0.0f,	0.0f };
FbVector3	FbVector3::UnitVecY		=	{	0.0f,	1.0f,	0.0f };
FbVector3	FbVector3::UnitVecZ		=	{	0.0f,	0.0f,	1.0f };
FbVector3	FbVector3::ZeroVec		=	{	0.0f,	0.0f,	0.0f };

FbVector4	FbVector4::UnitVecX		=	{	1.0f,	0.0f,	0.0f,	0.0f };
FbVector4	FbVector4::UnitVecY		=	{	0.0f,	1.0f,	0.0f,	0.0f };
FbVector4	FbVector4::UnitVecZ		=	{	0.0f,	0.0f,	1.0f,	0.0f };
FbVector4	FbVector4::ZeroVec		=	{	0.0f,	0.0f,	0.0f,	0.0f };
FbVector4	FbVector4::ZeroPoint	=	{	0.0f,	0.0f,	0.0f,	1.0f };