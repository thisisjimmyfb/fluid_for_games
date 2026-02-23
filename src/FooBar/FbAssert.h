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
#ifndef FB_ASSERT_H
#define FB_ASSERT_H


namespace FbAssert
{
	template <bool CONDITION>	class FbCompileAssert;
	template < >				class FbCompileAssert<true>	{};
}

#define FB_COMPILE_ASSERT(X) enum {	FB_COMPILE_ASSERT_##_FILE_##_LINE_ = sizeof( FbAssert::FbCompileAssert< X > ) } 


#if defined(RELEASE) || defined(_RELEASE)
#define FB_ASSERT( X )	((void*)0)
#else
#include <assert.h>
#define FB_ASSERT( X )	assert(X)

#endif


#endif	//FB_ASSERT_H