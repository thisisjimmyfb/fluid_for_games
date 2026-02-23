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
#ifndef FB_AUTOARRAY_H
#define FB_AUTOARRAY_H

#include <new>

template <typename T>
class FbAutoArray  
{
public:
					FbAutoArray(size_t a_Size);
					~FbAutoArray();

					operator T* ();

private:
	T				*buffer;
	size_t			size;
};

template<typename T>
inline FbAutoArray<T>::FbAutoArray(size_t a_Size)
{
	size = a_Size;
	buffer = new T[size];
}

template<typename T>
inline FbAutoArray<T>::~FbAutoArray()
{
	delete[] buffer;
}

template<typename T>
inline FbAutoArray<T>::operator T* ()
{
	return buffer;
}

#endif
