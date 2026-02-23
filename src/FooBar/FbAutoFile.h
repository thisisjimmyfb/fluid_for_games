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
#ifndef FB_AUTOFILE_H
#define FB_AUTOFILE_H

#include <stdio.h>

class FbAutoFile
{
public:
	enum EFileAccess
	{
		eFileAccess_ReadText,
		eFileAccess_WriteText,
		eFileAccess_ReadWriteText,

		eFileAccess_ReadBinary,
		eFileAccess_WriteBinary,
		eFileAccess_ReadWriteBinary,

		eFileAccess_Count
	};

	FbAutoFile( const char *sFileName, EFileAccess eFileAccess );
	~FbAutoFile();

	operator FILE *();

private:
	FILE * file;
};

inline FbAutoFile::FbAutoFile( const char *sFileName, EFileAccess eFileAccess )
{
	static const char * sFileAccessStr[eFileAccess_Count] =
	{
		"r",	//eFileAcess_ReadText
		"w",	//eFileAcess_WriteText
		"rw",	//eFileAcess_ReadWriteText

		"rb",	//eFileAcess_ReadBinary
		"wb",	//eFileAcess_WriteBinary
		"rwb",	//eFileAcess_ReadWriteBinary
	};

	file = fopen(sFileName, sFileAccessStr[eFileAccess]);
}

inline FbAutoFile::~FbAutoFile()
{
	if (file) fclose(file);
}

inline FbAutoFile::operator FILE *()
{
	return file;
}

#endif