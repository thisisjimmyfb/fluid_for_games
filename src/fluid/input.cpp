#include <stdlib.h>

#include "input.h"
#include "fluid.h"
#include "../FooBar/FooBar.h"
#include "particle.h"

#ifndef _WIN32
#include <GLUT/GLUT.h>
#include <OpenGL/OpenGL.h>
#else
#include "glut.h"
#include <GL/gl.h>
#pragma comment(lib,"freeglut")
#endif

extern FbCamera		g_Camera;

extern FbVector3	g_Spin;
extern float		g_SpinStrength;


int					g_MouseX, g_MouseY;
FbVector3			g_CamX, g_CamY, g_CamDisp;


extern void resetPosition();
extern FbVector3 g_Particle[];

void mouse(int button, int state, int x, int y)
{
	//------------------------------------------------------
	g_MouseX = x;
	g_MouseY = y;

	g_CamX = g_Camera.GetX();
	g_CamY = g_Camera.GetY();
	g_CamDisp = g_Camera.m_vCamPos - g_Camera.m_vLookAt;
}

void motion(int x, int y)
{
	float dx = ( x - g_MouseX ) / float(WINDOW_WIDTH);
	float dy = ( g_MouseY - y ) / float(WINDOW_HEIGHT);

	FbVector3 vDrag = dx * g_CamX + dy * g_CamY;

	FbMatrix4x4 rot = FbMatrix4x4::Identity;

	float fDragLength = Length(vDrag);

	if ( fDragLength > 0.0 )
	{
		static float s_fTweak = 0.5f * FB_PI;
		float fScale = s_fTweak;
		float fAngle = fScale * fDragLength;

		FbVector3 vPivot = Normalize( Cross( g_CamDisp, vDrag ) );

		rot = FbRotateMatrix( fAngle, vPivot );

		g_Camera.m_vCamPos = g_Camera.m_vLookAt + g_CamDisp * rot;
		g_Camera.m_vUp = g_CamY * rot;
	}

	//--------------------------------------------------
	{
		static float s_fSpinTweak = 2.5f * FB_PI;

		g_SpinStrength = s_fSpinTweak * Length(vDrag);

		FbVector3 vSpin = Cross( g_CamDisp, vDrag );

		if ( LengthSqr(vSpin) > 0.0f )
		{
			g_Spin = Normalize( vSpin );
		}
	}
}

void keyboard(unsigned char key, int x, int y)
{
    if('q' == key){
        exit(0);
    }
    else if ('p' ==key){
		resetPosition();

		particleSetConfig( &g_Particle[0] );
	}
}
