#include <stdio.h>
#include <algorithm>
#include <time.h>
#include <sys/stat.h>
#include "../FooBar/FooBar.h"
#include "input.h"
#include "fluid.h"
#include "navier_stoke.h"
#include "particle.h"

#ifndef _WIN32
#include <GLUT/GLUT.h>
#include <OpenGL/OpenGL.h>
#else
#include "glut.h"
#include <GL/gl.h>
#include <GL/wglew.h>
#pragma comment(lib,"freeglut")
#endif

using namespace std;

//---------------------------------------------------------------
float			g_TimeStep = 1.0f/60;

FbCamera		g_Camera;

FbVector3		g_Velocity[LATTICE_LENGTH][LATTICE_LENGTH][LATTICE_LENGTH];

FbVector3		g_Particle[PARTICLE_COUNT];

FbVector3		g_Spin = FbVector3::UnitVecY;
float			g_SpinStrength = 0.0f;

// use this function to read opencl files
char * readSource( const char *fileName )
{
	struct stat statbuf;
	stat( fileName, &statbuf );

	FILE *file = fopen( fileName, "rt" );

	unsigned int sourceSize = statbuf.st_size + 1;
	
	char *source = (char*)calloc(sourceSize,sizeof(char));

	unsigned int nSize = 0;

	do { source[nSize] = fgetc(file); } 
	while (	source[nSize++] != EOF && nSize < sourceSize );

	source[nSize-1] = NULL;

	fclose( file );

	return source;
}

void resetPosition()
{
	for (int i=PARTICLE_COUNT;i--;)
	{
		float x = rand() / float(RAND_MAX);
		float y = rand() / float(RAND_MAX);
		float z = rand() / float(RAND_MAX);

		x = (x * LATTICE_LENGTH - HALF_LENGTH) * GRID_SIZE;
		y = (y * LATTICE_LENGTH - HALF_LENGTH) * GRID_SIZE;
		z = (z * LATTICE_LENGTH - HALF_LENGTH) * GRID_SIZE;

		g_Particle[i] = MakeVector3( x, y, z );
	}

}

void init() {

	srand( time(0) );

	//----------------------------------------------------------
	for (int i=LATTICE_LENGTH;i--;)
	for (int j=LATTICE_LENGTH;j--;)
	for (int k=LATTICE_LENGTH;k--;)
	{
		float x = ( i - HALF_LENGTH + 0.5f ) * GRID_SIZE;
		float y = ( j - HALF_LENGTH + 0.5f ) * GRID_SIZE;
		float z = ( k - HALF_LENGTH + 0.5f ) * GRID_SIZE;

		FbVector3 r = MakeVector3( x, y, 0.0f );

		float R = HALF_LENGTH*HALF_LENGTH*GRID_SIZE*GRID_SIZE;

		if ( LengthSqr(r) > 0.0f && LengthSqr(r) < R )
		{
			g_Velocity[i][j][k] = 16.0f * Normalize( Cross( FbVector3::UnitVecZ, r ) );
		}
		else
		{
			g_Velocity[i][j][k] = FbVector3::ZeroVec;
		}
	}

	resetPosition();

	//----------------------------------------------------------
	nsInit(	&g_Velocity[0][0][0] );

	particleInit( PARTICLE_COUNT );

	particleSetConfig( &g_Particle[0] );

	//----------------------------------------------------------
	float aspect_ratio = float(WINDOW_WIDTH)/WINDOW_HEIGHT;

	g_Camera.SetPerspective( FOV, aspect_ratio, ZNEAR, ZFAR );

	float fSize = max(HALF_LENGTH,max(HALF_LENGTH, HALF_LENGTH)) * GRID_SIZE;

	float z = 2.5f * fSize / tanf( .5f * FOV );

	g_Camera.m_vCamPos = MakeVector3( 0.0f, 0.0f, z );
	g_Camera.m_vLookAt = MakeVector3( 0.0f, 0.0f, 0.0f );
	g_Camera.m_vUp = FbVector3::UnitVecY;

	//glPointSize(2);

	wglSwapIntervalEXT(1);
}

void deinit()
{
	particleDeinit();

	nsDeinit();
}

void renderLattice()
{
	float x = (HALF_LENGTH-1) * GRID_SIZE;
	float y = (HALF_LENGTH-1) * GRID_SIZE;
	float z = (HALF_LENGTH-1) * GRID_SIZE;

	glBegin(GL_LINE_LOOP);
		glVertex3f(	-x,	y,	z);
		glVertex3f(	-x,	-y,	z);
		glVertex3f(	x,	-y,	z);
		glVertex3f(	x,	y,	z);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(	-x,	y,	-z);
		glVertex3f(	-x,	-y,	-z);
		glVertex3f(	x,	-y,	-z);
		glVertex3f(	x,	y,	-z);
	glEnd();

	glBegin(GL_LINES);
		glVertex3f(	-x,	y,	-z);
		glVertex3f(	-x,	y,	z);
		glVertex3f(	x,	y,	-z);
		glVertex3f(	x,	y,	z);
	glEnd();

	glBegin(GL_LINES);
		glVertex3f(	-x,	-y,	-z);
		glVertex3f(	-x,	-y,	z);
		glVertex3f(	x,	-y,	-z);
		glVertex3f(	x,	-y,	z);
	glEnd();
}

void display() {

#if 1
	const float MAX_DT = 0.016666f;
	float dt = g_TimeStep < MAX_DT ? g_TimeStep : MAX_DT;
#else
	float dt = g_TimeStep;
#endif

	clock_t start_time = clock();

	FbVector3 gravity = -GRAVITY * Normalize( g_Camera.GetY() );

	nsStep( dt, gravity, g_Spin, g_SpinStrength  );

	g_Spin = FbVector3::UnitVecY;
	g_SpinStrength = 0.0f;

	
	particleUpdate(	dt, nsGetVelocityGrid() );

	//-------------------------------------------------------
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf( g_Camera.GetProjectionMatrix() );
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf( g_Camera.GetModelViewMatrix() );
	
	
	glColor3f(1.0f,0.0f,0.0f);

	particleRender();

	glColor3f( 0.0f, 0.0f, 1.0f );

	renderLattice();

    //glFlush();

	//-----------------------------------------------------------------
	clock_t end_time = clock();

	if ( end_time != start_time )
	{
		float fps = CLOCKS_PER_SEC / float(end_time - start_time);

		char window_title[128];
		sprintf(window_title,"%s - %f fps",WINDOW_TITLE,fps);

		glutSetWindowTitle(window_title);

		static float s_fBlend = 0.5f;
		g_TimeStep = (1.0f-s_fBlend)*g_TimeStep + s_fBlend/fps;
	}

    glutSwapBuffers();

}


int main (int argc, char ** argv) {

    glutInit(&argc, argv);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutCreateWindow(WINDOW_TITLE);

	//--------------------------------------------------------
	glutDisplayFunc(display);
    glutIdleFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable( GL_DEPTH_TEST );

	init();

	atexit( deinit );

    glutMainLoop();

    return 0;
}