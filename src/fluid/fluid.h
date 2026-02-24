

#define WINDOW_TITLE	"Fluid Simulation"
#define WINDOW_WIDTH	1024
#define WINDOW_HEIGHT	738

#define FB_PI           3.141592654f

#define FOV				(0.45f * FB_PI)
#define ZNEAR			0.1f
#define ZFAR			800.0f


//---------------------------------------------------------------
#define LATTICE_WIDTH			64
#define LATTICE_HEIGHT			64
#define LATTICE_DEPTH			64
#define GRID_SIZE				0.25f

#define PARTICLE_COUNT			(128*128*128)
//#define PARTICLE_COUNT			(1280000)//(64*32*32)
//#define PARTICLE_COUNT			(16*16*16)


#define HALF_WIDTH				(0.5f*LATTICE_WIDTH)
#define HALF_HEIGHT				(0.5f*LATTICE_HEIGHT)
#define HALF_DEPTH				(0.5f*LATTICE_DEPTH)

#define ITERATION				20

#define VISCOSITY				50.0f

//#define GRAVITY				9.8f
#define GRAVITY					0

