

#define WINDOW_TITLE	"Fluid Simulation"
#define WINDOW_WIDTH	1024
#define WINDOW_HEIGHT	738

#define FB_PI           3.141592654f

#define FOV				(0.45f * FB_PI)
#define ZNEAR			0.1f
#define ZFAR			800.0f


//---------------------------------------------------------------
#define LATTICE_N				128
#define LATTICE_WIDTH			LATTICE_N
#define LATTICE_HEIGHT			LATTICE_N
#define LATTICE_DEPTH			LATTICE_N
#define GRID_SIZE				0.25f

#define PARTICLE_COUNT			(64*64*64)
//#define PARTICLE_COUNT			(1280000)//(64*32*32)
//#define PARTICLE_COUNT			(16*16*16)


#define HALF_WIDTH				(0.5f*LATTICE_WIDTH)
#define HALF_HEIGHT				(0.5f*LATTICE_HEIGHT)
#define HALF_DEPTH				(0.5f*LATTICE_DEPTH)

#define ITERATION				20

#define VISCOSITY				30.0f

#define GRAVITY					0.8f
//#define GRAVITY					0

