

#define WINDOW_TITLE	"Fluid Simulation"
#define WINDOW_WIDTH	1024
#define WINDOW_HEIGHT	738

#define FB_PI           3.141592654f

#define FOV				(0.45f * FB_PI)
#define ZNEAR			0.1f
#define ZFAR			800.0f


//---------------------------------------------------------------
#define LATTICE_LENGTH			(128)
#define GRID_SIZE				0.25f

#define PARTICLE_COUNT			(128*128*128)
//#define PARTICLE_COUNT			(64*64*64)
//#define PARTICLE_COUNT			(1280000)//(64*32*32)
//#define PARTICLE_COUNT			(16*16*16)

#define HALF_LENGTH				(LATTICE_LENGTH/2)

#define ITERATION				20

#define VISCOSITY				2.0f

#define GRAVITY					0.098f
//#define GRAVITY					0

