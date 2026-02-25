
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "navier_stoke.h"
#include "fluid.h"

__device__ float3 getGridPos( int3 id )
{
	float3 grid_pos;
	grid_pos.x = ( id.x - HALF_LENGTH + 0.5f )	* GRID_SIZE;
	grid_pos.y = ( id.y - HALF_LENGTH + 0.5f )	* GRID_SIZE;
	grid_pos.z = ( id.z - HALF_LENGTH + 0.5f )	* GRID_SIZE;

	return grid_pos;
}

__device__ int3 getGridCoord( float3 pos )
{
	int3 grid_coord;
	grid_coord.x = int( pos.x / GRID_SIZE + HALF_LENGTH);
	grid_coord.y = int( pos.y / GRID_SIZE + HALF_LENGTH);
	grid_coord.z = int( pos.z / GRID_SIZE + HALF_LENGTH);

	grid_coord.x = max( 0, min( grid_coord.x, LATTICE_LENGTH-1 ) );
	grid_coord.y = max( 0, min( grid_coord.y, LATTICE_LENGTH-1 ) );
	grid_coord.z = max( 0, min( grid_coord.z, LATTICE_LENGTH-1 ) );

	return grid_coord;
}

__device__ float3 getGridEdgeCoord( int3 id )
{
	float3 grid_coord;
	grid_coord.x = ( id.x - HALF_LENGTH)	* GRID_SIZE;
	grid_coord.y = ( id.y - HALF_LENGTH)	* GRID_SIZE;
	grid_coord.z = ( id.z - HALF_LENGTH)	* GRID_SIZE;

	return grid_coord;
}

__device__ int3 getGridId( int3 offset )
{
	int3 id;
	id.x = blockIdx.x*blockDim.x + threadIdx.x + offset.x;
	id.y = blockIdx.y*blockDim.y + threadIdx.y + offset.y;
	id.z = blockIdx.z*blockDim.z + threadIdx.z + offset.z;
	
	return id;
}

__device__ int getGridIndex( int3 id )
{
	return	id.x * LATTICE_LENGTH * LATTICE_LENGTH +
			id.y * LATTICE_LENGTH +
			id.z;
}

__device__ float3 cross( float3 a, float3 b )
{
	float3 c;
	c.x = a.y*b.z - a.z*b.y;
	c.y = a.z*b.x - a.x*b.z;
	c.z = a.x*b.y - a.y*b.x;
	return c;
}

__device__ float norm( float3 v )
{
	return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z );
}

//--------------------------------------------------------------------------
static float3			*data[2];	//double buffering
static int				current;
static cudaStream_t		s_cudaStream;

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
struct SimpleConfig
{
	dim3			problemSize;

	dim3			blockDim;
	dim3			gridDim;
};

static void initSimpleConfig( SimpleConfig &config, cudaDeviceProp &prop )
{
	config.problemSize.x = LATTICE_LENGTH + 2;
	config.problemSize.y = LATTICE_LENGTH + 2;
	config.problemSize.z = LATTICE_LENGTH + 2;

	int blockSize = min(config.problemSize.x * config.problemSize.y * config.problemSize.z, prop.maxThreadsPerBlock);

	float fBlockLength = pow((float)blockSize, 0.3333333333f);
	//then truncate
	int nBlockLength = (int)fBlockLength;

	config.blockDim.x = min(min(config.problemSize.x, nBlockLength), prop.maxThreadsDim[0]);
	config.blockDim.y = min(min(config.problemSize.y, nBlockLength), prop.maxThreadsDim[1]);
	config.blockDim.z = min(min(config.problemSize.z, nBlockLength), prop.maxThreadsDim[2]);

	//----------------------------------------------------------------------
	config.gridDim.x = (config.problemSize.x + config.blockDim.x - 1) / config.blockDim.x;
	config.gridDim.y = (config.problemSize.y + config.blockDim.y - 1) / config.blockDim.y;
	config.gridDim.z = (config.problemSize.z + config.blockDim.z - 1) / config.blockDim.z;

	config.gridDim.x = min( config.gridDim.x, prop.maxGridSize[0] );
	config.gridDim.y = min( config.gridDim.y, prop.maxGridSize[1] );
	config.gridDim.z = min( config.gridDim.z, prop.maxGridSize[2] );
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
struct OverlapConfig
{
	dim3			problemSize;

	dim3			blockDim;
	dim3			gridDim;
};

static void initOverlapConfig( OverlapConfig &config, cudaDeviceProp &prop )
{
	config.problemSize.x = LATTICE_LENGTH;
	config.problemSize.y = LATTICE_LENGTH;
	config.problemSize.z = LATTICE_LENGTH;

	int size = min( prop.maxThreadsPerBlock, 
		LATTICE_LENGTH * LATTICE_LENGTH * LATTICE_LENGTH);

	//take the cube root
	float fBlockLength = pow( (float)size, 0.3333333333f );

	//then truncate
	int nBlockLength = (int)fBlockLength;

	config.blockDim.x = max(3, min( nBlockLength, prop.maxThreadsDim[0] ));
	config.blockDim.y = max(3, min( nBlockLength, prop.maxThreadsDim[1] ));
	config.blockDim.z = max(3, min( nBlockLength, prop.maxThreadsDim[2] ));

	//--------------------------------------------------------------------
	dim3 logicalSimDim;
	logicalSimDim.x = config.problemSize.x - 2;
	logicalSimDim.y = config.problemSize.y - 2;
	logicalSimDim.z = config.problemSize.z - 2;

	dim3 logicalBlockDim;
	logicalBlockDim.x = config.blockDim.x - 2;
	logicalBlockDim.y = config.blockDim.y - 2;
	logicalBlockDim.z = config.blockDim.z - 2;

	dim3 logicalGridDim;
	logicalGridDim.x = logicalSimDim.x / logicalBlockDim.x;
	logicalGridDim.y = logicalSimDim.y / logicalBlockDim.y;
	logicalGridDim.z = logicalSimDim.z / logicalBlockDim.z;

	logicalGridDim.x += logicalSimDim.x % logicalBlockDim.x ? 1:0;
	logicalGridDim.y += logicalSimDim.y % logicalBlockDim.y ? 1:0;
	logicalGridDim.z += logicalSimDim.z % logicalBlockDim.z ? 1:0;

	//----------------------------------------------------------------------
	config.gridDim.x = min( logicalGridDim.x, prop.maxGridSize[0] );
	config.gridDim.y = min( logicalGridDim.y, prop.maxGridSize[1] );
	config.gridDim.z = min( logicalGridDim.z, prop.maxGridSize[2] );
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
__global__ void set_bound( float3 *v, int dir )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	//this will be 0 (lower bound) or 1 (upper bound)
	unsigned int b = blockIdx.z*blockDim.z + threadIdx.z;

	//-----------------------------------------------------------
	// map our planar coordinate to 3D
	int3 id, neighborId;

	if ( dir == 0 ) //x direction => zy plane
	{
		id.x = b * (LATTICE_LENGTH-1);
		id.y = y;
		id.z = x;

		if ( id.z >= LATTICE_LENGTH || id.y >= LATTICE_LENGTH) return;

		neighborId.x = b ? LATTICE_LENGTH-2 : 1;
		neighborId.y = max( 1, min( id.y, LATTICE_LENGTH -2 ) );
		neighborId.z = max( 1, min( id.z, LATTICE_LENGTH -2 ) );
	}
	else if ( dir == 1 ) //y direction => xz plane
	{
		id.x = x;
		id.y = b * (LATTICE_LENGTH-1);
		id.z = y;

		if ( id.x >= LATTICE_LENGTH || id.z >= LATTICE_LENGTH) return;

		neighborId.x = max( 1, min( id.x, LATTICE_LENGTH-2 ) );
		neighborId.y = b ? LATTICE_LENGTH-2 : 1;
		neighborId.z = max( 1, min( id.z, LATTICE_LENGTH-2 ) );
	}
	else //if ( dir == 2 ) //z direction => xy plane
	{
		id.x = x;
		id.y = y;
		id.z = b * (LATTICE_LENGTH-1);

		if (id.x >= LATTICE_LENGTH || id.y >= LATTICE_LENGTH) return;

		neighborId.x = max( 1, min( id.x, LATTICE_LENGTH-2 ) );
		neighborId.y = max( 1, min( id.y, LATTICE_LENGTH-2 ) );
		neighborId.z = b ? LATTICE_LENGTH-2 : 1;
	}

	int neighborIndex = getGridIndex( neighborId );

	float3 vec = v[neighborIndex];

	// Apply the no-penetration condition by reflecting v' = v - 2(v·n)n
	// once per boundary face using axis-aligned unit normals.  For a face
	// cell this negates one component; for edge/corner cells each
	// boundary-normal component is negated independently.  Applying the
	// reflections one at a time (rather than combining into a single diagonal
	// normal) avoids cross-coupling that would otherwise trap particles at
	// edges and corners.
	if ( id.x == 0 || id.x == LATTICE_LENGTH-1  ) vec.x = -vec.x;
	if ( id.y == 0 || id.y == LATTICE_LENGTH-1 ) vec.y = -vec.y;
	if ( id.z == 0 || id.z == LATTICE_LENGTH-1  ) vec.z = -vec.z;

	//----------------------------------------------------------
	int index = getGridIndex( id );

	v[index] = vec;
}

//boundary operation works on 2D plane in 3 directions
static struct BoundConfig
{
	dim3			problemSize[3];

	dim3			blockDim[3];
	dim3			gridDim[3];

}					bound_config;

static void initBound( BoundConfig &config, cudaDeviceProp &prop )
{
	int blockSize = (int)sqrt( (float)(prop.maxThreadsPerBlock/2) );

	// zy plane
	config.problemSize[0].x = LATTICE_LENGTH;
	config.problemSize[0].y = LATTICE_LENGTH;
	config.problemSize[0].z = 2;

	// xz plane
	config.problemSize[1].x = LATTICE_LENGTH;
	config.problemSize[1].y = LATTICE_LENGTH;
	config.problemSize[1].z = 2;

	// xy plane
	config.problemSize[2].x = LATTICE_LENGTH;
	config.problemSize[2].y = LATTICE_LENGTH;
	config.problemSize[2].z = 2;

	for (int i=3;i--;)
	{
		config.blockDim[i].x = min( config.problemSize[i].x, blockSize );
		config.blockDim[i].y = min( config.problemSize[i].y, blockSize );
		config.blockDim[i].z = config.problemSize[i].z;

		config.blockDim[i].x = min( config.blockDim[i].x, prop.maxThreadsDim[0] );
		config.blockDim[i].y = min( config.blockDim[i].y, prop.maxThreadsDim[1] );
		config.blockDim[i].z = min( config.blockDim[i].z, prop.maxThreadsDim[2] );

		config.gridDim[i].x = config.problemSize[i].x / config.blockDim[i].x;
		config.gridDim[i].y = config.problemSize[i].y / config.blockDim[i].y;
		config.gridDim[i].z = config.problemSize[i].z / config.blockDim[i].z;

		config.gridDim[i].x += config.problemSize[i].x % config.blockDim[i].x ? 1:0;
		config.gridDim[i].y += config.problemSize[i].y % config.blockDim[i].y ? 1:0;
		config.gridDim[i].z += config.problemSize[i].z % config.blockDim[i].z ? 1:0;

		config.gridDim[i].x = min( config.gridDim[i].x, prop.maxGridSize[0] );
		config.gridDim[i].y = min( config.gridDim[i].y, prop.maxGridSize[1] );
		config.gridDim[i].z = min( config.gridDim[i].z, prop.maxGridSize[2] );
	}
}

void nsBound()
{
	for (int l=3;l--;)
	{
		set_bound<<<bound_config.gridDim[l],bound_config.blockDim[l],0,s_cudaStream>>>
			(data[current], l);
	}
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
__global__ void add_force( float3 *v, float3 force, float4 spin, float dt )
{
	int3 id = getGridId({ 0,0,0 });
	if (id.x < 0 || id.x >= LATTICE_LENGTH ||
		id.y < 0 || id.y >= LATTICE_LENGTH ||
		id.z < 0 || id.z >= LATTICE_LENGTH) return;

	int index = getGridIndex( id );

	float3 vec = v[index];

	//---------------------------------------------------------
	vec.x += force.x * dt;
	vec.y += force.y * dt;
	vec.z += force.z * dt;

	float3 grid_pos = getGridPos( id );

	float3 axis;
	axis.x = spin.x;
	axis.y = spin.y;
	axis.z = spin.z;

	force = cross( axis, grid_pos );

	float force_magnitude = norm( force );

	if ( fabs(force_magnitude) > FLT_EPSILON )
	{
		vec.x += spin.w * force.x / force_magnitude * dt;
		vec.y += spin.w * force.y / force_magnitude * dt;
		vec.z += spin.w * force.z / force_magnitude * dt;
	}

	v[index] = vec;
}

static struct AddForceConfig : SimpleConfig
{

}					addforce_config;

static void initAddForce( AddForceConfig &config, cudaDeviceProp &prop )
{
	initSimpleConfig( config, prop );
}

static void nsAddForce(	const FbVector3 &force,
						const FbVector3 &axis, float spin, float dt )
{
	float3 f;
	f.x = force.x,	f.y = force.y,	f.z = force.z;

	float4 a;
	a.x = axis.x,	a.y = axis.y,	a.z = axis.z,	a.w = spin;

	add_force<<<addforce_config.gridDim,addforce_config.blockDim,0,s_cudaStream>>>( data[current], f, a, dt );
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//----------------------------------------------------------------------
// since diffusion requires reading neighbor values, values are first
// read into shared memory, then used for diffusion calculation. This
// requires overlapping blocks so that there is no gap between blocks.
//----------------------------------------------------------------------
__global__ void stam_diffuse(const float3 * __restrict__ input, float3 *output,
							 float A, float C, float dt )
{
	extern __shared__ float3 shared[];

	int3 shift;
	shift.x = -2*blockIdx.x;
	shift.y = -2*blockIdx.y;
	shift.z = -2*blockIdx.z;

	int3 id = getGridId( shift );
	
	if (id.x < 0 ||
		id.y < 0 ||
		id.z < 0 ||
		id.x >= LATTICE_LENGTH ||
		id.y >= LATTICE_LENGTH ||
		id.z >= LATTICE_LENGTH)
		return;

	int index = getGridIndex( id );

	float3 vec = input[index];
	
	//---------------------------------------------------------
	int2 stride = { (int)(blockDim.y*blockDim.z), (int)blockDim.z };

	int tx = max(0, min(threadIdx.x, blockDim.x - 1));
	int ty = max(0, min(threadIdx.y, blockDim.y - 1));
	int tz = max(0, min(threadIdx.z, blockDim.z - 1));
	int sharedIndex = tx * stride.x + ty * stride.y + tz;

	shared[sharedIndex] = vec;
	
	__syncthreads();

	//---------------------------------------------------------
	bool bInnerBlock =	threadIdx.x > 0 &&
						threadIdx.y > 0 &&
						threadIdx.z > 0 &&
						threadIdx.x < blockDim.x-1 &&
						threadIdx.y < blockDim.y-1 &&
						threadIdx.z < blockDim.z-1;

	if ( bInnerBlock )
	{
		float3 neighborSum = { 0.0f, 0.0f, 0.0f };

		float3 neighbor;
		neighbor = shared[ sharedIndex - stride.x ];
		neighborSum.x += neighbor.x;
		neighborSum.y += neighbor.y;
		neighborSum.z += neighbor.z;

		neighbor = shared[ sharedIndex + stride.x ];
		neighborSum.x += neighbor.x;
		neighborSum.y += neighbor.y;
		neighborSum.z += neighbor.z;

		neighbor = shared[ sharedIndex - stride.y ];
		neighborSum.x += neighbor.x;
		neighborSum.y += neighbor.y;
		neighborSum.z += neighbor.z;

		neighbor = shared[ sharedIndex + stride.y ];
		neighborSum.x += neighbor.x;
		neighborSum.y += neighbor.y;
		neighborSum.z += neighbor.z;

		neighbor = shared[ sharedIndex - 1 ];
		neighborSum.x += neighbor.x;
		neighborSum.y += neighbor.y;
		neighborSum.z += neighbor.z;

		neighbor = shared[ sharedIndex + 1 ];
		neighborSum.x += neighbor.x;
		neighborSum.y += neighbor.y;
		neighborSum.z += neighbor.z;

		vec.x += A * neighborSum.x;
		vec.y += A * neighborSum.y;
		vec.z += A * neighborSum.z;

		vec.x /= C;
		vec.y /= C;
		vec.z /= C;

		output[index] = vec;
	}
}

static struct DiffuseConfig : OverlapConfig
{

} diffuse_config;

static void initDiffuse( DiffuseConfig &config, cudaDeviceProp &prop )
{
	initOverlapConfig( config, prop );
}

void nsDiffuse( float dt )
{
	int sharedMemSize = diffuse_config.blockDim.x *
						diffuse_config.blockDim.y *
						diffuse_config.blockDim.z * sizeof(float3);

	float A = dt * VISCOSITY * (LATTICE_LENGTH-2) * (LATTICE_LENGTH-2);

	float C = 1.0f + 6.0f*A;

	for (unsigned int l=ITERATION;l--;)
	{
		int next = 1 - current;

		stam_diffuse<<<diffuse_config.gridDim,diffuse_config.blockDim,sharedMemSize,s_cudaStream>>>
			( data[current], data[next], A, C, dt );
	
		current = next;
		
		nsBound();
	}
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
__global__ void stam_advect( const float3 * __restrict__ v_in, float3 *v_out, float dt )
{
	int3 id = getGridId({0,0,0});

	if (id.x < 0 || id.x >= LATTICE_LENGTH ||
		id.y < 0 || id.y >= LATTICE_LENGTH ||
		id.z < 0 || id.z >= LATTICE_LENGTH) return;

	int index = getGridIndex(id);
	
	int prev_index = index;

	float3 accum = { 0.0f, 0.0f, 0.0f };

	float t = dt;

	while ( t > 0.0f )
	{
		float3 vec = v_in[prev_index];

		float3 grid_pos = getGridPos( id );

		float3 grid_coord1, grid_coord2;
		grid_coord1 = getGridEdgeCoord( id );

		id.x += 1;
		id.y += 1;
		id.z += 1;

		grid_coord2 = getGridEdgeCoord( id );							

		//------------------------------------------------------------
		float3 displace_time1, displace_time2;
		
		displace_time1.x = fabs(vec.x) > FLT_EPSILON ? ( grid_coord1.x - grid_pos.x ) / -vec.x : 0.0f;
		displace_time2.x = fabs(vec.x) > FLT_EPSILON ? ( grid_coord2.x - grid_pos.x ) / -vec.x : 0.0f;
		
		displace_time1.y = fabs(vec.y) > FLT_EPSILON ? ( grid_coord1.y - grid_pos.y ) / -vec.y : 0.0f;
		displace_time2.y = fabs(vec.y) > FLT_EPSILON ? ( grid_coord2.y - grid_pos.y ) / -vec.y : 0.0f;
		
		displace_time1.z = fabs(vec.z) > FLT_EPSILON ? ( grid_coord1.z - grid_pos.z ) / -vec.z : 0.0f;
		displace_time2.z = fabs(vec.z) > FLT_EPSILON ? ( grid_coord2.z - grid_pos.z ) / -vec.z : 0.0f;
		
		//the reason i let division by zero happen above is because any 
		//boolean comparison with QNAN results in a false value, which will
		//prevent ddt be assigned such value here
		float ddt = FLT_MAX;

		if ( displace_time1.x >= 0.0f && displace_time1.x < ddt ) 
			ddt = displace_time1.x;
		if ( displace_time1.y >= 0.0f && displace_time1.y < ddt )
			ddt = displace_time1.y;
		if ( displace_time1.z >= 0.0f && displace_time1.z < ddt )
			ddt = displace_time1.z;
		if ( displace_time2.x >= 0.0f && displace_time2.x < ddt )
			ddt = displace_time2.x;
		if ( displace_time2.y >= 0.0f && displace_time2.y < ddt )
			ddt = displace_time2.y;
		if ( displace_time2.z >= 0.0f && displace_time2.z < ddt )
			ddt = displace_time2.z;

		if ( ddt == 0 ) ddt = t;

		ddt = min( ddt, t );

		grid_pos.x -= vec.x * ddt;
		grid_pos.y -= vec.y * ddt;
		grid_pos.z -= vec.z * ddt;

		t -= ddt;

		if (fabs(dt) > FLT_EPSILON)
		{
			accum.x += vec.x * ddt / dt;
			accum.y += vec.y * ddt / dt;
			accum.z += vec.z * ddt / dt;
		}

		id = getGridCoord( grid_pos );

		prev_index = getGridIndex( id );
	}

	//----------------------------------------------------
	v_out[index] = accum;
}

static struct AdvectConfig : SimpleConfig
{

}					advect_config;

static void initAdvect( AdvectConfig &config, cudaDeviceProp &prop )
{
	initSimpleConfig( config, prop );
}

void nsAdvect( float dt )
{
	int next = (current+1)%2;

	//we do the trace method by jo stam
	stam_advect<<<advect_config.gridDim,advect_config.blockDim,0,s_cudaStream>>>
		( data[current], data[next], dt );

	current = next;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
__global__ void compute_div_and_reset_pressure(float* div, float* p, const float3* __restrict__ v, float H)
{
	extern __shared__ float3 shared[];

	int3 shift;
	shift.x = -2 * int(blockIdx.x);
	shift.y = -2 * int(blockIdx.y);
	shift.z = -2 * int(blockIdx.z);

	int3 gridId = getGridId(shift);
	if (gridId.x < 0 ||
		gridId.y < 0 ||
		gridId.z < 0 ||
		gridId.x >= LATTICE_LENGTH ||
		gridId.y >= LATTICE_LENGTH ||
		gridId.z >= LATTICE_LENGTH)
		return;

	//---------------------------------------------------------
	int index = getGridIndex(gridId);

	float3 vec = v[index];

	int2 stride = { (int)(blockDim.y * blockDim.z), (int)blockDim.z };

	int sharedIndex = threadIdx.x * stride.x +
					  threadIdx.y * stride.y +
		              threadIdx.z;

	shared[sharedIndex] = vec;

	__syncthreads();

	//---------------------------------------------------------
	// don't write to edges
	if (threadIdx.x == 0 ||
		threadIdx.y == 0 ||
		threadIdx.z == 0 ||
		threadIdx.x == blockDim.x-1 ||
		threadIdx.y == blockDim.y-1 ||
		threadIdx.z == blockDim.z-1)
		return;
	
	float divergence = 0.0f;

	divergence -= shared[sharedIndex - stride.x].x;

	divergence += shared[sharedIndex + stride.x].x;

	divergence -= shared[sharedIndex - stride.y].y;

	divergence += shared[sharedIndex + stride.y].y;

	divergence -= shared[sharedIndex - 1].z;

	divergence += shared[sharedIndex + 1].z;

	divergence *= H;

	div[index] = divergence;

	p[index] = 0.0f;
}

static struct DivConfig : OverlapConfig
{
	//will be used to hold our divergence and height value
	float			*div;

}					div_config;

static void initDiv( DivConfig &config, cudaDeviceProp &prop )
{
	initOverlapConfig( config, prop );

	//----------------------------------------------------------------------
	int memSize = LATTICE_LENGTH * LATTICE_LENGTH * LATTICE_LENGTH * sizeof(float);

	cudaMalloc( &config.div, memSize );
	cudaMemset( config.div, 0, memSize );
}

static void deinitDiv( DivConfig &config )
{
	cudaFree( config.div );
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
__global__ void compute_pressure( const float * __restrict__ p_in, float *p_out, const float * __restrict__ div )
{
	extern __shared__ float sharedMem[];

	int3 shift;
	shift.x = -2 * int(blockIdx.x);
	shift.y = -2 * int(blockIdx.y);
	shift.z = -2 * int(blockIdx.z);

	int3 id = getGridId( shift );

	if (id.x < 0 ||
		id.y < 0 ||
		id.z < 0 ||
		id.x >= LATTICE_LENGTH ||
		id.y >= LATTICE_LENGTH ||
		id.z >= LATTICE_LENGTH)
		return;

	int index = getGridIndex( id );

	//---------------------------------------------------------
	int2 stride = { (int)(blockDim.y*blockDim.z), (int)blockDim.z };

	int tx = max(0, min(threadIdx.x, blockDim.x - 1));
	int ty = max(0, min(threadIdx.y, blockDim.y - 1));
	int tz = max(0, min(threadIdx.z, blockDim.z - 1));
	int sharedIndex = tx * stride.x + ty * stride.y + tz;

	// Load current pressure (not divergence) into shared memory
	sharedMem[sharedIndex] = p_in[index];

	__syncthreads();

	//---------------------------------------------------------
	bool bInnerBlock =	threadIdx.x > 0 &&
						threadIdx.y > 0 &&
						threadIdx.z > 0 &&
						threadIdx.x < blockDim.x-1 &&
						threadIdx.y < blockDim.y-1 &&
						threadIdx.z < blockDim.z-1;

	// Dirichlet BC takes priority: boundary cells are always zero pressure,
	// regardless of thread index. This prevents overlap-zone threads whose
	// domain-side neighbor is OOB.
	if (id.x == 0 || id.x == LATTICE_LENGTH - 1 ||
		id.y == 0 || id.y == LATTICE_LENGTH - 1 ||
		id.z == 0 || id.z == LATTICE_LENGTH - 1)
	{
		p_out[index] = 0.0f;
	}
	else if (bInnerBlock)
	{
		const float C = 1.0f / 6.0f;

		// p_new = (div + sum_neighbors) / 6
		float sum = sharedMem[ sharedIndex - stride.x ]
				  + sharedMem[ sharedIndex + stride.x ]
				  + sharedMem[ sharedIndex - stride.y ]
				  + sharedMem[ sharedIndex + stride.y ]
				  + sharedMem[ sharedIndex - 1 ]
				  + sharedMem[ sharedIndex + 1 ];

		p_out[index] = (div[index] + sum) * C;
	}
}

static struct PressureConfig : SimpleConfig
{
	//will be used to hold our pressure field
	float *p[2];

} pressure_config;

static void initPressure(PressureConfig &config, cudaDeviceProp &prop)
{
	initSimpleConfig( config, prop );

	//----------------------------------------------------------------------
	int memSize = LATTICE_LENGTH * LATTICE_LENGTH * LATTICE_LENGTH * sizeof(float);

	cudaMalloc(&config.p[0], memSize);
	cudaMalloc(&config.p[1], memSize);
	cudaMemset(config.p[0], 0, memSize);
	cudaMemset(config.p[1], 0, memSize);
}

static void deinitPressure(PressureConfig &config)
{
	cudaFree(config.p[1]);
	cudaFree(config.p[0]);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
__global__ void subtract_gradient( float3 *v, const float * __restrict__ p, float A )
{
	extern __shared__ float sharedMem[];

	int3 shift;
	shift.x = -2 * int(blockIdx.x);
	shift.y = -2 * int(blockIdx.y);
	shift.z = -2 * int(blockIdx.z);

	int3 id = getGridId( shift );

	if (id.x < 0 ||
		id.y < 0 ||
		id.z < 0 ||
		id.x >= LATTICE_LENGTH - 1 ||
		id.y >= LATTICE_LENGTH - 1 ||
		id.z >= LATTICE_LENGTH - 1)
		return;

	int index = getGridIndex( id );

	float pressure = p[index];
	
	//---------------------------------------------------------
	int2 stride = { (int)(blockDim.y*blockDim.z), (int)blockDim.z };

	int sharedIndex =	threadIdx.x*stride.x +
						threadIdx.y*stride.y +
						threadIdx.z;

	sharedMem[sharedIndex] = pressure;
	
	__syncthreads();

	//---------------------------------------------------------
	bool bInnerBlock = threadIdx.x > 0 &&
					   threadIdx.y > 0 &&
					   threadIdx.z > 0 &&
					   threadIdx.x < blockDim.x-1 &&
					   threadIdx.y < blockDim.y-1 &&
					   threadIdx.z < blockDim.z-1;

	if ( bInnerBlock )
	{
		float3 vec = v[index];

		float pressure_left = sharedMem[sharedIndex - stride.x];
		float pressure_right = sharedMem[ sharedIndex + stride.x ];
		float pressure_down = sharedMem[ sharedIndex - stride.y ];
		float pressure_up = sharedMem[ sharedIndex + stride.y ];
		float pressure_front = sharedMem[ sharedIndex - 1 ];
		float pressure_back = sharedMem[ sharedIndex + 1 ];

		vec.x -= A * (pressure_right - pressure_left);
		vec.y -= A * (pressure_up - pressure_down);
		vec.z -= A * (pressure_back - pressure_front);

		v[index] = vec;
	}
}

static struct SubGradConfig : OverlapConfig
{
}					subgrad_config;

static void initSubGrad( SubGradConfig &config, cudaDeviceProp &prop )
{
	initOverlapConfig( config, prop );
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void nsProject( float dt )
{	
	int next = 1 - current;

	//-----------------------------------------------------------
	//compute divergence
	//-----------------------------------------------------------
	{
		int sharedMemSize = div_config.blockDim.x *
							div_config.blockDim.y *
							div_config.blockDim.z * sizeof(float3);

		float H = -0.5f / (LATTICE_LENGTH-2);
	
		compute_div_and_reset_pressure<<<div_config.gridDim,div_config.blockDim,sharedMemSize,s_cudaStream>>>
			( div_config.div, pressure_config.p[current], data[current], H);
	}

	//-----------------------------------------------------------
	// compute pressure field
	// Each Jacobi iteration reads from p[current] and writes to p[1-current],
	// then swaps, so reads and writes never alias.
	{
		int sharedMemSize = pressure_config.blockDim.x *
							pressure_config.blockDim.y *
							pressure_config.blockDim.z * sizeof(float);

		for (unsigned int l = ITERATION; l--;)
		{
			compute_pressure<<<pressure_config.gridDim, pressure_config.blockDim, sharedMemSize, s_cudaStream>>>
				( pressure_config.p[current], pressure_config.p[next], div_config.div );
	
			current = next;
			next = 1 - current;
		}
	}

	//-----------------------------------------------------------
	// subtract gradient
	//-----------------------------------------------------------
	{
		int sharedMemSize = subgrad_config.blockDim.x *
							subgrad_config.blockDim.y *
							subgrad_config.blockDim.z * sizeof(float);

		float A = 0.5f * (LATTICE_LENGTH-2);

	
		subtract_gradient<<<subgrad_config.gridDim, subgrad_config.blockDim, sharedMemSize, s_cudaStream>>>
			(data[current], pressure_config.p[next], A);
	}
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void _printOutProp(const cudaDeviceProp& prop)
{
    printf("cudaDeviceProp::canMapHostMemory %d \n", prop.canMapHostMemory);
    //printf("cudaDeviceProp::clockRate %d \n", prop.clockRate);
    //printf("cudaDeviceProp::computeMode %d \n", prop.computeMode);
    //printf("cudaDeviceProp::deviceOverlap %d \n", prop.deviceOverlap);
    printf("cudaDeviceProp::integrated %d \n", prop.integrated);
    //printf("cudaDeviceProp::kernelExecTimeoutEnabled %d \n", prop.kernelExecTimeoutEnabled);
    printf("cudaDeviceProp::major %d \n", prop.major);
    printf("cudaDeviceProp::maxGridSize %d %d %d \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("cudaDeviceProp::maxThreadsDim %d %d %d \n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("cudaDeviceProp::maxThreadsPerBlock %d \n", prop.maxThreadsPerBlock);
    printf("cudaDeviceProp::memPitch %zu \n", prop.memPitch);
    printf("cudaDeviceProp::minor %d \n", prop.minor);
    printf("cudaDeviceProp::multiProcessorCount %d \n", prop.multiProcessorCount);
    printf("cudaDeviceProp::name %s \n", prop.name);
    printf("cudaDeviceProp::regsPerBlock %d \n", prop.regsPerBlock);
    printf("cudaDeviceProp::sharedMemPerBlock %zu \n", prop.sharedMemPerBlock);
    printf("cudaDeviceProp::textureAlignment %zu \n", prop.textureAlignment);
    printf("cudaDeviceProp::totalConstMem %zu \n", prop.totalConstMem);
    printf("cudaDeviceProp::totalGlobalMem %zu \n", prop.totalGlobalMem);
    printf("cudaDeviceProp::warpSize %d \n", prop.warpSize);
}

void nsInit( const FbVector3 *initConfig )
{
	
	//-----------------------------------------------------------------------
	cudaDeviceProp prop;
	int device = 0;
	cudaGetDeviceProperties(&prop, device);

	cudaChooseDevice( &device, &prop );
	cudaSetDevice( device );

	cudaGetDeviceProperties( &prop, device );
    
    _printOutProp(prop);

	cudaStreamCreate(&s_cudaStream);

	//-----------------------------------------------------------------------
	initAddForce( addforce_config, prop );

	initDiffuse( diffuse_config, prop );

	initBound( bound_config, prop );

	initAdvect( advect_config, prop );

	initDiv( div_config, prop );

	initPressure( pressure_config, prop );

	initSubGrad( subgrad_config, prop );

	//-----------------------------------------------------------------------
	int memSize = LATTICE_LENGTH * LATTICE_LENGTH * LATTICE_LENGTH * sizeof(float3);

	cudaMalloc( &data[0], memSize );
	cudaMalloc( &data[1], memSize );

	current = 0;

	if ( initConfig )
	{
		cudaMemcpy( data[current], initConfig, memSize, cudaMemcpyHostToDevice );
	}

	//-----------------------------------------------------------------------

}

void nsDeinit()
{
	deinitDiv( div_config );
	deinitPressure( pressure_config );

	cudaFree( data[0] );
	cudaFree( data[1] );

	cudaStreamDestroy(s_cudaStream);
}

void nsStep( float dt, const FbVector3 &force, const FbVector3 &axis, float spin )
{
	nsAddForce(force, axis, spin, dt);

	nsDiffuse(dt);

	nsProject(dt);

	nsAdvect(dt);

	nsProject(dt);

	nsBound();

	//cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("ERROR(%d): %s\n", err, cudaGetErrorString(err));
		exit(-1);
	}
}

void * nsGetVelocityGrid()
{
	return data[current];
}




