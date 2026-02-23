#ifndef _WIN32
#include <OpenGL/OpenGL.h>
#else
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <GL/glew.h>
#endif

#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "particle.h"
#include "fluid.h"


__device__ int3 getGridId( float3 pos )
{
	int3 grid_id;
	grid_id.x = int( pos.x / GRID_SIZE + HALF_WIDTH );
	grid_id.y = int( pos.y / GRID_SIZE + HALF_HEIGHT );
	grid_id.z = int( pos.z / GRID_SIZE + HALF_DEPTH );

	return grid_id;
}

static __device__ float3 getGridEdgeCoord( int3 id )
{
	float3 grid_coord;
	grid_coord.x = ( id.x - HALF_WIDTH )	* GRID_SIZE;
	grid_coord.y = ( id.y - HALF_HEIGHT )	* GRID_SIZE;
	grid_coord.z = ( id.z - HALF_DEPTH )	* GRID_SIZE;

	return grid_coord;
}

__device__ float3 clampBounds( float3 p )
{
	float3 grid_range;
	grid_range.x = (HALF_WIDTH-1)*GRID_SIZE;
	grid_range.y = (HALF_HEIGHT-1)*GRID_SIZE;
	grid_range.z = (HALF_DEPTH-1)*GRID_SIZE;

	p.x = max( -grid_range.x, min( p.x, grid_range.x ) );
	p.y = max( -grid_range.y, min( p.y, grid_range.y ) );
	p.z = max( -grid_range.z, min( p.z, grid_range.z ) );

	return p;
}

__device__ float4 Box_Muller( float4 u )
{
	float2 s;
	s.x = u.x*u.x + u.y*u.y;
	s.y = u.z*u.z + u.w*u.w;

	s.x = sqrtf( -2.0f*logf(s.x)/s.x );
	s.y = sqrtf( -2.0f*logf(s.y)/s.y );

	float4 z;
	z.x = u.x * s.x;
	z.y = u.y * s.x;
	z.z = u.z * s.y;
	z.w = u.w * s.y;

	return z;
}

__global__ void particle(	float3 *p, int3 offset, dim3 dim,
							const float3 *v, float dt )
{
	int index_p = blockIdx.x*blockDim.x + threadIdx.x + offset.x;
	
	float3 pos = p[index_p];

#if 1
	while ( dt > 0.0f )
	{
		int3 grid_id = getGridId(pos);
		// Ensure grid_id is within bounds
		grid_id.x = max( 1, min( grid_id.x, LATTICE_WIDTH-2 ) );
		grid_id.y = max( 1, min( grid_id.y, LATTICE_HEIGHT-2 ) );
		grid_id.z = max( 1, min( grid_id.z, LATTICE_DEPTH-2 ) );

		int index_v = grid_id.x * LATTICE_HEIGHT * LATTICE_DEPTH +
					  grid_id.y * LATTICE_DEPTH +
					  grid_id.z;

		float3 vec = v[index_v];

		float3 grid_coord1, grid_coord2;
		grid_coord1 = getGridEdgeCoord(grid_id);

		grid_id.x += 1;
		grid_id.y += 1;
		grid_id.z += 1;

		grid_coord2 = getGridEdgeCoord(grid_id);

		float3 displace_time1, displace_time2;
		
		displace_time1.x = (grid_coord1.x - pos.x) / vec.x;
		displace_time2.x = ( grid_coord2.x - pos.x ) / vec.x;
		
		displace_time1.y = ( grid_coord1.y - pos.y ) / vec.y;
		displace_time2.y = ( grid_coord2.y - pos.y ) / vec.y;
		
		displace_time1.z = ( grid_coord1.z - pos.z ) / vec.z;
		displace_time2.z = ( grid_coord2.z - pos.z ) / vec.z;
		
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

		if ( ddt == 0 ) ddt = dt;

		ddt = min( ddt, dt );

		pos.x += vec.x * ddt;
		pos.y += vec.y * ddt;
		pos.z += vec.z * ddt;

		dt -= ddt;
	}
#else
	int3 grid_index = getGridId( pos );

	int index_v =	grid_index.x*LATTICE_HEIGHT*LATTICE_DEPTH +
					grid_index.y*LATTICE_DEPTH + 
					grid_index.z;

	float3 vec = v[index_v];

	pos.x += vec.x * dt;
	pos.y += vec.y * dt;
	pos.z += vec.z * dt;
#endif

	//-----------------------------------------------------------
	
	p[index_p] = clampBounds( pos );
}

//--------------------------------------------------------------------------
static dim3			runtimeBlockDim;
static dim3			runtimeGridDim;
static dim3			simulationDim;

static const int	STREAM_COUNT = 4;
static cudaStream_t	s_cudaStream[STREAM_COUNT];

static unsigned int	vbo;


static void computeConfiguration(	int device, dim3 problemSize,
									dim3 &gridSize, dim3 &blockSize )
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, device );

	//then truncate
	int blockLength = min( problemSize.x, prop.maxThreadsPerBlock );

	blockSize = dim3( blockLength );

	//-------------------------------------------------------------------------
	blockSize.x = min( blockSize.x, prop.maxThreadsDim[0] );
	blockSize.y = 1;
	blockSize.z = 1;

	gridSize.x = problemSize.x / blockSize.x + (problemSize.x % blockSize.x ? 1:0);
	gridSize.y = 1;
	gridSize.z = 1;

	//-------------------------------------------------------------------------
	gridSize.x = min( gridSize.x, prop.maxGridSize[0] );
	gridSize.y = 1;
	gridSize.z = 1;
}

void particleInit( int particleCount )
{
	simulationDim = dim3(particleCount);

	int device;
	cudaGetDevice(&device);
	computeConfiguration(device, simulationDim, runtimeGridDim, runtimeBlockDim);


	//-----------------------------------------------------------------------
	int memSize =	simulationDim.x *
					simulationDim.y *
					simulationDim.z * sizeof(float3);


#ifdef WIN32
	//-----------------------------------------------------------------------
	glewInit();
	if (	!GLEW_ARB_vertex_buffer_object ||
			!GLEW_ARB_point_sprite )
	{
		printf("OpenGL extension failed\n");
		exit(0);
	}
#endif

	cudaStreamCreate( &s_cudaStream[0] );
	cudaStreamCreate( &s_cudaStream[1] );
	cudaStreamCreate( &s_cudaStream[2] );
	cudaStreamCreate( &s_cudaStream[3] );

	//-----------------------------------------------------------------------

	glGenBuffersARB(1,&vbo);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB,vbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB,memSize,0,GL_STATIC_DRAW_ARB);

	int bsize;
	glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize); 
    if (bsize != memSize)
	{
		printf("something failed\n");
		__debugbreak();
	}

	glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);

	cudaGLRegisterBufferObject(vbo);
}

void particleDeinit()
{
	cudaGLUnregisterBufferObject(vbo);

	glDeleteBuffersARB(1,&vbo);

	cudaStreamDestroy( s_cudaStream[3] );
	cudaStreamDestroy( s_cudaStream[2] );
	cudaStreamDestroy( s_cudaStream[1] );
	cudaStreamDestroy( s_cudaStream[0] );
}

void particleSetConfig( FbVector3 *config )
{
	int memSize =	simulationDim.x *
					simulationDim.y *
					simulationDim.z * sizeof(float3);

	float3  *data;

	cudaGLMapBufferObject((void**)&data,vbo);
	
	cudaMemcpy( data, config, memSize, cudaMemcpyHostToDevice );
	
	cudaGLUnmapBufferObject(vbo);
}

void particleUpdate( float dt, void * v )
{
	//--------------------------------------------------------------
	float3 *data;

	cudaGLMapBufferObject((void**)&data,vbo);

	int batch_x = (int)(runtimeGridDim.x*runtimeBlockDim.x);
	int launch  = (int)(simulationDim.x/batch_x);

	for ( int i = 0; i < launch; )
	{
		unsigned int batch_count = min( launch-i, STREAM_COUNT );

		for ( unsigned int j = 0; j < batch_count; ++j,++i )
		{
			int3 offset = make_int3(i*batch_x,0,0);

			particle<<<runtimeGridDim,runtimeBlockDim,0,s_cudaStream[j]>>>(
				data,
				offset,
				simulationDim,
				(float3*)v,
				dt );
		}
	}


	//cudaError_t err = cudaGetLastError();

	cudaGLUnmapBufferObject(vbo);

}


void particleRender()
{
	glEnable(GL_POINT_SPRITE);
	
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER,vbo);
	
	glVertexPointer(3,GL_FLOAT,0,0);

	glDrawArrays(GL_POINTS,0,simulationDim.x);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	glDisableClientState(GL_VERTEX_ARRAY);

	glDisable(GL_POINT_SPRITE);
}
