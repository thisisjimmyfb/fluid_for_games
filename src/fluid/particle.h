
#include "../FooBar/Math/FbVector.h"
#include <cuda_runtime.h>

void particleInit( int particleCount );
void particleDeinit();

void particleSetConfig( FbVector3 *config );

void particleUpdate( float dt, void * v );

void particleRender();

