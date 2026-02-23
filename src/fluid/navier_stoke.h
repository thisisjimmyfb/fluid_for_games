
#include "../FooBar/Math/FbVector.h"
#include <cuda_runtime.h>

void		nsInit( const FbVector3 *initConfig );
void		nsDeinit();

void		nsStep( float dt, const FbVector3 &force, const FbVector3 &axis, float spin );

void *		nsGetVelocityGrid();

void		nsAddForce( FbVector3 &axis, float spin );

void		nsCheck();