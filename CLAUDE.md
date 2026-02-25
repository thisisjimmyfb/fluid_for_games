
# Context: This implements navier-stokes fluid simulation in 3D base on the paper "Stable Fluids" (Jos Stam, SIGGRAPH 99)

## 1. Overview
**Core Contribution:** 
Proposes an unconditionally stable solver for the Navier-Stokes equations, designed specifically for computer graphics and animation. Unlike explicit solvers which require tiny time steps to avoid "blow-up," this implicit method allows for large time steps, enabling real-time interaction.

**Primary Use Cases:**
*   Simulation of gaseous phenomena (smoke, fire, watercolor).
*   Advecting textures (liquid textures).
*   Interactive fluid modeling (user applies forces in real-time).

**Key Trade-offs:**
*   **Stability > Accuracy:** The method is not physically accurate enough for engineering. It suffers from "numerical dissipation" (flow dampens too rapidly).
*   **Visuals First:** Designed for visual appearance (swirling motion) rather than strict conservation of physical quantities.

---

## 2. Mathematical Foundation

### 2.1 The Navier-Stokes Equations
The solver models a fluid with constant density rho (incompressible) governed by:

1.  **Mass Conservation (Incompressibility):** div u = 0
2.  **Momentum Conservation:** du/dt = -(u . grad)u - (1/rho)grad p + nu * Laplacian u + f

Where:
*   u: Velocity vector field.
*   p: Pressure field.
*   nu: Kinematic viscosity.
*   f: External force.

### 2.2 Helmholtz-Hodge Decomposition
To solve for velocity u, the paper uses the decomposition of any vector field w into a divergence-free part u and a gradient part grad q:
w = u + grad q

By taking the divergence of both sides, we get a Poisson equation for the scalar field q:
Laplacian q = div w

Solving this allows us to "project" any vector field w onto the space of divergence-free fields:
u = Pw = w - grad q

---

## 3. The Solver Algorithm (4-Step Method)

The simulation advances in time steps $\Delta t$. The general procedure (illustrated in Figure 1 of the paper) is:

w0 -- add force --> w1 -- advect --> w2 -- diffuse --> w3 -- project --> u4

### Step 1: Add External Force
Simple explicit Euler step. Forces are applied at the beginning of the time step.
w1(x) = w0(x) + dt * f(x, t)

### Step 2: Advection (Method of Characteristics)
Handles the non-linear term -(u . grad)u.
*   **Mechanism:** Trace the path of a fluid particle backward in time from current position x to a previous position p(x, -dt).
*   **Formula:** w2(x) = w1(p(x, -dt)).
*   **Implementation:** Requires a particle tracer and a linear interpolator. This step is unconditionally stable.

### Step 3: Diffusion (Viscosity)
Handles the term nu * Laplacian u.
*   **Mechanism:** Solved using an **implicit** Euler method to ensure stability.
*   **Equation:** (I - nu * dt * Laplacian)w3 = w2.
*   **Solver:** Results in a sparse linear system (Poisson-like).
    *   *Periodic Boundaries:* Solved via Fast Fourier Transform (FFT).
    *   *Fixed Boundaries:* Solved via multi-grid or relaxation methods (e.g., FISHPAK).

### Step 4: Projection (Enforce Divergence-Free)
Removes the divergence introduced by advection and diffusion.
*   **Mechanism:** Solves the Poisson equation Laplacian q = div w3.
*   **Result:** u4 = w3 - grad q.
*   **Solver:** Same linear solver as the Diffusion step.

---

## 4. Implementation Details

### 4.1 Grid Setup
*   **Cell-Centered:** Velocity is defined at the center of grid cells (Figure 3).
*   **Data Structures:**
    *   Velocity grids: U0, U1 (swapped each step).
    *   Scalar grids: S0, S1 (for density/temperature).
*   **Dimensions:** Supports 2D (N=2) and 3D (N=3).

### 4.2 Scalar Transport (Density/Color)
Scalars (like smoke density d) are advected and diffused but **do not** require the projection step (mass is not conserved in the same way, and there is no pressure term).
**Equation:** da/dt = -u . grad a + kappa_a * Laplacian a - alpha_a * a + S_a
*   Includes a dissipation term -alpha_a * a (dies out over time) and source term S_a.

### 4.3 Pseudocode Structure (From Section 3.2)

**Velocity Solver (`V step`):**
```c
for (i=0; i<NDIM; i++) {
    addForce ( U0[i], F[i], dt );
    Transport ( U1[i], U0[i], U0, dt );
    Diffuse ( U0[i], U1[i], visc, dt );
}
Project ( U1, U0, dt );

addForce ( S0, source, dt );
Transport ( S1, S0, U, dt );
Diffuse ( S0, S1, k, dt );
Dissipate ( S1, S0, a, dt );
```

**Scalar Solver ('S step'):**
```c
addForce ( S0, source, dt );
Transport ( S1, S0, U, dt );
Diffuse ( S0, S1, k, dt );
Dissipate ( S1, S0, a, dt );
```

### 4.4 Boundary Conditions
    1.  Periodic: Fluid wraps around (like Pac-Man). Allows use of FFT for O(NlogN) complexity.
    2.  Fixed: Fluid hits a wall (velocity normal to wall is zero). Requires iterative solvers (FISHPAK/Multi-grid).

## 5. Detailed Implementation & Algorithms

### 5.1 Particle Tracing (Advection)
The advection step relies on tracing the path of a particle backward in time.
*   **Algorithm:** Uses a simple second-order Runge-Kutta (RK2) method.
*   **Logic:**
    1.  Start at position $\mathbf{X}$.
    2.  Estimate velocity at halfway point.
    3.  Trace path over time $-\Delta t$ to find origin point $\mathbf{X}_0$.
    4.  Interpolate value at $\mathbf{X}_0$ and assign to $\mathbf{X}$.
*   **Interpolation:** Linear interpolation (`LinInterp`) is used for the scalar field to avoid oscillations/overshoots inherent in higher-order splines, which could cause instability in density values.

### 5.2 Linear Solvers (Diffusion & Projection)
The diffusion and projection steps require solving a Poisson equation (a sparse linear system). The choice of solver depends on boundary conditions:

| Boundary Type | Solver Method | Complexity | Notes |
| :--- | :--- | :--- | :--- |
| **Periodic** | Fast Fourier Transform (FFT) | O(N log N) | Exact solution. Transforms grid to frequency domain, solves algebraically, transforms back. |
| **Fixed** | Multi-grid / Relaxation (FISHPAK) | O(N) (theoretical) | Iterative solver. FISHPAK was used in the original implementation. |

### 5.3 Pseudocode for Core Routines
Based on Section 3.2 of the paper:

**Transport Routine (Advection):**
```c
// Transports scalar field S0 to S1 using velocity field U
Transport ( S1, S0, U, dt ) {
    for each cell (i, j, k) {
        // Calculate center of cell
        X = O + (i+0.5, j+0.5, k+0.5) * D;
        
        // Trace particle backward in time
        TraceParticle ( X, U, -dt, X0 );
        
        // Interpolate value from starting point
        S1[i,j,k] = LinInterp ( X0, S0 );
    }
}
```

**Diffusion/Projection Routine (Implicit):**
```c
// Solves (I - nu*dt*Laplacian) w3 = w2
Diffuse ( w3, w2, visc, dt ) {
    // Construct sparse linear system Ax = b
    // A = (1 + dt*visc*Laplacian_matrix)
    // b = w2
    SolveLin ( A, b, w3 );
}
```

###6. Scalar Field Simulation (Density/Texture)
The paper distinguishes between Velocity (incompressible, vector field) and Scalars (density, temperature, texture coordinates).

Equation:
da/dt = -u . grad a + kappa_a * Laplacian a - alpha_a * a + S_a
 
Differences from Velocity:
*   1.  No Projection: Scalars do not need to be divergence-free.
*   2.  Dissipation: Includes a term -alpha_a \* a to simulate the substance fading away (e.g., smoke dissipating).
*   3.  Source Term: Sa allows injecting new material (e.g., smoke coming from a chimney).

###7. Visual Effects & "Tricks"
To achieve visually pleasing results suitable for animation, the author employs specific techniques:
*   **Multiple Texture Maps:** Uses three sets of texture coordinates that are advected. They are periodically reset to their initial values to prevent the texture from tearing or becoming too chaotic. The final color is a superposition of these maps.
*   **Fractal Noise:** A "noise" term proportional to the amount of density is added to the visualization to simulate billowing and turbulence not explicitly resolved by the grid.
*   **Grid Resolution:** Animations were generated using grids ranging from 16^3 to 30^3 voxels.
*   **Rendering:** Volume rendering is used (ray marching through the density grid).

###8. Limitations & Future Work
*   **Surface Tension/Free Boundaries:** The solver assumes the fluid fills the entire grid. It does not handle free surfaces (like a wave crashing or water splashing) where the boundary between fluid and air moves dynamically.
*   **Physical Accuracy:** The method has high numerical dissipation. Vortices die out faster than they would in reality. It is not suitable for scientific engineering simulations.
*   **Boundary Complexity:** While it handles fixed boundaries, complex objects inside the flow require sophisticated multi-grid solvers or relaxation routines (like FISHPAK) to handle the non-periodic boundaries efficiently.

###9. Glossary of Terms
*   **Incompressible:** Density is constant; volume of fluid parcels does not change.
*   **Divergence-Free:** The net flow into any region equals the net flow out. Mathematically div u = 0.
*   **Advection:** The transport of a quantity (like velocity or density) by the bulk motion of the fluid.
*   **Diffusion:** The spreading of a quantity due to viscosity (for velocity) or molecular mixing (for density).
*   **Projection:** The mathematical step of removing the divergent part of a vector field to enforce mass conservation.
*   **Method of Characteristics:** A technique to solve advection equations by tracking particles backward along flow lines.

###10. References for Further Reading
*   **Chorin & Marsden:** Standard text on fluid mechanics derivation.
*   **Foster & Metaxas:** The explicit solver this paper improves upon ("Realistic Animation of Liquids").
*   **Hackbusch:** Text on Multi-grid methods.
*   **Max et al.:** Early work on advecting cloud textures.