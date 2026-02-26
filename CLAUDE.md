
# Overview
Navier-Stokes Fluid Simulation based on Jos Stam's Real-Time Fluid Dynamics for Games

**Based on Real-Time Fluid Dynamics for Games**
*   **Author:** Jos Stam  
*   **Source:** Proceedings of SIGGRAPH Conference

The paper presents a simple and rapid implementation of a fluid dynamics solver specifically designed for game engines. The primary goal is to create realistic fluid-like effects (such as swirling smoke) that enhance visual immersion without the heavy computational cost of strict physical accuracy.

**Key Characteristics:**
*   **Visual Quality over Accuracy:** Unlike solvers for engineering (e.g., aerodynamics), this solver prioritizes "convincing" visuals.
*   **Stability:** The algorithms are designed to be stable, allowing for arbitrary time steps without the simulation "blowing up."
*   **Simplicity:** The code is concise (approx. 100 lines of C code) and easy to integrate.
*   **Real-time Performance:** It runs in real-time on standard PC hardware for reasonable grid sizes (2D and 3D).

---

# The Physics of Fluids

The simulation is based on the **Navier-Stokes equations**, which describe the motion of fluid substances. The paper focuses on two main equations: one for the evolution of velocity and one for the evolution of density (or a scalar quantity like smoke concentration).

## The Navier-Stokes Equations

Figure 1 in the paper depicts these equations in compact vector notation:

### 1. The Velocity Equation (Momentum)
Describes how the velocity changes over time due to advection, diffusion, and external forces.
du / dt  =  -(u . grad)u  +  nu * grad^2(u)  +  f
*   **u:** Velocity vector field
*   **t:** Time
*   **grad:** Gradient operator
*   **nu:** Viscosity coefficient
*   **f:** External forces (e.g., gravity, user input)

### 2. The Density Equation
Describes how the density (or smoke) moves and spreads.
drho / dt  =  -(u . grad)rho  +  kappa * grad^2(rho)  +  S
*   **rho:** Density scalar field
*   **kappa:** Diffusion coefficient
*   **S:** Source of density (e.g., a smoke generator)

---

# How the Simulation Works

The solver treats the fluid domain as a grid of cells. It updates the state of the grid in discrete time steps. The process is divided into two main solvers: one for density and one for velocity.

## 1. The Computational Grid
The fluid is modeled on a square grid (2D) or cube (3D).
*   **Size:** An N x N grid is used, surrounded by an extra layer of cells to handle boundary conditions.
*   **Storage:** Arrays are used to store density and velocity components (u, v, and dens).
*   **Indexing:** A 2D grid is flattened into 1D arrays for efficiency using a macro: IX(i, j) = (i) + (N+2)*(j).

## 2. The Density Solver
The density solver moves density through a fixed velocity field. It resolves the three terms of the density equation in reverse order for stability:

1.  **Add Sources:**
    *   Injects density into the grid based on user input or game events (e.g., a cigarette tip).
    *   Mathematically adds the source term $S$ to the current density.

2.  **Diffuse:**
    *   Allows density to spread from a cell to its neighbors.
    *   **Mathematical Challenge:** A naive forward-time implementation is unstable (oscillates and diverges).
    *   **Solution:** The paper uses a **backward-time** method solved via **Gauss-Seidel relaxation**. This ensures stability regardless of the diffusion rate.
    *   *Equation logic:* The density at a cell is updated based on the average of its neighbors.

3.  **Move (Advect):**
    *   Density is moved along the velocity field.
    **Technique:** Instead of moving fluid forward (which creates holes), the solver uses a Semi-Lagrangian approach. It traces the path of a fluid particle backwards in time from the center of the current grid cell to find where it originated.
	*   It then interpolates the density from the surrounding grid cells at that origin point and assigns that value to the current cell.

## 3. The Velocity Solver
The velocity solver updates the movement of the fluid. It follows a similar structure to the density solver but includes a critical final step for mass conservation.

1.  **Add Forces:** Applies external forces (like gravity or user mouse movement) to the velocity field.
2.  **Diffuse:** Spreads the velocity (viscosity) using the same stable Gauss-Seidel method used for density.
3.  **Advect:** Moves the velocity field along itself (self-advection).
4.  **Project (Mass Conservation):**
    *   This is the crucial step for fluid realism. It ensures the velocity field is **divergence-free** ($\nabla \cdot \mathbf{u} = 0$).
    *   **Hodge Decomposition:** The solver treats the velocity field as the sum of an incompressible field and a gradient field. It computes the gradient field (pressure) and subtracts it from the velocity.
    *   **Implementation:** This requires solving a **Poisson equation**. The paper uses Gauss-Seidel relaxation again to solve this linear system efficiently.
	*   This step ensures the fluid is incompressible (mass-conserving).
    *   It forces the divergence of the velocity field to zero.
    *   *Method:* It uses *Hodge Decomposition*. The velocity field is treated as the sum of a mass-conserving field and a gradient field. The gradient field (pressure) is computed by solving a Poisson equation (using Gauss-Seidel relaxation) and then subtracted from the velocity.

---

# Mathematical Details (Discretized)
## Advection (Backtracing)
To move density d along velocity u, v for time dt:

*   1. Calculate the position x, y where the particle came from:
text
```
x = i - dt * u[i, j]
y = j - dt * v[i, j]
```
*   2. Clamp x and y to the grid boundaries.
*   3. Interpolate the density from the four neighbors of (x, y) in the previous time step.

## Diffusion (Gauss-Seidel)
To solve for density x given previous density x0, diffusion rate diff, and time dt:
*   The equation relates a cell to its neighbors.
*   The solver iterates through the grid multiple times (e.g., 20 iterations).
*   In each iteration, the new value of a cell is calculated based on the average of its neighbors plus the source term:
```
x[i, j] = (x0[i, j] + a * (sum of neighbors)) / (1 + 4 * a)
```
(Where a = dt * diff * N * N)

## Projection (Poisson Solver)
To enforce mass conservation:

*   1. Compute the divergence div of the current velocity field.
```
div[i, j] = -0.5 * h * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1])
```
*   2. Solve for the pressure field p such that grad^2(p) = div.
*   3. Subtract the gradient of p from the velocity field to get the corrected, divergence-free velocity:
```
u[i, j] = u[i, j] - 0.5 * (p[i+1, j] - p[i-1, j]) / h
v[i, j] = v[i, j] - 0.5 * (p[i, j+1] - p[i, j-1]) / h
```

---

# Code Implementation Highlights
The paper provides complete C code. Key functions include:

*   ``add_source``: Adds density or force to the grid.
*   ``diffuse``: Uses Gauss-Seidel relaxation to handle diffusion stably.
*   ``advect``: Uses backtracing to move quantities along the velocity field.
*   ``project``: Solves the Poisson equation to remove divergence.
*   ``set_bnd``: Handles boundary conditions (e.g., ensuring velocity is zero at walls).

## Boundary Handling
The ``set_bnd`` function ensures no flow exits the box.

*    For vertical walls: The horizontal component of velocity is set to zero.
*    For density: Continuity is assumed.

---

# Extensions and Future Work
*   **3D:** Extending to 3D is straightforward; it requires adding a z component to velocity and an additional loop in the routines.
*   **Internal Boundaries:** To simulate flow around objects (like characters), a Boolean grid can mark occupied cells. The set_bnd routine can be modified to reflect flow off these internal surfaces.
*   **Fixed Point Arithmetic:** The solver can run on devices without floating-point support (like PDAs) by replacing math operations with fixed-point macros.
*   **Vorticity Confinement:** To improve visual quality and reduce "numerical dissipation" (where fluids dampen too fast), a force can be added to encourage small-scale vorticity.

---

# Results
The paper demonstrates that this approach allows for real-time fluid simulations in games. Visual results include:

*   Swirling smoke trails.
*   2D and 3D smoke simulations.
*   Integration into commercial software (MAYA Fluid Effects).
*   Comparisons to standard fluid motion books show favorable visual matches.