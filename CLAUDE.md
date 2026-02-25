
This project implements navier-stokes fluid simulation in 3D based on Jos Stam's paper titled "Real-Time Fluid Dynamics for Games" with a fixed boundary condition instead of periodic/wrap-around boundary. 

The main simulation happpens inside nsStep, which consist of nsAddForce, nsDiffuse, nsProject, nsAdvect, and nsBound.