# Simulate Schroedinger

This is a script to calculate the Eigenvalues and Eigenfunctions of the Schrödinger-Equation.

## Used Tools
- Python
  - numpy
  - matplotlib
  - scipy for integration and curve fitting
  
## Getting Started

The file ```code/main.py``` is outdated, ```code/main2.py``` should be used. It changes the style of programming from fruntional to object-oriented. The plot should be called from the main-directory.

All necessary classes are contained in ```code/wave.py```. The script ```code/main2.py``` then creates all the plots.

## Classes

```SchroedingerWave``` - Typical solution of the Schrödinger Equation with methods to integrate over the wave and plot
```StationarySchroedinger``` - Child of ```SchroedingerWave``` to integrate the stationary solution with Runge-Kutta of 4th order
```GeneralSchroedinger``` - Child of ```SchroedingerWave``` to integrate the time-dependent Schrödinger eequation with the Crank-Nicolson scheme

