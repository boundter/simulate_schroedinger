#!/usr/bin/env python

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import math

#########
# Input #
#########

#Task 1
DeltaX = 0.001 # Spatial resolution
Boundary = 10 # Boundaries for the interval
ScalingFactor = 5. # For the potential
# Precision for the transition from left to right solution
TransitionPrecision = 10**(-3)
InitialEnergy = 0.01 # Initial energy to be tested
FinalEnergy = 15 # Final energy to be tested
EnergyStepSize = 0.1 # Step size for the energy
XSym = 0 # Point where the solutions should match (ideally the symmetry point)
ReverseList = [2, 4] # List of antisymmetric solutions
# Scaling factor for the initial conditions, to offset overflow
ScalingFactorPsi = 10**(-150)
EigenvaluesEigenfunctionX = (-6., 6.) # X-Range to be plotted for the scheme
EigenvaluesEigenfunctionY = (0., 13.) # Y-Range to be plotted for the scheme
BisectionAccuracy = 10**(-6)

#Task2
dT = 0.01

##########
# Task 1 #
##########

# Returns the double well potential V(x)=A/2*(1-x**2)**2 at point x
def GetPotential(X):
    return ScalingFactor/2*(1-X**2)**2

# Returns the function f(x) from Psi''(x) = f(x)*Psi(x) by breaking the second
# order derivative in a coupled system of first order derivatives 
# Psi'(x) = Phi(x) and Phi'(x) = f(x)*Psi(x)
def SecondSpatialDerivative(X, FunctionValues, Energy):
    Psi, Phi = FunctionValues
    f = (GetPotential(X) - Energy)
    return [Phi, f*Psi]

# Integrates the stationary Schroedinger equation and returns Psi and Phi at
# the match point. Beware, that Phi_+ is the negative derivative of Psi at
# the match point.
def IntegrateStationarySchroedinger(Energy, X0, XMatch = XSym, plot = False, \
                                    PsiList = [], XList = []):
    integrator = ode(SecondSpatialDerivative).set_integrator('dopri5')
    Psi = ScalingFactorPsi*np.exp(-np.sqrt(Energy)*np.absolute(X0))
    if X0 < 0:
        assert XMatch > X0, "Initial value is smaller than match point"
        Phi = np.sqrt(Energy)*Psi
        dX = DeltaX
    else:
        assert XMatch < X0, "Initial value is bigger than match point" 
        Phi = -np.sqrt(Energy)*Psi
        dX = -DeltaX
    InitialValues = [Psi, Phi]
    if plot:
        PsiList.append(Psi)
        XList.append(X0)
    integrator.set_initial_value(InitialValues, X0).set_f_params(Energy)
    while integrator.successful() :
        integrator.integrate(integrator.t + dX)
        if plot:
            PsiList.append(integrator.y[0])
            XList.append(integrator.t)
        if dX > 0 and integrator.t + dX > XMatch:
            return integrator.y
        elif dX < 0 and integrator.t + dX < XMatch:
            return integrator.y

# Returns W(E)=Psi'_+(XMatch)Psi_-(XMatch) - Psi_-'(XMatch)Psi_+(XMatch).
# The root of W(E) equals an eigenvalue of the system.
def GetW(Energy):
    PsiPlus = IntegrateStationarySchroedinger(Energy, Boundary)
    PsiMinus = IntegrateStationarySchroedinger(Energy, -Boundary)
    return (PsiPlus[1]*PsiMinus[0] - PsiMinus[1]*PsiPlus[0])

# Bisection to find the roots of W
def BisectionW(LowerBound, UpperBound):
    NewBound = (LowerBound + UpperBound)/2
    NewW = GetW(NewBound)
    while np.absolute(UpperBound) - np.absolute(LowerBound) > BisectionAccuracy:
        if NewW < 0:
            LowerBound = NewBound
        else:
            UpperBound = NewBound
        NewBound = (LowerBound + UpperBound)/2
        NewW = GetW(NewBound)
    return NewBound

# Calculates W(E) at multiple points in the given energy-range and the looks
# for a change in the sign. If it finds one it calls BisectionW to finnd the
# root.
def Find4RootsW():
    Roots = []
    LastValue = [InitialEnergy, GetW(InitialEnergy)]
    for Energy in np.arange(InitialEnergy + EnergyStepSize, \
                            FinalEnergy + EnergyStepSize, EnergyStepSize):
        print("Energy = %f" % Energy)
        NextValue = [Energy, GetW(Energy)]
        if NextValue[1] < 0 and LastValue[1] > 0:
            Roots.append(BisectionW(NextValue[0], LastValue[0]))
        elif NextValue[1] > 0 and LastValue[1] < 0:
            Roots.append(BisectionW(LastValue[0], NextValue[0]))
        LastValue = NextValue
        if len(Roots) == 4:
            return Roots
    return "Not enough roots found. Enlarge Energy-width"

# Checks the integral of the probability |Psi|^2 in the interval [BeginIntegral,
# EndIntegral] with stepsize dX. Psi is given on the interval 
# [BeginInterval, BeginInterval + dX*len(Psi) - 1]
def CheckProbability(Psi, BeginIntegral, EndIntegral, BeginInterval, dX):
  # First we need to find which elements of Psi we need
  FirstElement = int((BeginIntegral - BeginInterval)/dX)
  LastElement = int((EndIntegral - BeginIntegral)/dX + 1)
  Probability = 0
  for i in range(FirstElement, LastElement - 1):
    Probability += dX*(np.absolute(Psi[i+1])**2 + np.absolute(Psi[i])**2)/2
  return Probability

# Normalize Psi by calculating the integral of |Psi|^2  over the interval and 
# then diving by the result.
def NormalizePsi(Psi):
    NormalizeParameter = CheckProbability(Psi, -Boundary, Boundary, -Boundary, \
                                          DeltaX)
    Psi = Psi/np.sqrt(NormalizeParameter)
    print(NormalizeParameter)
    return Psi

# Plots the eigenfunctions for given roots
def PlotEigenfunctions(Roots):
    Iterator = 1
    X = np.arange(-Boundary, Boundary + 0.5*DeltaX, DeltaX)
    Psi = [X]
    for Element in Roots:
        plt.figure(Iterator)
        plt.xlabel("x")
        plt.ylabel(r"$\Psi$(x)")
        PsiMinus, XMinus, PsiPlus, XPlus = ([] for i in range(4))
        IntegrateStationarySchroedinger(Element, -Boundary, 0, True, \
                                        PsiMinus, XMinus)
        IntegrateStationarySchroedinger(Element, Boundary, 0, True, \
                                        PsiPlus, XPlus)
        if Iterator in ReverseList:
           PsiMinus = [-i for i in PsiMinus]
        PsiArray = np.asarray(PsiMinus[0:-1] + PsiPlus[::-1])
        PsiArray = NormalizePsi(PsiArray)
        Psi.append(PsiArray)
        plt.plot(X, PsiArray)
        plt.savefig("plots/eigenfunction_%i.eps" % Iterator, format = "eps", \
                    dpi = 1000)
        Iterator += 1
    return np.asarray(Psi)

# Plots a scheme, where the eigenfunctions are offset by their eigenvalue and
# also plots the potetnial, to get an overview over the solutions in context 
# of their energys.
def PlotScheme(Roots, Psi):
    plt.figure(len(Roots) + 1)
    plt.title("ScalingFactor = %f" % ScalingFactor)
    plt.xlabel("x")
    plt.ylabel("Energy")
    XCoordinates = np.linspace(EigenvaluesEigenfunctionX[0], \
                               EigenvaluesEigenfunctionX[1], 1000)
    plt.plot(XCoordinates, GetPotential(XCoordinates))
    Iterator = 1
    for Element in Roots:
        plt.axhline(y = Element, xmin = EigenvaluesEigenfunctionX[0],\
         xmax = EigenvaluesEigenfunctionX[1])
        plt.plot(Psi[0], Psi[Iterator] + Element)
        Iterator += 1
    plt.ylim(EigenvaluesEigenfunctionY)
    plt.xlim(EigenvaluesEigenfunctionX)
    plt.savefig("plots/eigenfunction_eigenvalue.eps", format = "eps", \
                dpi = 1000)

# Uncomment the following part to solve task 1

Results = Find4RootsW()
if type(Results) is str:
 print(Results)
 exit()
print(Results)
PsiSolutions = PlotEigenfunctions(Results)
PlotScheme(Results, PsiSolutions)

##########
# Task 2 #
##########
# Generates a numpy array, that holds the potetnial for every point in the 
# interval [BeginInterval, EndInterval] with resolution dX
def GetPotentialArray(BeginInterval, EndInterval, dX):
  PotentialArray = np.arange(BeginInterval, EndInterval + 0.5*dX, dX)
  PotentialArray = GetPotential(PotentialArray)
  return PotentialArray

# Integrates the Schroedinger equation one step in time using the 
# Crank-Nicholson-Scheme
def IntegrateSchroedinger(dX, InitialCondition, PotentialArray):
  dX2 = dX*dX
  NumberPoints = len(InitialCondition)
  A = -0.25j*dT/dX2
  B = 0.5 + 0.5j*dT/dX2 + 0.25j*dT*PotentialArray
  Alpha = [InitialCondition[0]/B]
  Beta = [-A/B]
  for i in range(1, NumberPoints):
    Alpha.append((InitialCondition[i] - Alpha[i-1]*A)/(A*Beta[i-1] + B[i]))
    Beta.append(-A/(A*Beta[i-1] + B[i]))
  Chi = np.zeros((NumberPoints,), dtype = np.complex_)
  Chi[-1] = Alpha[-1]
  for i in range(1, NumberPoints):
    l = NumberPoints - i - 1
    Chi[l] = Alpha[l] - Beta[l]*Chi[l+1]
  return Chi - InitialCondition

