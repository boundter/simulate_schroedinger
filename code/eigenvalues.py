#!/usr/bin/env python

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import math

#########
# Input #
#########

DeltaX = 0.001 # Spatial resolution
EndValue = 10 # Boundaries for the interval
ScalingFactor = 5. # For the potential
# Precision for the transition from left to right solution
TransitionPrecision = 10**(-3)
InitialEnergy = 0.01 # Initial energy to be tested
FinalEnergy = 30 # Final energy to be tested
EnergyStepSize = 0.1 # Step size for the energy
XSym = 0 # Point where the solutions should match (ideally the symmetry point)
ReverseList = [2, 4] # List of antisymmetric solutions
# Scaling factor for the initial conditions, to offset overflow
ScalingFactorPsi = 10**(-150)
EigenvaluesEigenfunctionX = (-6., 6.) # X-Range to be plotted for the scheme
EigenvaluesEigenfunctionY = (0., 13.) # Y-Range to be plotted for the scheme

########
# Code #
########

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
    integrator.set_initial_value(InitialValues, X0).set_f_params(Energy)
    while integrator.successful() :
        integrator.integrate(integrator.t + dX)
        if plot:
            PsiList.append(integrator.y[0])
            XList.append(integrator.t)
        if dX > 0 and integrator.t >= XMatch:
            return integrator.y
        elif dX < 0 and integrator.t <= XMatch:
            return integrator.y

# Returns W(E)=Psi'_+(XMatch)Psi_-(XMatch) - Psi_-'(XMatch)Psi_+(XMatch).
# The root of W(E) equals an eigenvalue of the system.
def GetW(Energy):
    PsiPlus = IntegrateStationarySchroedinger(Energy, EndValue)
    PsiMinus = IntegrateStationarySchroedinger(Energy, -EndValue)
    return (PsiPlus[1]*PsiMinus[0] - PsiMinus[1]*PsiPlus[0])

# Bisection to find the roots of W
def BisectionW(LowerBound, UpperBound):
    Accuracy = 10**(-6)
    NewBound = (LowerBound + UpperBound)/2
    NewW = GetW(NewBound)
    while np.absolute(UpperBound) - np.absolute(LowerBound) > Accuracy:
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
    EnergyList, WList, Roots = ([] for i in range(3))
    Accuracy = 10**(-5)
    LastValue = [InitialEnergy, GetW(InitialEnergy)]
    EnergyList.append(InitialEnergy)
    WList.append(LastValue[1])
    for Energy in np.arange(InitialEnergy + EnergyStepSize, \
                            FinalEnergy + EnergyStepSize, EnergyStepSize):
        print("Energy = %f" % Energy)
        NextValue = [Energy, GetW(Energy)]
        if np.absolute(NextValue[1]) < Accuracy:
            Roots.append(NextValue[0])
        elif NextValue[1] < 0 and LastValue[1] > 0:
            Roots.append(BisectionW(NextValue[0], LastValue[0]))
        elif NextValue[1] > 0 and LastValue[1] < 0:
            Roots.append(BisectionW(LastValue[0], NextValue[0]))
        LastValue = NextValue
        EnergyList.append(LastValue[0])
        WList.append(LastValue[1])
        if len(Roots) == 4:
            return Roots
    return "Not enough roots found. Enlarge Energy-width"

# Normalize Psi by calculating the integral of |Psi|^2  over the interval and 
# then diving by the result.
def NormalizePsi(XArray, PsiArray):
    NormalizeParameter = 0
    for i in range(0, len(XArray)):
        NormalizeParameter += DeltaX*PsiArray[i]*PsiArray[i]
    print(NormalizeParameter)
    PsiArray = PsiArray/np.sqrt(NormalizeParameter)
    return PsiArray

# Plots the eigenfunctions for given roots
def PlotEigenfunctions(Roots):
    Iterator = 1
    PsiSolutions = []
    for Element in Roots:
        plt.figure(Iterator)
        plt.xlabel("x")
        plt.ylabel(r"$\Psi$(x)")
        PsiMinusList, XMinusList, PsiPlusList, XPlusList = \
                                                          ([] for i in range(4))
        IntegrateStationarySchroedinger(Element, -EndValue, 0, True, \
                                        PsiMinusList, XMinusList)
        IntegrateStationarySchroedinger(Element, EndValue, 0, True, \
                                        PsiPlusList, XPlusList)
        if Iterator in ReverseList:
           PsiMinusList = [-i for i in PsiMinusList]
        XArray = np.asarray(XMinusList[0:-1] + XPlusList[::-1])
        PsiArray = np.asarray(PsiMinusList[0:-1] + PsiPlusList[::-1])
        PsiArray = NormalizePsi(XArray, PsiArray)
        PsiSolutions.append([XArray, PsiArray])
        plt.plot(XArray, PsiArray)
        plt.savefig("eigenfunction_%i.eps" % Iterator, format = "eps", \
                    dpi = 1000)
        Iterator += 1
    return PsiSolutions

# Plots a scheme, where the eigenfunctions are offset by their eigenvalue and
# also plots the potetnial, to get an overview over the solutions in context 
# of their energys.
def PlotScheme(Roots, PsiSolutions):
    plt.figure(len(Roots) + 1)
    plt.title("ScalingFactor = %f" % ScalingFactor)
    plt.xlabel("x")
    plt.ylabel("Energy")
    XCoordinates = np.linspace(EigenvaluesEigenfunctionX[0], \
                               EigenvaluesEigenfunctionX[1], 1000)
    plt.plot(XCoordinates, GetPotential(XCoordinates))
    Iterator = 0
    for Element in Roots:
        plt.axhline(y = Element, xmin = EigenvaluesEigenfunctionX[0],\
         xmax = EigenvaluesEigenfunctionX[1])
        plt.plot(PsiSolutions[Iterator][0], PsiSolutions[Iterator][1] + Element)
        Iterator += 1
    plt.ylim(EigenvaluesEigenfunctionY)
    plt.xlim(EigenvaluesEigenfunctionX)
    plt.savefig("eigenfunction_eigenvalue.eps", format = "eps", dpi = 1000)

Results = Find4RootsW()
if type(Results) is str:
 print(Results)
 exit()
print(Results)
os.chdir("../plots")
PsiSolutions = PlotEigenfunctions(Results)
PlotScheme(Results, PsiSolutions)