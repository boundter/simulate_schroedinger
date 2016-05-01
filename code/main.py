#!/usr/bin/env python

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import scipy.constants as con

#########
# Input #
#########

DeltaX = 0.001 # Spatial resolution
EndValue = 10 # Boundaries for the interval
ScalingFactor = 1 # For the potential
# Precision for the transition from left to right solution
TransitionPrecision = 10**(-3)
InitialEnergy = 0.1
FinalEnergy = 10
EnergyStepSize = 0.1
xMatch = 0

########
# Code #
########

# Returns the number of data points in the interval [-EndValue, EndValue] with
# a resolution DeltaX. EndValue should be a multiple of DeltaX to get good
# solutions.
#def GetLengthOfArray():  
#    return int(2*EndValue/DeltaX)

# Returns the double well potential V(x)=A/2*(1-x**2)**2 at point x
def GetPotential(x):
    #return ScalingFactor/2*(1-x**2)**2
    return x**2

# Returns the double well potential V(x)=A/2*(1-x**2)**2 for all data points
# in the interval [-EndValue, EndValue] with a resolution of DeltaX.
#def GetPotentialArray():
#    x = np.arange(-EndValue, EndValue + DeltaX, DeltaX)
#    Potential = GetPotential(x)
#    return Potential

# Returns the function f(x) from Psi''(x) = f(x)*Psi(x) by breaking the second
# order derivative in a coupled system of first order derivatives 
# Psi'(x) = Phi(x) and Phi'(x) = f(x)*Psi(x)
def SecondSpatialDerivative(x, FunctionValues, Energy):
    Psi, Phi = FunctionValues
    f = (GetPotential(x) - Energy)
    return [Phi, f*Psi]

# Integrates the stationary Schroedinger equation and returns Psi and Phi at
# the match point. Beware, that Phi_+ is the negative derivative of Psi at
# the match point.
def IntegrateStationarySchroedinger(Energy, x0):
    integrator = ode(SecondSpatialDerivative).set_integrator('dopri5')
    Psi = np.exp(-np.sqrt(Energy)*np.absolute(x0))
    if x0 < 0:
        assert xMatch > x0, "Initial value is smaller than match point"
        Phi = np.sqrt(Energy)*Psi
        dx = DeltaX
    else:
        assert xMatch < x0, "Initial value is bigger than match point" 
        Phi = -np.sqrt(Energy)*Psi
        dx = -DeltaX
    InitialValues = [Psi, Phi]
    integrator.set_initial_value(InitialValues, x0).set_f_params(Energy)
    while integrator.successful() :
        integrator.integrate(integrator.t + dx)
        #DataList.append(integrator.integrate(integrator.t + dx))
        #XList.append(integrator.t)
        if dx > 0 and integrator.t >= xMatch:
            return integrator.y
        elif dx < 0 and integrator.t <= xMatch:
            return integrator.y

# Returns W(E)=Psi'_+(XMatch)Psi_-(XMatch) - Psi_-'(XMatch)Psi_+(XMatch).
# The root of W(E) equals an eigenvalue of the system.
def GetW(Energy):
    PsiPlus = IntegrateStationarySchroedinger(Energy, EndValue)
    PsiMinus = IntegrateStationarySchroedinger(Energy, -EndValue)
    return (PsiPlus[1]*PsiMinus[0] - PsiMinus[1]*PsiPlus[0])

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

# Bisection to find the first four roots of W(E)
def Find4RootsW():
    Accuracy = 10**(-5)
    LastValue = [InitialEnergy, GetW(InitialEnergy)]
    Roots = []
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
        if len(Roots) == 4:
            return Roots
    print(Roots)
    return "Not enough roots found. Enlarge Energy-width"

print(Find4RootsW())
#DataList = []
#XList = []
##xMatch = -1
#TestEnergy = 3
#IntegrateStationarySchroedinger(TestEnergy, EndValue)
#PsiList = [X[0] for X in DataList]
#DataList = []
#xMatch = 1
#XList1 = XList
#XList = []
#IntegrateStationarySchroedinger(TestEnergy, -EndValue)
#PsiList1 = [X[0] for X in DataList]
#plt.plot(XList1, PsiList)
#plt.plot(XList, PsiList1)
#plt.show()
