#!/usr/bin/env python

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

#########
# Input #
#########

DeltaX = 0.01 # Spatial resolution
EndValue = 10 # Boundaries for the interval
ScalingFactor = 1 # For the potential
# Precision for the transition from left to right solution
TransitionPrecision = 10**(-5)
InitialEnergy = 0
FinalEnergy = 5

########
# Code #
########

# Returns the number of data points in the interval [-EndValue, EndValue] with
# a resolution DeltaX. EndValue should be a multiple of DeltaX to get good
# solutions.
def GetLengthOfArray():  
    return int(2*EndValue/DeltaX)

# Returns the double well potential V(x)=A/2*(1-x**2)**2 at point x
def GetPotential(x):
    return ScalingFactor/2*(1-x**2)**2

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
    f = (Energy - GetPotential(x))
    return [Phi, f*Psi]

integrator = ode(SecondSpatialDerivative).set_integrator('dopri5')
Energy = 0.25
xMinus = -EndValue
PsiMinus = np.exp(np.sqrt(Energy)*xMinus)
PhiMinus = -np.sqrt(Energy)*PsiMinus
InitialValues = [PsiMinus, PhiMinus]
integrator.set_initial_value(InitialValues, xMinus).set_f_params(Energy)
xMatch = EndValue
xList = []
DataList = []
while integrator.successful() and integrator.t < xMatch:
    xList.append(integrator.t)
    DataList.append(integrator.integrate(integrator.t + DeltaX))

xPlus = EndValue
PsiPlus = np.exp(np.sqrt(Energy)*xPlus)
PhiPlus = np.sqrt(Energy)*PsiPlus
InitialValues = [PsiPlus, PhiPlus]
integrator.set_initial_value(InitialValues, xPlus).set_f_params(Energy)
OtherxList = []
OtherPsiList = []
while integrator.successful() and integrator.t > -xMatch:
    OtherxList.append(integrator.t)
    OtherPsiList.append(integrator.integrate(integrator.t - DeltaX)[0])


PhiList = [x[0] for x in DataList]
plt.figure(1)
plt.plot(xList, PhiList)
plt.plot(OtherxList, OtherPsiList)
plt.show()

#x = np.arange(-1.5, 1.5 + DeltaX, DeltaX)
#plt.plot(x, GetPotential(x))
#plt.show()
