#!/usr/bin/env python

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import scipy.constants as con

#########
# Input #
#########

DeltaX = 0.001 # Spatial resolution
EndValue = 2 # Boundaries for the interval
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
#def GetLengthOfArray():  
#    return int(2*EndValue/DeltaX)

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
        xList.append(integrator.t)
        DataList.append(integrator.integrate(integrator.t + dx))
        if dx > 0 and integrator.t > xMatch:
            return integrator.y
        elif dx < 0 and integrator.t < xMatch:
            return integrator.y

# Integrates Psi''(x)=f(x)*Psi(x) from left and right to XMatch and returns
# the value of both functions and their derivative.
#def IntegrateToXMatch(Energy):
    

# Returns W(E)=Psi'_+(XMatch)Psi_-(XMatch) - Psi_-'(XMatch)Psi_+(XMatch).
# The root of W(E) equals an eigenvalue of the system.
#def GetW

integrator = ode(SecondSpatialDerivative).set_integrator('dopri5')
Energy = 2
xMinus = -EndValue
PsiMinus = np.exp(np.sqrt(Energy)*xMinus)
PhiMinus = np.sqrt(Energy)*PsiMinus
InitialValues = [PsiMinus, PhiMinus]
integrator.set_initial_value(InitialValues, xMinus).set_f_params(Energy)
xMatch = EndValue
xList1 = []
DataList1 = []
while integrator.successful() and integrator.t < xMatch:
    xList1.append(integrator.t)
    DataList1.append(integrator.integrate(integrator.t + DeltaX))

xPlus = EndValue
PsiPlus = np.exp(-np.sqrt(Energy)*xPlus)
PhiPlus = -np.sqrt(Energy)*PsiPlus
InitialValues = [PsiPlus, PhiPlus]
integrator.set_initial_value(InitialValues, xPlus).set_f_params(Energy)
OtherxList = []
OtherPsiList = []
while integrator.successful() and integrator.t > -xMatch:
    OtherxList.append(integrator.t)
    OtherPsiList.append(integrator.integrate(integrator.t - DeltaX)[0])
#
#
DataList =[]
xList = []
xMatch = 0
IntegrateStationarySchroedinger(2, 2)
PsiList = [x[0] for x in DataList]
PsiList1 = [x[0] for x in DataList1]
#plt.figure(1)
plt.plot(xList, PsiList)
plt.plot(xList1, PsiList1)
plt.plot(OtherxList, OtherPsiList)
plt.xlim(-2, 2)
plt.show()
#x = np.arange(-1.5, 1.5 + DeltaX, DeltaX)
#plt.plot(x, GetPotential(x))
#plt.show()
