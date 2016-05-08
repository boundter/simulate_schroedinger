#!/usr/bin/env python

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
NumberRoots = 4

#Task2
dT = 0.001
TStart = 0.
TEnd = 10.
TStep = float(int(1/30/dT))

####################
# Functions Task 1 #
####################

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
    while np.absolute(UpperBound) - np.absolute(LowerBound) >BisectionAccuracy:
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
        if len(Roots) == NumberRoots:
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

####################
# Functions Task 2 #
####################

# Generates a numpy array, that holds the potetnial for every point in the 
# interval [BeginInterval, EndInterval] with resolution dX
def GetPotentialArray(BeginInterval, EndInterval, dX):
  PotentialArray = np.arange(BeginInterval, EndInterval + 0.5*dX, dX)
  PotentialArray = GetPotential(PotentialArray)
  return PotentialArray

# Function to test, if (1 + 1/2*i*H*dT)Psi^{n+1} = (1 - 1/2*i*H*dT)Psi^{n}
def TestFunction(Hamiltonian, F, Psi):
    H = Hamiltonian
    H = [-Element for Element in H]
    FLeft = GetRightSide(dT, H, Psi)
    print((FLeft - F)[0], (FLeft - F)[10000], (FLeft - F)[-1])

# Generates the Hamiltonian, which is a 3-diag matrix as three Vectors
def GetHamiltonian(dX, PotentialArray):
    A = np.zeros((len(PotentialArray), ), dtype = np.complex_)
    A.fill(1/(dX*dX))
    B = -2/(dX*dX) + PotentialArray
    B = B.astype(np.complex_)
    return [A, B, A]

# Solves the linear system given by a 3-diag matrix (given as 3 vecotrs A B, C)
# and the left side as F
def SolveThreeDiag(A, B, C, F):
    Alpha = np.zeros((len(F), ), dtype = np.complex_)
    Alpha[0] = F[0]/B[0]
    Beta = np.zeros((len(F), ), dtype = np.complex_)
    Beta[0] = -C[0]/B[0]
    for i in range(1, len(F)):
        Alpha[i] = (F[i] - A[i]*Alpha[i-1])/(B[i] + A[i]*Beta[i-1])
        Beta[i] = -C[i]/(B[i] + A[i]*Beta[i-1])
    x = np.zeros((len(F),), dtype = np.complex_)
    x[-1] = Alpha[-1]
    for i in range(2, len(F)+1):
        x[-i] = Alpha[-i] + Beta[-i]*x[-i+1]
    return x

# Calculates (1 - 1/2*i*H*dT)Psi^{n}
def GetRightSide(dT, Hamiltonian, InitialCondition):
    H = [-0.5j*dT*Vec for Vec in Hamiltonian]
    A = H[0]
    B = 1 + H[1]
    C = H[2]
    F = np.zeros((len(InitialCondition),), dtype = np.complex_)
    F[0] = B[0]*InitialCondition[0] + C[0]*InitialCondition[1]
    for i in range(1, len(InitialCondition) - 1):
        F[i] = A[i]*InitialCondition[i-1] + B[i]*InitialCondition[i] + \
               C[i]*InitialCondition[i+1]
    F[-1] = A[-1]*InitialCondition[-2] + B[-1]*InitialCondition[-1]
    return F

# Calculates Psi^{n+1} from (1 + 1/2*i*H*dT)Psi^{n+1} = (1 - 1/2*i*H*dT)Psi^{n}
def GetPsi(dT, Hamiltonian, F):
    H = [0.5j*dT*Vec for Vec in Hamiltonian]
    A = H[0]
    B = 1 + H[1]
    C = H[2]
    Psi = SolveThreeDiag(A, B, C, F)
    #TestFunction(Hamiltonian, F, Psi)
    return Psi

# Integrates the Schroedinger equation one step in time using the 
# Crank-Nicholson-Scheme
def IntegrateSchroedinger(dX, InitialCondition, PotentialArray):
  Hamiltonian = GetHamiltonian(dX, PotentialArray)
  F = GetRightSide(dT, Hamiltonian, InitialCondition)
  Psi = GetPsi(dT, Hamiltonian, F)
  return Psi


# Animation
def init():
    PotentialLine.set_data([], [])
    PsiLine.set_data([], [])
    time_text.set_text('')
    return (PotentialLine, PsiLine)

# Animation
def animate(j):
    x = np.real(PsiEvolution[0])
    y = np.absolute(PsiEvolution[j+1])**2
    PsiLine.set_data(x, y)
    PotentialLine.set_data(X, GetPotential(X)/10)
    time_text.set_text('time = %.2f' % (j*dT*TStep))
    return(PotentialLine, PsiLine, time_text)

#Animnation
def PlotAnimation(Iterator):   
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True, \
                                   frames = len(PsiEvolution) - 1)
    anim.save('plots/animation_%i.mp4' % (Iterator - NumberRoots - 1), \
              extra_args=['-vcodec', 'libx264'], fps = 1/(dT*TStep))

##########
# Task 1 #
##########
# Find the 4 eigenvalues
Results = Find4RootsW()
if type(Results) is str:
 print(Results)
 exit()
print(Results)
# Plot the eigenfnctions
Eigenfunctions = PlotEigenfunctions(Results)
PlotScheme(Results, Eigenfunctions)

##########
# Task 2 #
##########
#Calculate the evolution in time of the 4 eigenfuntions
Eigenfunctions = Eigenfunctions.astype(np.complex_)
PotentialArray = GetPotentialArray(-Boundary, Boundary, DeltaX)
Iterator = NumberRoots + 2
# The loop calculates psi^{n+1} for every time step and saves every TStep 
# array for the animation
for Element in Eigenfunctions[1:]:
    print("==Starting next Eigenfunction==")
    Psi = Element
    PsiEvolution = [Eigenfunctions[0], Psi]
    Counter = 1
    for i in np.arange(TStart, TEnd + dT, dT):
        Psi = IntegrateSchroedinger(DeltaX, Psi, PotentialArray)
        if i%1 < 0.5*dT:  print("t = %f" %i)
        # Decide if the current values should be saved for the animation
        if Counter == TStep:
            PsiEvolution.append(Psi)
            Counter = -1
        Counter += 1
    # Checks, if the solution is still normalized
    print("Probability before = %f" % \
          CheckProbability(PsiEvolution[1], -Boundary, Boundary, -Boundary, \
                           DeltaX))
    print("Probability after = %f" % \
          CheckProbability(PsiEvolution[-1], -Boundary, Boundary, -Boundary, \
                           DeltaX))
    # The rest is just the animation
    fig = plt.figure(Iterator)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5),\
                         ylim=(0, 1))
    PsiLine, = ax.plot([], [], lw=2)
    PotentialLine,  = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    X = np.linspace(-2.6, 2.6, 1000) 
    PlotAnimation(Iterator)
    Iterator += 1