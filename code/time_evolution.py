#!/usr/bin/env python

import eigenvalues
import csv
import numpy as np

#########
# Input #
#########

dT = 0.01

DataFile = eigenvalues.DataFileName
ScalingFactor = eigenvalues.ScalingFactor
EndValue = eigenvalues.EndValue

########
# Code #
########

# Reads the eigenfunctions, that have been found from the data-file
def GetEigenfunctions():
  Eigenfunctions = [[], [], [], [], []]
  with open(DataFile, "r") as CsvFile:
    CsvReader = csv.reader(CsvFile, delimiter = " ")
    for Row in CsvReader:
      for i in range(0, len(Row)):
        Eigenfunctions[i].append(float(Row[i]))
  return Eigenfunctions

# Generates a numpy array, that holds the potetnial for every point in the 
# interval [BeginInterval, EndInterval] with resolution dX
def GetPotentialArray(BeginInterval, EndInterval, dX):
  PotentialArray = np.arange(BeginInterval, EndInterval + 0.5*dX, dX)
  PotentialArray = eigenvalues.GetPotential(PotentialArray)
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