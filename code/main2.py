#!/usr/bin/env python

import numpy as np
import wave as wv
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#########
# Input #
#########

#Task 1
dx = 1e-3 # Spatial step
Boundary = 10 # Solution in the interval [-Boundary, Boundary]
EMax = 15. # Maximum of E 
# These dx will be used for chekcing of Probability conservation of Crank-Nico.
dxList = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]

#Task 2
dt = 1e-3 # Time step for the integration
tEnd = 10 # Stoptime


# Global variables
NumberRoots = 4 # How many roots to search
xPoints = 2*Boundary/dx # #points in the interval [-Boundary, Boundary]
if xPoints%2 == 0: # Make sure we have an uneven number of points, to get 0
  xPoints += 1
fps = 30 # Frames per second for the animation
StepsPerAnimation = int(1/(dt*fps)) # #timesteps per frame

#############
# Functions #
#############

def V(x):
  """
  Potential V(x) for the Schroedinger-equation
  """
  return 0.5*(1-x**2)**2

def Bisection(LowerE, UpperE, Wave):
  """
  Bisection of w(E) to find the eigenvalues
  """
  NewE = (LowerE + UpperE)/2
  Wave.E = NewE
  NewW = Wave.w()
  BisectionAccuracy = 1e-6
  while np.absolute(np.absolute(UpperE) - np.absolute(LowerE)) > BisectionAccuracy:
    if NewW < 0:
        LowerE = NewE
    else:
        UpperE = NewE
    NewE = (LowerE + UpperE)/2
    Wave.E = NewE
    NewW = Wave.w()
  return NewE

def FindEigenvalues(InitialE, EndE, dE):
  """
  Finds the eigenvalues E by using bisection in E. It will return NumberRoots 
  of eigenvalues in a dictionary of the form {'eigenvalue0i':E}
  """
  Eigenvalues = {}
  x = np.linspace(-Boundary, Boundary, xPoints)
  Wave = wv.StationarySchroedinger(x, V, InitialE)
  LastValue = [Wave.E, Wave.w()]
  E = InitialE
  while E < EndE and len(Eigenvalues) < NumberRoots:
    Wave.E = E
    CurrentValue = [E, Wave.w()]
    print("Energy = %3g" %CurrentValue[0])
    if CurrentValue[1] < 0 and LastValue[1] > 0:
      Eigenvalues['eigenvalue%02i' %(len(Eigenvalues) + 1)] = \
      Bisection(CurrentValue[0], LastValue[0], Wave)
    elif CurrentValue[1] > 0 and LastValue[1] < 0:
      Eigenvalues['eigenvalue%02i' %(len(Eigenvalues) + 1)] = \
      Bisection(LastValue[0], CurrentValue[0], Wave)
    E += dE
    LastValue = CurrentValue
  if len(Eigenvalues) < NumberRoots:
    print('Not enough eigenvalues found. Enlarge Energy.')
    exit()
  return Eigenvalues

def Superposition(Key1, Key2, x):
  """
  Takes two eigenfunctions as arguments and integrates the over the time. The
  probability of begin <0 will be ploted.
  """
  print('%s and %s' % (Key1, Key2))
  plt.figure()
  plt.xlabel('t')
  plt.ylabel(r'$\vert \Psi  (x < 0) \vert ^2$')
  Psi = Eigenvalues[Key1][1].Psi - Eigenvalues[Key2][1].Psi
  Psi = wv.GeneralSchroedinger(x, Psi, V)
  Psi.Normalize()
  Time = np.arange(0, tEnd, dt)
  Probability = [Psi.IntegrateAbsolute2(-Boundary, 0)]
  for i in Time:
    Psi.IntegrateTime(dt)
    Probability.append(Psi.IntegrateAbsolute2(-Boundary, 0))
  Probability = Probability[:-1]
  plt.plot(Time, Probability, label = r'$E_%s$ - $E_%s$' %(Key1[-1], Key2[-1]))
  plt.xlim([0., tEnd])
  plt.legend(loc = 3)
  plt.savefig('plots/probability_left%s%s.eps' % (Key1[-1], Key2[-1]), \
              format = 'eps', dpi = 1000)
  plt.close()
  Period = Periodicity(Time, Probability)
  return (Period, Eigenvalues[Key1][0] - Eigenvalues[Key2][0])

def Periodicity(Time, Probability):
  """
  Finds the periodicity of the Probability by looking for the maximum in the 
  range of 5 points
  """
  ProbOrder = sorted(Probability, reverse = True)
  PeriodTime = []
  for element in ProbOrder[:500]:
    Indx = Probability.index(element)
    if Indx < len(Probability) - 1 and \
      Indx > 1 and \
      Probability[Indx] > Probability[Indx - 1] and \
      Probability[Indx] > Probability[Indx + 1] and \
      Probability[Indx] > Probability[Indx - 2] and \
      Probability[Indx] > Probability[Indx + 2]:
      PeriodTime.append(Time[Indx])
  PeriodTime = np.asarray(PeriodTime); PeriodTime.sort()
  print(PeriodTime)
  PeriodTime = PeriodTime[1:] - PeriodTime[:-1]
  PeriodTime = np.mean(PeriodTime)
  return PeriodTime

def FitFunc(DeltaE, a):
  """
  Fit-function for the Periodicity of being left
  """
  return a/DeltaE

###############################################################################
# Animation
def init():
    VLine.set_data([], [])
    PsiLine.set_data([], [])
    time_text.set_text('')
    return (VLine, PsiLine, time_text)

def animate(j, Wave):
    for i in range(0, StepsPerAnimation):
      Wave.IntegrateTime(dt)  
    Wave.IntegrateTime(dt)
    PsiLine.set_data(Wave.x, np.absolute(Wave.Psi)**2)
    VLine.set_data(x, Wave.V)
    time_text.set_text('time = %.2f' % (Wave.t))
    return(VLine, PsiLine, time_text)

def PlotAnimation(Wave, Key):   
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True, \
                                   frames = int(30*tEnd), fargs = (Wave, ))
    anim.save('plots/animation_%s.mp4' % Key, \
              extra_args=['-vcodec', 'libx264'], fps = fps)


##########
# Task 1 #
##########

# Find the eigenvalues and eigenfunctions and plot the eigenfunctions
Eigenvalues = FindEigenvalues(0.01, EMax, 0.1)
print(Eigenvalues)
x = np.linspace(-Boundary, Boundary, xPoints)
for Key in Eigenvalues:
  Wave = wv.StationarySchroedinger(x, V, Eigenvalues[Key])
  Wave.Integrate(Boundary, save = True); Wave.Integrate(-Boundary, save = True)
  Wave.Normalize()
  Wave.FlipAntisymmetric()
  Wave.PlotWave([-6., 6.], [-1., 1.], '%s.eps' %Key, \
                Label = ': E = %.3g' %Eigenvalues[Key])
  Eigenvalues[Key] = [Eigenvalues[Key], Wave]

# Plot a scheme of the eigenfunctions and eigenvalues
plt.figure()
plt.xlabel('x')
plt.ylabel('Energy')
plt.plot(Eigenvalues['eigenvalue01'][1].x, \
         V(Eigenvalues['eigenvalue01'][1].x))
for Key in Eigenvalues:
  plt.axhline(y = Eigenvalues[Key][0], xmin =  -6.,xmax = 6.)
  plt.plot(Eigenvalues[Key][1].x, \
           Eigenvalues[Key][1].Psi + Eigenvalues[Key][0])
plt.xlim([-6., 6.])
plt.ylim([0., 8.5])
plt.savefig('plots/eigenfunctions.eps', format = 'eps', dpi = 1000)
plt.close()

##########
# Task 2 #
##########

# Animate the Eigenvalues to see, if they are stationary and plot them at t0 
# and at tEnd.
for Key in Eigenvalues:
  Psi = wv.GeneralSchroedinger(Eigenvalues[Key][1].x, Eigenvalues[Key][1].Psi,\
                               V)
  Psi.PlotAbsolute2([-6., 6.,], [-0.2, 1.], '%s_t0.eps' % Key)
  fig = plt.figure()
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3),\
                       ylim=(0, 1))
  PsiLine, = ax.plot([], [], lw=2)
  VLine,  = ax.plot([], [], lw=2)
  ax.set_xlabel('x')
  ax.set_ylabel(r'$\vert \Psi (x) \vert ^2$')
  time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
  print('Eigenfunction %s' % Key)
  PlotAnimation(Psi, Key)
  Psi.PlotAbsolute2([-6., 6.,], [-0.2, 1.], '%s_t1.eps' % Key)

# Plot the Probability |Psi(x)|^2 over the whole spatial interval in
# dependence of the time and the stepsize dx for a Gaussian-wave
plt.figure()
plt.xlabel('t')
plt.ylabel(r'$\Delta \vert \Psi \vert ^2$')
for Deltax in dxList:
  x = np.arange(-10., 10., Deltax)
  Gauss = 1/np.sqrt(2*np.pi)*np.exp(-(x-1)**2/2)
  Psi = wv.GeneralSchroedinger(x, Gauss, V)
  Psi.Normalize()
  Time = np.arange(0, tEnd, dt)
  Probability = [1. - Psi.IntegrateAbsolute2(-Boundary, Boundary)]
  for i in Time:
    Psi.IntegrateTime(dt)
    Probability.append(1. - Psi.IntegrateAbsolute2(-Boundary, Boundary))
  Probability = Probability[:-1]
  plt.plot(Time, Probability, label = r'$\Delta$x = %.3f' % Deltax)
plt.xlim([0., tEnd])
plt.legend(loc = 3)
plt.savefig('plots/probability.eps', format = 'eps', dpi = 1000)
plt.close()

# Animate a Gaussian wave over the whole time range
x = np.linspace(-Boundary, Boundary, xPoints)
Gauss = 1/np.sqrt(2*np.pi)*np.exp(-(x-1)**2/2)
Psi = wv.GeneralSchroedinger(x, Gauss, V)
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3),\
                     ylim=(0, 1))
PsiLine, = ax.plot([], [], lw=2)
VLine,  = ax.plot([], [], lw=2)
ax.set_xlabel('x')
ax.set_ylabel(r'$\vert \Psi (x) \vert ^2$')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
print('Gauss')
PlotAnimation(Psi, 'Gauss')

# Plot the Probability of being left of 0 for a superposition of two 
# eigenvalues. The superposition is of the form E1 - E2
Period = []
Energies = []
y = Superposition('eigenvalue02', 'eigenvalue01', x)
Period.append(y[0]); Energies.append(y[1])
y = Superposition('eigenvalue04', 'eigenvalue01', x)
Period.append(y[0]); Energies.append(y[1])
y = Superposition('eigenvalue03', 'eigenvalue02', x)
Period.append(y[0]); Energies.append(y[1])
y = Superposition('eigenvalue04', 'eigenvalue03', x)
Period.append(y[0]); Energies.append(y[1])
Period = np.asarray(Period)
Energies = np.asarray(Energies)
popt, pcov = curve_fit(FitFunc, Energies, Period)
plt.figure()
plt.xlabel(r'$\Delta$E')
plt.ylabel('T')
ELin = np.linspace(1., 7., 1000)
plt.plot(ELin, FitFunc(ELin, popt[0]), \
         label = r'T = $\frac{%.2f}{\Delta E}$' % popt[0])
plt.plot(Energies, Period, 'x')
plt.legend()
plt.savefig('plots/periodicity.eps', format = 'eps', dpi = 1000)
plt.close()