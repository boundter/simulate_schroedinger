import numpy as np
from scipy.integrate import ode
from math import copysign
import matplotlib.pyplot as plt

__all__ = ['SchroedingerWave', 'StationarySchroedinger', 'GeneralSchroedinger']

class SchroedingerWave:
  """ 
  Class of a typical solution for the Schroedinger-equation. 

  Attributes:
  x: Spatial array
  Psi: Array of Psi(x)

  Methods:
  PlotWave(xRange, yRange, FileName, Label = ''): 
   Plots the Wavefunction in the specified range and saves it to plots/FileName
  PlotAbsolute2(xRange, yRange, FileName, Label = ''): 
   Plots |Psi(x)|^2 in the specified range and saves it to plots/FileName
  IntegrateAbsolute2(Begin, End): 
   Integrates |Psi(x)|^2 in the range[Begin, End] and returns the value
  Normalize():
    Normalizes the wave by dividing by sqrt(IntegrateAbsolute2)
  """
  def __init__(self, x, Psi):
    self.x = x # np.array
    self.Psi = Psi # Psi(x)

  def __call__(self):
    return self.Psi

  def PlotWave(self, xRange, yRange, FileName, Label = ''):
    plt.figure()
    plt.xlabel('x'); plt.ylabel(r'$\Psi(x)$')
    plt.plot(self.x, self.Psi.real, label = 'Real Part%s' %Label)
    plt.plot(self.x, self.Psi.imag, label = 'Imag Part%s' %Label)
    plt.legend()
    plt.xlim(xRange); plt.ylim(yRange)
    plt.savefig('plots/%s' %FileName, format = 'eps', dpi = 1000)
    plt.close()

  def PlotAbsolute2(self, xRange, yRange, FileName, Label = ''):
    plt.figure()
    plt.xlabel('x'); plt.ylabel(r'$\vert\Psi(x)\vert^2$')
    plt.plot(self.x, np.absolute(self.Psi)**2, label = Label)
    if Label: plt.legend()
    plt.xlim(xRange); plt.ylim(yRange)
    plt.savefig('plots/%s' %FileName, format = 'eps', dpi = 1000)
    plt.close()    

  def _FindNearestIndex(self, x):
    return (np.abs(self.x-x)).argmin()

  def IntegrateAbsolute2(self, Begin, End):
    BeginIndex = self._FindNearestIndex(Begin)
    EndIndex = self._FindNearestIndex(End) - 1
    dx = self.x[BeginIndex + 1: EndIndex + 1] - self.x[BeginIndex: EndIndex]
    dPsi2 = (np.absolute(self.Psi[BeginIndex + 1: EndIndex + 1])**2 + \
             np.absolute(self.Psi[BeginIndex: EndIndex])**2)/2
    Integral = np.sum(dx*dPsi2)
    return Integral

  def Normalize(self):
    self.Psi /= np.sqrt(self.IntegrateAbsolute2(self.x[0], self.x[-1]))

class StationarySchroedinger(SchroedingerWave):
  """ 
  Class for a Stationary Schroedinger-equation.

  Attributes:
  x: Spatial array
  V: Function for the Potential V(x)
  E: Energy of the Wave
  Psi: Psi(x) needs to be initialized by calling Integrate() with save 
    parameter 

  Methods:
  PlotWave(xRange, yRange, FileName, Label = ''): 
   Plots the Wavefunction in the specified range and saves it to plots/FileName
  PlotAbsolute2(xRange, yRange, FileName, Label = ''): 
   Plots |Psi(x)|^2 in the specified range and saves it to plots/FileName
  IntegrateAbsolute2(Begin, End): 
   Integrates |Psi(x)|^2 in the range[Begin, End] and returns the value
  Normalize():
    Normalizes the wave by dividing by sqrt(IntegrateAbsolute2)
  Integrate(Start, End = 0., Scaling = 1e-20, save = False):
    Integrates the stationary Schroedinger-equation from Start to End using 
    Runge-Kutta. Scaling helps to prevent overflow or underflow. The save 
    parameter saves the solution in the Psi Attribute
  w(Boundary = 10.):
    Calculate w = Psi_-*Phi_+ - Phi_-*Psi+, where a root of w corrresponds to
    an eigenvalue E/ eigenfunction Psi
  FlipAntisymmetric():
    Flips an solution, if it is antisymmetric. This method should be called 
    after an eigenfunction has been found, otherwise all solutions will be
    symmetric 
  """
  def __init__(self, x, V, E):
    self.x = x
    self.Psi = np.zeros(len(x))
    self.V = V
    self.E = E

  def __call__(self):
    SchroedingerWave.__call__(self)

  def _InitialCondition(self, x, Scaling):
    Psi = Scaling*np.exp(-np.sqrt(self.E)*np.absolute(x))
    if x < 0:
      Phi = np.sqrt(self.E)*Psi
    else:
      Phi = -np.sqrt(self.E)*Psi
    return [Psi, Phi]

  def _SecondDerivative(self, x, y):
    Psi, Phi = y
    f = self.V(x) - self.E
    return [Phi, f*Psi]

  def Integrate(self, Start, End = 0., Scaling = 1e-20, save = False):
    InitialIndex = SchroedingerWave._FindNearestIndex(self, Start)
    FinalIndex = SchroedingerWave._FindNearestIndex(self, End)
    Iterator = -int(copysign(1., Start))
    Initial = self._InitialCondition(Start, Scaling)
    if save: self.Psi[InitialIndex] = Initial[0]
    Integrator = ode(self._SecondDerivative).set_integrator('dopri5')
    Integrator.set_initial_value(Initial, self.x[InitialIndex])
    for i in np.linspace(InitialIndex, FinalIndex, abs(FinalIndex - InitialIndex) + 1, dtype=int):
      dx = self.x[i + Iterator] - self.x[i]
      Integrator.integrate(Integrator.t + dx)
      i += Iterator
      if save: self.Psi[i] = Integrator.y[0]
    return Integrator.y

  def w(self, Boundary = 10.):
    PsiPlus = self.Integrate(Boundary)
    PsiMinus = self.Integrate(-Boundary)
    return (PsiPlus[1]*PsiMinus[0] - PsiMinus[1]*PsiPlus[0])

  def FlipAntisymmetric(self):
    ZeroIndex = SchroedingerWave._FindNearestIndex(self, 0.)
    if abs(self.Psi[ZeroIndex]) <= 2e-3 and \
       int(copysign(1, self.Psi[ZeroIndex-1])) == int(copysign(1, self.Psi[ZeroIndex+1])):
      self.Psi[0:ZeroIndex] *= -1

class GeneralSchroedinger(SchroedingerWave):
  """ 
  Class of a typical solution for the Schroedinger-equation. 

  Attributes:
  x: Spatial array
  Psi: Array of Psi(x)
  V: Spatial array of the Potential. It will be initialized by a function V(x)

  Methods:
  PlotWave(xRange, yRange, FileName, Label = ''): 
   Plots the Wavefunction in the specified range and saves it to plots/FileName
  PlotAbsolute2(xRange, yRange, FileName, Label = ''): 
   Plots |Psi(x)|^2 in the specified range and saves it to plots/FileName
  IntegrateAbsolute2(Begin, End): 
   Integrates |Psi(x)|^2 in the range[Begin, End] and returns the value
  Normalize():
    Normalizes the wave by dividing by sqrt(IntegrateAbsolute2)
  IntegrateTime(dt):
    Integrates Psi one timestep dt by using the Crank-Nicolson-scheme
  """
  def __init__(self, x, Psi, V, t0 = 0.):
    self._x = x
    self.Psi = Psi.astype(np.complex_)
    self.V = V(x)
    self.t = t0
    self._Calcdx()

  def __call__(self):
    SchroedingerWave.__call__(self)

  def _Getx(self):
    return self._x

  def _Setx(self, x):
    self._x = x
    self._Calcdx()

  def _Calcdx(self):
    self.dx = np.mean(self._x[1:] - self._x[0:-1])

  def _RightSide(self, dt):
    A = 0.5j*dt/self.dx**2
    B = 1 - 1j*dt/self.dx**2 - 0.5j*dt*self.V
    F = np.zeros(len(self.Psi), dtype = np.complex_)
    F[1:-1] = A*(self.Psi[0:-2] + self.Psi[2:]) + B[1:-1]*self.Psi[1:-1]
    F[0] = B[0]*self.Psi[0] + A*self.Psi[1]
    F[-1] = B[-1]*self.Psi[-1] + A*self.Psi[-2]
    return F

  def _TestFunction(self, dt, F):
    A = -0.5j*dt/self.dx**2
    B = 1 + 1j*dt/self.dx**2 + 0.5j*dt*self.V
    FRef = np.zeros(len(self.Psi), dtype = np.complex_)
    FRef[1:-1] = A*(self.Psi[0:-2] + self.Psi[2:]) + B[1:-1]*self.Psi[1:-1]
    FRef[0] = B[0]*self.Psi[0] + A*Psi[1]
    FRef[-1] = B[-1]*self.Psi[-1] + A*Psi[-2]
    print(np.amax((FRef - F).real))


  def IntegrateTime(self, dt):
    A = -0.5j*dt/self.dx**2
    B = 1 + 1j*dt/self.dx**2 + 0.5j*dt*self.V
    F = self._RightSide(dt)
    alpha = np.zeros(len(self.Psi), dtype = np.complex_)
    beta = np.zeros(len(self.Psi), dtype = np.complex_)
    alpha[0] = F[0]/B[0]; beta[0] = -A/B[0]
    for i in range(1, len(self.Psi)):
      alpha[i] = (F[i] - A*alpha[i-1])/(B[i] + A*beta[i-1])
      beta[i] = -A/(B[i] + A*beta[i-1])
    self.Psi[-1] = alpha[-1]
    for i in np.linspace(len(self.Psi) - 2, 0, len(self.Psi) + 1,\
                 dtype = int):
      self.Psi[i] = alpha[i] + beta[i]*self.Psi[i+1]
    self.t += dt

  x = property(_Getx, _Setx)  
