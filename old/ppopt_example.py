#!/usr/bin/env python 

# copied from https://github.com/TAMUparametric/PPOPT/blob/903f1f7b6c219eb9f445847444839a501b9b2558/doc/control_allocation_example.rst


import numpy as np
import ipdb

# Vehicle Parameters
m = 5.0  # Vehicle mass [kg]
g = 9.8  # Gravitational acceleration [m/s^2]
r = 0.35 # Rotor radius [m]
rotDir = np.array([+1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0], float)
    # positive is right handed rotation about thrust direction
n = 8 # Number of rotors
m = 4 # Number of control axes

# Rotor locations in x forward, y right, z down coordinate frame at CG
# Assumed to be axisymmetric about CG
phi = np.linspace(0.0, 2.0*np.pi, n+1, True, False, float)
phi = phi[0:-1] # Azimuth
xRotor = np.sin(phi) # Rotor x location [m]
yRotor = np.cos(phi) # Rotor y location [m]
zRotor = np.zeros((n), float) # Rotor z location [m]

# Rotor Parameters
Ct = 0.014 # Thrust coefficient [T = rho*pi*r^4*Ct*omega^2]
FoM = 0.7  # Figure of merit
Cq = Ct**1.5/FoM/np.sqrt(2.0) # Torque coefficient [Q = rho*pi*r^5*Cq*omega^2]

# Motor Sizing
thrustRatio = 1.4 # Individual rotor maximum thrust over hover thrust



# Generate hover control Jacobian: dFM/dTau
# Ignore Fx and Fy (assume no cant on rotors)
dFMdTau = np.zeros((m, n), float)

# Thrust per torque
dTdTau = Ct/(r*Cq)

# Fz = -dTdTau
dFMdTau[0,:] = -dTdTau*np.ones((n), float)

# Mx = dTdTau*-y
dFMdTau[1,:] = -dTdTau*yRotor

# My = dTdTau*x
dFMdTau[2,:] = dTdTau*xRotor

# Mz = rotDir
dFMdTau[3,:] = rotDir



# Force/Moment trim command, assuming CG is at origin
FMTrim = np.array([-m*g, 0.0, 0.0, 0.0], float) # 1g thrust, 0 moment commands

# Compute trim actuation using Moore-Penrose pseudo-inverse
xTrim = (np.linalg.pinv(dFMdTau)@FMTrim.reshape((4, 1))).reshape((n))


WFM = np.diag([20.0, 100.0, 100.0, 5.0]) # Fz, Mx, My, Mz



xMin = np.zeros((n), float)
xMax = thrustRatio*np.mean(xTrim)*np.ones((n), float)

rollPitchMomentLimits = np.array([-15.0, 15.0], float)
yawMomentLimits = np.array([-3.0, 3.0], float)
thrustLimits = np.array([-1.2*m*g, -0.8*m*g], float)


FMCmdMin = np.array([thrustLimits[0], rollPitchMomentLimits[0], rollPitchMomentLimits[0], yawMomentLimits[0]])*1.1
FMCmdMax = np.array([thrustLimits[1], rollPitchMomentLimits[1], rollPitchMomentLimits[1], yawMomentLimits[1]])*1.1

Q = dFMdTau.T@WFM@dFMdTau + dFMdTau.T@dFMdTau
c = -dFMdTau.T@dFMdTau@xTrim.reshape((n, 1))
H = -dFMdTau.T@WFM


A = np.concatenate((-np.eye(n, n, 0, float), np.eye(n, n, 0, float)), 0)
b = np.concatenate((-xMin.reshape((n, 1)), xMax.reshape((n, 1))), 0)
F = np.zeros((2*n, m), float)

CRa = np.concatenate((-np.eye(m, m, 0, float), np.eye(m, m, 0, float)), 0)
CRb = np.concatenate((-FMCmdMin.reshape((m, 1)), FMCmdMax.reshape((m, 1))), 0)


from ppopt.mpqp_program import MPQP_Program as mpqp_program
prog = mpqp_program(A, b, c, H, Q, CRa, CRb, F)


print('processing constraints')
prog.process_constraints()


print('solving problem!!!')
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
solution = solve_mpqp(prog, mpqp_algorithm.combinatorial)


# seems to work nicely, runs in just a few seconds. has 518 critical regions. 
# solution.evaluate returns None though.
# digging a tiny bit in the source we see that all critical regions 
# of the solution return region.is_inside(pt) == False. Why?

pt = np.random.random((4,))*(FMCmdMax - FMCmdMin) + FMCmdMin
print(solution.evaluate(pt))

ipdb.set_trace()
