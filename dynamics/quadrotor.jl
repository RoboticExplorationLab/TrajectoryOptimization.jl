function quadrotor_dynamics!(ẋ,X,u)
      #TODO change concatentations to make faster!
      # Quaternion representation
      # Modified from D. Mellinger, N. Michael, and V. Kumar,
      # "Trajectory generation and control for precise aggressive maneuvers with quadrotors",
      # In Proceedings of the 12th International Symposium on Experimental Robotics (ISER 2010), 2010.

      ## States: X ∈ R^13; q = [s;v]
      # x
      # y
      # z
      # q0
      # q1
      # q2
      # q3
      # xdot
      # ydot
      # zdot
      # omega1
      # omega2
      # omega3

      x = X[1:3]
      q = X[4:7]./norm(X[4:7]) #normalize quaternion
      v = X[8:10]
      omega = X[11:13]

      # Parameters
      m = .5 # mass
      J = Matrix(Diagonal([0.0023; 0.0023; 0.004])) # inertia matrix
      Jinv = Matrix(Diagonal(1.0./[0.0023; 0.0023; 0.004])) # inverted inertia matrix
      g = 9.81 # gravity
      L = 0.1750 # distance between motors

      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      kf = 1; # 6.11*10^-8;
      F1 = kf*w1;
      F2 = kf*w2;
      F3 = kf*w3;
      F4 = kf*w4;
      F = [0;0;F1+F2+F3+F4] #total rotor force in body frame

      km = 0.0245;
      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;
      tau = [L*(F2-F4);L*(F3-F1);(M1-M2+M3-M4)] #total rotor torque in body frame

      ẋ[1:3] = v # velocity in world frame
      ẋ[4:7] = 0.5*qmult(q,[0;omega]) #quaternion derivative
      ẋ[8:10] = [0;0;-g] + (1/m)*qrot(q,F) #acceleration in world frame
      ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
end

function quadrotor_dynamics(X,u)
      ẋ = zeros(13,1)
      quadrotor_dynamics!(ẋ,X,u)
      ẋ
end


## Utilities
"""
@(SIGNATURES)
    Rotate a vector by a quaternion
"""
function qrot(q,r)
      r + 2*cross(q[2:4],cross(q[2:4],r) + q[1]*r)
end

"""
@(SIGNATURES)
    Multiplication of two quaternions (q = [s;v])
"""
function qmult(q1,q2)
      [q1[1]*q2[1] - q1[2:4]'*q2[2:4]; q1[1]*q2[2:4] + q2[1]*q1[2:4] + cross(q1[2:4],q2[2:4])]
end

# Model
n = 13
m = 4

model = Model(quadrotor_dynamics!,n,m)

# Objective and constraints
Qf = 100.0*Diagonal(I,n)
Q = (0.1)*Diagonal(I,n)
R = (0.01)*Diagonal(I,m)
tf = 5.0
dt = 0.05

# -initial state
x0 = zeros(n)
x0[4:7] = [1;0;0;0]

# -final state
xf = zeros(n)
xf[1:3] = [10.0;10.0;5.0] # xyz position
xf[4:7] = [1;0;0;0]

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# Model + objective
quadrotor = [model, obj_uncon]
