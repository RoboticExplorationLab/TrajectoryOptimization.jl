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

# Unconstrained
Q = (1e-1)*Matrix(I,n,n)
Q[4,4] = 1.0; Q[5,5] = 1.0; Q[6,6] = 1.0; Q[7,7] = 1.0
R = (1.0)*Matrix(I,m,m)
# R = (1e-1)*Matrix(I,m,m)
Qf = (1000.0)*Matrix(I,n,n)
tf = 5.0
dt = 0.05

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;40.;0.] # xyz position
xf[4:7] = q0

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# Model + objective
quadrotor = [model, obj_uncon]

## Constrained

r_quad = 3.0
r_sphere = 3.0
spheres = ((0.,10.,0.,r_sphere),(0.,20.,0.,r_sphere),(0.,30.,0.,r_sphere))
n_spheres = 3

# -control limits
u_min = 0.0
u_max = 10.0

# 3 sphere obstacles
function cI_3obs_quad(c,x,u)
    for i = 1:n_spheres
        c[i] = TrajectoryOptimization.sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
    end
    c
end

# unit quaternion constraint
function unit_quaternion(c,x,u)
    c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

obj_uq = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,cE=unit_quaternion)
obj_3obs = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,cI=cI_3obs_quad,cE=unit_quaternion)

quadrotor_unit_quaternion = [model, obj_uq]
quadrotor_3obs = [model, obj_3obs]
