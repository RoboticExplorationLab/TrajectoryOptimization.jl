function quadrotor_dynamics!(ẋ,X,u)
      #TODO change concatentations to make faster!
      # Quaternion representation
      # Modified from D. Mellinger, N. Michael, and V. Kumar,
      # "Trajectory generation and control for precise aggressive maneuvers with quadrotors",
      # In Proceedings of the 12th International Symposium on Experimental Robotics (ISER 2010), 2010.

      ## States: X ∈ R^13; q = [v;s]
      # x
      # y
      # z
      # q1
      # q2
      # q3
      # q4
      # ẋ
      # ydot
      # zdot
      # omega1
      # omega2
      # omega3

      x = X[1:3]
      q = X[4:7]
      v = X[8:10]
      omega = X[11:13]

      # Parameters
      m = .5 # mass
      IM = Matrix(Diagonal([0.0023,0.0023,0.004])) # inertia matrix
      invIM = Matrix(Diagonal(1 ./[0.0023,0.0023,0.004])) # inverted inertia matrix
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

      km = 0.0245;

      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;
      tmp = hamilton_product(q,hamilton_product([0;0;F1+F2+F3+F4;0],quaternion_conjugate(q)))#TODO does the quaternion need to be unit when we do this rotation? or is unit quaternion only required when we convert quaterion to rotation matrix
      a = (1/m)*([0;0;-m*g] + tmp[1:3]);

      # if !all(isapprox.(quat2rot(q)*[0;0;F1+F2+F3+F4],tmp[1:3]))
      #       println("$(tmp[1:3])")
      #       println("$(quat2rot(q)*[0;0;F1+F2+F3+F4])")
      #       error("hamilton product does not match rotation matrix")
      # end

      # a = (1/m)*([0;0;-m*g] + quat2rot(q)*[0;0;F1+F2+F3+F4]);

      ẋ[1:3] = v # velocity
      ẋ[4:7] = 0.5*hamilton_product(q,[omega;0]) # TODO should q be unit?
      ẋ[8:10] = a # acceleration
      ẋ[11:13] = invIM*([L*(F2-F4);L*(F3-F1);(M1-M2+M3-M4)] - cross(omega,IM*omega)) # ̇ω; Euler's equation: I(̇ω) + ω x Iω = τ
end

function quadrotor_dynamics(X,u)
      ẋ = zero(13,1)
      quadrotor_dynamics!(ẋ,X,u)
      ẋ
end


## Utilities
"""
@(SIGNATURES)
    Convert quaternion to unit quaternion
"""
function unit_quat(q)
      q./norm(q)
end

"""
@(SIGNATURES)
    Multiplication of two quaternions (q = [v;s]) using Hamilton product
"""
function hamilton_product(q1,q2)
      # perform quaternion multiplication
      Q1 = [q1[4] -q1[3] q1[2] q1[1];
            q1[3]  q1[4] -q1[1]  q1[2];
            -q1[2]  q1[1]  q1[4] q1[3];
            -q1[1] -q1[2]  -q1[3]  q1[4]]

      Q1*q2
end

"""
@(SIGNATURES)
    Quaternion conjugate for q = [v;s]
"""
function quaternion_conjugate(q)
      # calculate the congugate of a quaternion: q^+; q = [v;s] -> q^+ = [-v;s]
      q_ = zero(q)
      q_[1] = -1*q[1]
      q_[2] = -1*q[2]
      q_[3] = -1*q[3]
      q_[4] = 1*q[4]

      q_
end

function quat2rot(q)
      q = q./norm(q)
      x = q[1]; y = q[2]; z = q[3]; w = q[4]

      [(-z^2 - y^2 + x^2 + w^2) (2*x*y - 2*z*w) (2*x*z + 2*y*w);
       (2*z*w + 2*x*y) (-z^2 + y^2 - x^2 + w^2) (2*y*z - 2*x*w);
       (2*x*z - 2*y*w) (2*y*z + 2*x*w) (z^2 - y^2 - x^2 + w^2)]
end

# w = [1;2;3;4]
# w /= norm(w)
#
# omega = [1;2;3]
#
# hamilton_product(w,[omega;0])
# hamilton_product([omega;0],w)

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
quat0 = TrajectoryOptimization.eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
x0[4:7] = quat0
x0

# -final state
xf = zeros(n)
xf[1:3] = [10.0;10.0;5.0] # xyz position
quatf = TrajectoryOptimization.eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
xf[4:7] = quatf
xf

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

# Model + objective
quadrotor = [model, obj_uncon]
