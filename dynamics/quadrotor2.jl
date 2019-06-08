using StaticArrays
function quadrotor_dynamics2(x::AbstractVector,u::AbstractVector,params) where T
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

      # x = X[1:3]
      q = normalize(Quaternion(x[SVector(4,5,6,7)]))
      # q = view(x,4:7)
      # normalize!(q)
      # v = view(x,8:10)
      v = x[SVector(8,9,10)]
      # omega = view(x,11:13)
      omega = x[SVector(11,12,13)]

      # Parameters
      m = params[:m] # mass
      J = params[:J] # inertia matrix
      Jinv = params[:Jinv] # inverted inertia matrix
      g = params[:gravity] # gravity
      L = params[:motor_dist] # distance between motors

      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      kf = params[:kf]; # 6.11*10^-8;
      # F = kf*u
      F1 = kf*w1
      F2 = kf*w2
      F3 = kf*w3
      F4 = kf*w4
      F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

      km = params[:km]
      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;
      tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

      xdot = v # velocity in world frame
      # ẋ[4:7] = 0.5*qmult(q,[0;omega]) #quaternion derivative
      qdot = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
      vdot = g + (1/m)*(q*F) #acceleration in world frame
      ωdot = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      return [xdot; qdot; vdot; ωdot]
end

quadrotor_model2 = Model(quadrotor_dynamics2!, 13, 4, quad_params)
