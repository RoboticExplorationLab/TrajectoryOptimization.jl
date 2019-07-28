using Plots

include(joinpath(pwd(),"dynamics/quaternions.jl"))

function quadrotor_lift_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,params) where T
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
      q = normalize(Quaternion(view(x,4:7)))
      # q = view(x,4:7)
      # normalize!(q)
      v = view(x,8:10)
      omega = view(x,11:13)

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

      f_load = u[5:7]

      kf = params[:kf]; # 6.11*10^-8;
      F1 = kf*w1;
      F2 = kf*w2;
      F3 = kf*w3;
      F4 = kf*w4;
      F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

      km = params[:km]
      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;
      tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

      ẋ[1:3] = v # velocity in world frame
      # ẋ[4:7] = 0.5*qmult(q,[0;omega]) #quaternion derivative
      ẋ[4:7] = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
      ẋ[8:10] = g + (1/m)*(q*F + f_load) #acceleration in world frame
      ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      return tau, omega, J, Jinv
end

quad_params = (m=0.5,
             J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
             Jinv=SMatrix{3,3}(Diagonal(1.0./[0.0023, 0.0023, 0.004])),
             gravity=SVector(0,0,-9.81),
             motor_dist=0.1750,
             kf=1.0,
             km=0.0245)

quadrotor_lift = Model(quadrotor_lift_dynamics!, 13, 7, quad_params)
quadrotor_lift_d = rk3(quadrotor_lift)

N = 21
dt = 0.1
Q = 1.0e-2*Diagonal(I,quadrotor_lift.n)
Qf = 1.0*Diagonal(I,quadrotor_lift.n)
R = 1.0e-4*Diagonal(I,quadrotor_lift.m)


x0 = zeros(quadrotor_lift.n)
x0[4] = 1.
xf = copy(x0)
obj = LQRObjective(Q,R,Qf,xf,N)

u_lim = 9.81*(quad_params.m + 1.)/4.
U_lift = [[u_lim;u_lim;u_lim;u_lim;0.;0.;-9.81] for k = 1:N-1]

prob_lift = Problem(quadrotor_lift,
                obj,
                U_lift,
                integration=:midpoint,
                x0=x0,
                xf=xf,
                N=N,
                dt=dt)


rollout!(prob_lift)

plot(prob_lift.X,1:3)

function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
    u_slack1 = u[1:3]
    Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_slack1)
end

n_load = Dynamics.doubleintegrator3D.n
m_load = 3
doubleintegrator3D_load = Model(double_integrator_3D_dynamics_load!,n_load,m_load)
