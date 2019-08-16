using StaticArrays
num_lift = 3
num_load = 1

n_slack = 3

# 3D
di_mass_load = .350
di_mass_lift = .850
# Double integrator lift model
function double_integrator_3D_dynamics_lift!(ẋ,x,u)
    u_input = u[1:3]
    u_slack = u[4:6]
    Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_input+u_slack/di_mass_lift) end

doubleintegrator3D_lift = Model(double_integrator_3D_dynamics_lift!,Dynamics.doubleintegrator3D.n,Dynamics.doubleintegrator3D.m + n_slack)
doubleintegrator3D_lift.info[:radius] = 0.275

# Double integrator load model
function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
    u_slack1 = u[1:3]
    u_slack2 = u[4:6]
    u_slack3 = u[7:9]
    Dynamics.double_integrator_3D_dynamics!(ẋ,x,(u_slack1+u_slack2+u_slack3)/di_mass_load)
end

doubleintegrator3D_load = Model(double_integrator_3D_dynamics_load!,Dynamics.doubleintegrator3D.n,n_slack*num_lift)
doubleintegrator3D_load.info[:radius] = 0.2


function gen_di_load_dyn(num_lift,n_slack=3)

      function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
          u_slack = [u[3*(i-1) .+ (1:3)] for i = 1:num_lift]
          Dynamics.double_integrator_3D_dynamics!(ẋ,x,sum(u_slack)/di_mass_load)
      end

      doubleintegrator3D_load = Model(double_integrator_3D_dynamics_load!,Dynamics.doubleintegrator3D.n,n_slack*num_lift)
      doubleintegrator3D_load.info[:radius] = 0.2

      doubleintegrator3D_load
end


# Quadrotor lift model
function quadrotor_lift_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,params)
      q = normalize(Quaternion(view(x,4:7)))
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

      u_slack = u[5:7]

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
      ẋ[4:7] = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
      ẋ[8:10] = g + (1/m)*(q*F + u_slack) # acceleration in world frame
      ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      return tau, omega, J, Jinv
end

quad_params = (m=0.850,
             J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
             Jinv=SMatrix{3,3}(Diagonal(1.0./[0.0023, 0.0023, 0.004])),
             gravity=SVector(0,0,-9.81),
             motor_dist=0.175,
             kf=1.0,
             km=0.0245)

quadrotor_lift = Model(quadrotor_lift_dynamics!, 13, 7, quad_params)
# quadrotor_lift.info[:radius] = 0.275
quadrotor_lift.info[:radius] = 0.5  # keep 2m distance between quads
