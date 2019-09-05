include(joinpath(pwd(),"dynamics/quaternions.jl"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

load_params = (m=0.9,
            gravity=SVector(0,0,-9.81),
            radius=0.2)

quad_params = (m=1.0,
            J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
            Jinv=SMatrix{3,3}(Diagonal(1.0./[0.0023, 0.0023, 0.004])),
            gravity=SVector(0,0,-9.81),
            motor_dist=0.175,
            kf=1.0,
            km=0.0245,
            radius=0.275)

#~~~~~~~~~~~~~~~~~~~~~~~ BATCH PROBLEM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function load_dynamics!(ẋ,x,u)
    ẋ[1:3] = x[4:6]
    ẋ[4:6] = u[1:3]
    ẋ[6] -= 9.81 # gravity
end

function lift_dynamics!(ẋ,x,u,params)

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
      ẋ[8:10] = g + (1/m)*(q*F + u[5:7]) #acceleration in world frame
      ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      return tau, omega, J, Jinv
end

function batch_dynamics!(ẋ,x,u, params)
    lift_params = params.lift
    load_params = params.load
    load_mass = load_params.m

    x1 = x[1:3]
    x2 = x[13 .+ (1:3)]
    x3 = x[2*13 .+ (1:3)]

    xload = x[3*13 .+ (1:3)]

    dir1 = (xload - x1)/norm(xload - x1)
    dir2 = (xload - x2)/norm(xload - x2)
    dir3 = (xload - x3)/norm(xload - x3)

    lift_control_1 = [u[1:4]; u[5]*dir1]
    lift_control_2 = [u[5 .+ (1:4)]; u[10]*dir2]
    lift_control_3 = [u[2*5 .+ (1:4)]; u[15]*dir3]
    u_slack_load = -1.0*(u[3*5 + 1]*dir1 + u[3*5 + 2]*dir2 + u[3*5 + 3]*dir3)

    lift_dynamics!(view(ẋ,1:13),x[1:13],lift_control_1,lift_params)
    lift_dynamics!(view(ẋ,13 .+ (1:13)),x[13 .+ (1:13)],lift_control_2,lift_params)
    lift_dynamics!(view(ẋ,2*13 .+ (1:13)),x[2*13 .+ (1:13)],lift_control_3,lift_params)

    load_dynamics!(view(ẋ,3*13 .+ (1:6)),x[3*13 .+ (1:6)],u_slack_load/load_mass)

    return nothing
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~ SEQUENTIAL PROBLEMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


function gen_load_model_initial(xload0,xlift0,load_params)
    num_lift = length(xlift0)
    mass_load = load_params.m
    function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
        Δx = [xlift[1:3] - xload0[1:3] for xlift in xlift0]
        u_slack = @SVector zeros(3)
        for i = 1:num_lift
            u_slack += u[i]*normalize(Δx[i])
        end
        Dynamics.double_integrator_3D_dynamics!(ẋ, x, u_slack/mass_load)
    end
    Model(double_integrator_3D_dynamics_load!,6,num_lift)
end

function gen_lift_model_initial(xload0,xlift0,quad_params)

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
            Δx = xload0[1:3] - xlift0[1:3]
            dir = Δx/norm(Δx)
            ẋ[8:10] = g + (1/m)*(q*F + u[5]*dir) # acceleration in world frame
            ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
            return tau, omega, J, Jinv
        end
        Model(quadrotor_lift_dynamics!,13,5,quad_params)
end

function gen_lift_model(X_load,N,dt,quad_params)
      model = Model[]

      for k = 1:N-1
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
            Δx = X_load[k][1:3] - x[1:3]
            dir = Δx/norm(Δx)
            ẋ[8:10] = g + (1/m)*(q*F + u[5]*dir) # acceleration in world frame
            ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
            return tau, omega, J, Jinv
        end
        push!(model,midpoint(Model(quadrotor_lift_dynamics!,13,5,quad_params),dt))
    end
    model
end

function gen_load_model(X_lift,N,dt,load_params)
    model = Model[]
    mass_load = load_params.m
    num_lift = length(X_lift)
    for k = 1:N-1
        function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
            Δx = [xlift[k][1:3] - x[1:3] for xlift in X_lift]
            u_slack = @SVector zeros(3)
            for i = 1:num_lift
                u_slack += u[i]*Δx[i]/norm(Δx[i])
            end
            Dynamics.double_integrator_3D_dynamics!(ẋ, x, u_slack/mass_load)
        end
        push!(model,midpoint(Model(double_integrator_3D_dynamics_load!,6, num_lift),dt))
    end
    model
end
