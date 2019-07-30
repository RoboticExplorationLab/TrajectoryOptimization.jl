using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO

# visualization
function cable_transform(y,z)
    v1 = [0,0,1]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1,v2)
    ang = acos(v1'v2)
    R = AngleAxis(ang,ax...)
    compose(Translation(z),LinearMap(R))
end

function plot_cylinder(vis,c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setobject!(vis["cyl"][name],geom,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(vis,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0)),"cyl_$i")
    end
end

function visualize_lift_system(vis,prob_lift,prob_load,r_lift,r_load,n_slack=3)
    num_lift = length(prob_lift)

    # camera angle
    settransform!(vis["/Cameras/default"], compose(Translation(5., -3, 3.),LinearMap(RotX(pi/25)*RotZ(-pi/2))))

    # load in quad mesh
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.07
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling

    # intialize system
    for i = 1:num_lift
        setobject!(vis["lift$i"]["sphere"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 0.25)))
        setobject!(vis["lift$i"]["robot"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    addcylinders!(vis,_cyl,3.)

    anim = MeshCat.Animation(convert(Int,floor(1/prob_lift[1].dt)))
    for k = 1:prob_lift[1].N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob_load.X[k][1:n_slack]
            for i = 1:num_lift
                x_lift = prob_lift[i].X[k][1:n_slack]
                settransform!(frame["cable"]["$i"], cable_transform(x_lift,x_load))
                settransform!(frame["lift$i"], compose(Translation(x_lift...),LinearMap(Quat(prob_lift[i].X[k][4:7]...))))

            end
            settransform!(frame["load"], Translation(x_load...))
        end
    end
    MeshCat.setanimation!(vis,anim)
end

function gen_lift_cable_constraints(X_load,U_load,agent,n,m,d,n_slack=3)
    N = length(X_load)
    con_cable_lift = []
    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                c[1:n_slack] = u[(end-(n_slack-1)):end] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
            else
                c[1] = norm(x[1:n_slack] - X_load[k][1:n_slack])^2 - d^2
                if k < N
                    c[1 .+ (1:n_slack)] = u[(end-(n_slack-1)):end] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
                end
            end
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            if k == 1
                C[1:n_slack,(end-(n_slack-1)):end] = Is
            else
                C[1,1:n_slack] = 2*dif
                if k < N
                    C[1 .+ (1:n_slack),(end-(n_slack-1)):end] = Is
                end
            end
        end
        if k == 1
            p_con = n_slack
        else
            k < N ? p_con = 1+n_slack : p_con = 1
        end
        # p_con = 1
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_lift)
        push!(con_cable_lift,cc)
    end

    return con_cable_lift
end

function gen_load_cable_constraints(X_lift,U_lift,n,m,d,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    con_cable_load = []

    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    c[_shift .+ (1:n_slack)] = U_lift[i][k][(end-(n_slack-1)):end] + u[(i-1)*n_slack .+ (1:n_slack)]
                    _shift += n_slack
                end
            else
                for i = 1:num_lift
                    c[i] = norm(X_lift[i][k][1:n_slack] - x[1:n_slack])^2 - d[i]^2
                end

                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        c[_shift .+ (1:n_slack)] = U_lift[i][k][(end-(n_slack-1)):end] + u[(i-1)*n_slack .+ (1:n_slack)]
                        _shift += n_slack
                    end
                end
            end
        end

        function ∇con(C,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    u_idx = ((i-1)*n_slack .+ (1:n_slack))
                    C[_shift .+ (1:n_slack),n .+ u_idx] = Is
                    _shift += n_slack
                end
            else
                for i = 1:num_lift
                    x_pos = X_lift[i][k][1:n_slack]
                    x_load_pos = x[1:n_slack]
                    dif = x_pos - x_load_pos
                    C[i,1:n_slack] = -2*dif
                end
                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        u_idx = ((i-1)*n_slack .+ (1:n_slack))
                        C[_shift .+ (1:n_slack),n .+ u_idx] = Is
                        _shift += n_slack
                    end
                end
            end
        end
        if k == 1
            p_con = num_lift*n_slack
        else
            k < N ? p_con = num_lift*(1 + n_slack) : p_con = num_lift
        end
        # p_con = num_lift
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_load)
        push!(con_cable_load,cc)
    end

    return con_cable_load
end

function gen_self_collision_constraints(X_lift,agent,n,m,r_lift,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    p_con = num_lift - 1

    self_col_con = []

    for k = 1:N
        function col_con(c,x,u=zeros())
            p_shift = 1
            for i = 1:num_lift
                if i != agent
                    x_pos = x[1:n_slack]
                    x_pos2 = X_lift[i][k][1:n_slack]
                    c[p_shift] = (r_lift + r_lift)^2 - norm(x_pos - x_pos2)^2
                    p_shift += 1
                end
            end
        end

        function ∇col_con(C,x,u=zeros())
            p_shift = 1
            for i = 1:num_lift
                if i != agent
                    x_pos = x[1:n_slack]
                    x_pos2 = X_lift[i][k][1:n_slack]
                    dif = x_pos - x_pos2
                    C[p_shift,1:n_slack] = -2*dif
                    p_shift += 1
                end
            end
        end

        push!(self_col_con,Constraint{Inequality}(col_con,∇col_con,n,m,p_con,:self_col))
    end

    return self_col_con
end


function solve_admm(prob_lift,prob_load,n_slack,admm_type,opts)
    # admm_type = :sequential
    # admm_type = :parallel

    num_lift = length(prob_lift)
    n_lift = prob_lift[1].model.n
    m_lift = prob_lift[1].model.m
    n_load = prob_load.model.n
    m_load = prob_load.model.m

    # calculate cable lengths based on initial configuration
    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]

    # println(d)

    # initial rollout
    for i = 1:num_lift
        # rollout!(prob_lift[i])
        solve!(prob_lift[i],opts_al)
    end
    # rollout!(prob_load)
    solve!(prob_load,opts_al)

    # return prob_lift, prob_load, 1, 1

    # initial rollout
    # for i = 1:num_lift
    #     # rollout!(prob_lift[i])
    #     for k = 1:N
    #         prob_lift[i].X[k][1:n_slack] = prob_lift_al[i].X[k][1:n_slack]
    #         prob_lift[i].X[k][4] = 1.0
    #     end
    #     prob_lift[i] = infeasible_problem(prob_lift[i],1.0e-6)
    # end
    #
    # for k = 1:N
    #     prob_load.X[k] = prob_load_al.X[k]
    #     k < N ? prob_load.U[k] = prob_load_al.U[k] : nothing
    # end

    # generate cable constraints
    X_lift = [deepcopy(prob_lift[i].X) for i = 1:num_lift]
    U_lift = [deepcopy(prob_lift[i].U) for i = 1:num_lift]

    X_load = deepcopy(prob_load.X)
    U_load = deepcopy(prob_load.U)

    if admm_type == :sequential || admm_type == :parallel
        cable_lift = [gen_lift_cable_constraints(X_load,
                        U_load,
                        i,
                        n_lift,
                        m_lift,
                        d[i],
                        n_slack) for i = 1:num_lift]

        cable_load = gen_load_cable_constraints(X_lift,U_lift,n_load,m_load,d,n_slack)

        self_col = [gen_self_collision_constraints(X_lift,i,n_lift,m_lift,r_lift,n_slack) for i = 1:num_lift]

        for i = 1:num_lift
            for k = 1:N
                prob_lift[i].constraints[k] += cable_lift[i][k]
                (k != 1 && k != N) ? prob_lift[i].constraints[k] += self_col[i][k] : nothing
            end
        end

        for k = 1:N
            prob_load.constraints[k] += cable_load[k]
        end
    end

    # create augmented Lagrangian problems, solvers
    solver_lift_al = []
    prob_lift_al = []
    for i = 1:num_lift
        solver = AbstractSolver(prob_lift[i],opts)
        prob = AugmentedLagrangianProblem(prob_lift[i],solver)

        push!(solver_lift_al,solver)
        push!(prob_lift_al,prob)
    end
    solver_load_al = AbstractSolver(prob_load,opts)
    prob_load_al = AugmentedLagrangianProblem(prob_load,solver_load_al)

    for ii = 1:opts.iterations
        # solve lift agents using iLQR
        for i = 1:num_lift
            solve_aula!(prob_lift_al[i],solver_lift_al[i])

            if admm_type == :sequential
                X_lift[i] .= prob_lift_al[i].X
                U_lift[i] .= prob_lift_al[i].U
            end
            copyto!(solver_lift_al[i].C_prev, solver_lift_al[i].C)
        end

        if admm_type == :parallel
            for i = 1:num_lift
                X_lift[i] .= prob_lift_al[i].X
                U_lift[i] .= prob_lift_al[i].U
            end
        end

        # solve load using iLQR
        solve_aula!(prob_load_al,solver_load_al)
        X_load .= prob_load_al.X
        U_load .= prob_load_al.U

        # return prob_lift_al, prob_load_al, solver_lift_al, solver_load_al
        if max([max_violation(solver_lift_al[i]) for i = 1:num_lift]...,max_violation(solver_load_al)) < opts.constraint_tolerance
            @info "ADMM problem solved"
            break
        else
            # for i = 1:num_lift
            #     reset!(solver_lift_al[i].solver_uncon)
            # end
            # reset!(solver_load_al.solver_uncon)
        end
    end

    return prob_lift_al, prob_load_al, solver_lift_al, solver_load_al
end

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
      # ẋ[4:7] = 0.5*qmult(q,[0;omega]) #quaternion derivative
      ẋ[4:7] = SVector(0.5*q*Quaternion(zero(x[1]), omega...))
      ẋ[8:10] = g + (1/m)*(q*F) + u_slack #acceleration in world frame
      ẋ[11:13] = Jinv*(tau - cross(omega,J*omega)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      return tau, omega, J, Jinv
end

quad_params = (m=0.5,
             J=SMatrix{3,3}(Diagonal([0.0023, 0.0023, 0.004])),
             Jinv=SMatrix{3,3}(Diagonal(1.0./[0.0023, 0.0023, 0.004])),
             gravity=SVector(0,0,-9.81),
             motor_dist=0.2,
             kf=1.0,
             km=0.0245)

quadrotor_lift = Model(quadrotor_lift_dynamics!, 13, 7, quad_params)

# Set up lift (3x) and load (1x) models
num_lift = 3
num_load = 1

n_slack = 3
n_lift = quadrotor_lift.n
m_lift = quadrotor_lift.m

function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
    u_slack1 = u[1:3]
    u_slack2 = u[4:6]
    u_slack3 = u[7:9]
    Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_slack1+u_slack2+u_slack3)
end

n_load = Dynamics.doubleintegrator3D.n
m_load = n_slack*num_lift
doubleintegrator3D_load = Model(double_integrator_3D_dynamics_load!,n_load,m_load)

# Robot sizes
r_lift = 0.2
r_load = 0.1

# Control limits for lift robots
u_lim_l = -Inf*ones(m_lift)
u_lim_u = Inf*ones(m_lift)
u_lim_l[1:4] .= 0.
u_lim_u[1:4] .= 9.81*(quad_params.m + 1.)/4.0
# x_lim_l = -Inf*ones(n_lift)
# x_lim_l[3] = 0.
bnd = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)#,x_min=x_lim_l)

# Obstacle constraints
r_cylinder = 0.85

_cyl = []

push!(_cyl,(5.,1.3,r_cylinder))
push!(_cyl,(5.,-1.3,r_cylinder))
# push!(_cyl,(5.,1.25,r_cylinder))
# push!(_cyl,(5.,-1.25,r_cylinder))
# push!(_cyl,(5.,1.5,r_cylinder))
# push!(_cyl,(5.,-1.5,r_cylinder))
# push!(_cyl,(5.,1.75,r_cylinder))
# push!(_cyl,(5.,-1.75,r_cylinder))

function cI_cylinder_lift(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 2*r_lift)
    end
end
obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

function cI_cylinder_load(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 2*r_load)
    end
end
obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)


n = quadrotor_lift.n
m = quadrotor_lift.m

n_load = Dynamics.doubleintegrator3D.n
m_load = Dynamics.doubleintegrator3D.m*n_slack
shift_ = zeros(n)
shift_[1:3] = [0.0;0.0;1.0]
scaling = 1.
x10 = zeros(n)
x10[4] = 1.
x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
x10 += shift_
x20 = zeros(n)
x20[4] = 1.
x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
x20 += shift_
x30 = zeros(n)
x30[4] = 1.
x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
x30 += shift_
xload0 = zeros(n_load)
xload0[1:3] += shift_[1:3]

xlift0 = [x10,x20,x30]

_shift = zeros(n)
_shift[1:3] = [10.0;0.0;0.0]

norm(xload0[1:3]-x10[1:3])
norm(xload0[1:3]-x20[1:3])
norm(xload0[1:3]-x30[1:3])

xloadf = zeros(n_load)
xloadf[1:3] = xload0[1:3] + _shift[1:3]
x1f = copy(x10) + _shift
x2f = copy(x20) + _shift
x3f = copy(x30) + _shift

xliftf = [x1f,x2f,x3f]

# xf_bnd_lift_u = [Inf*ones(n_lift) for i = 1:num_lift]
# xf_bnd_lift_l = [-Inf*ones(n_lift) for i = 1:num_lift]
#
# for i = 1:num_lift
#     xf_bnd_lift_u[i][1:10] = xliftf[i][1:10]
#     # xf_bnd_lift_u[i][8:10] = xliftf[i][1:3]
#     xf_bnd_lift_l[i][1:10] = xliftf[i][1:10]
#     # xf_bnd_lift_l[i][8:10] = xliftf[i][1:3]
# end
# xf_bnd = [BoundConstraint(n_lift,m_lift,x_min=xf_bnd_lift_l[i],x_max=xf_bnd_lift_u[i]) for i = 1:num_lift]

d1 = norm(xloadf[1:3]-x1f[1:3])
d2 = norm(xloadf[1:3]-x2f[1:3])
d3 = norm(xloadf[1:3]-x3f[1:3])

d = [d1, d2, d3]

# discretization
N = 51
dt = 0.1

# objective
q_diag = ones(n)
# q_diag[4:7] .= 2.
r_diag = ones(m)
r_diag[1:4] .= 1.0e-6
r_diag[5:7] .= 1.0e-6
Q_lift = [1.0e-2*Diagonal(q_diag), 1.0e-2*Diagonal(q_diag), 1.0e-2*Diagonal(q_diag)]
Qf_lift = [1000.0*Diagonal(q_diag), 1000.0*Diagonal(q_diag), 1000.0*Diagonal(q_diag)]
R_lift = Diagonal(r_diag)
Q_load = 0.0*Diagonal(I,n_load)
Qf_load = 0.0*Diagonal(I,n_load)
R_load = 1.0e-6*Diagonal(I,m_load)

obj_lift = [LQRObjective(Q_lift[i],R_lift,Qf_lift[i],xliftf[i],N) for i = 1:num_lift]
obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N)

# constraints
constraints_lift = []
for i = 1:num_lift
    con = Constraints(N)
    for k = 1:N-1
        con[k] += bnd + obs_lift
    end
    con[N] += goal_constraint(xliftf[i])
    push!(constraints_lift,copy(con))
end

constraints_load = Constraints(N)
for k = 1:N-1
    constraints_load[k] += obs_load
end
constraints_load[N] += goal_constraint(xloadf)

u_load = [0.;0.;-9.81/num_lift]

u_lift = zeros(m)
u_lift[1:4] .= 9.81*(quad_params.m + 1.)/12.
u_lift[5:7] = u_load
U0_lift = [u_lift for k = 1:N-1]
U0_load = [-1.0*[u_load;u_load;u_load] for k = 1:N-1]

# create problems
prob_lift = [Problem(quadrotor_lift,
                obj_lift[i],
                U0_lift,
                integration=:rk3,
                constraints=constraints_lift[i],
                x0=xlift0[i],
                xf=xliftf[i],
                N=N,
                dt=dt)
                for i = 1:num_lift]

prob_load = Problem(doubleintegrator3D_load,
                obj_load,
                U0_load,
                integration=:rk3,
                constraints=constraints_load,
                x0=xload0,
                xf=xloadf,
                N=N,
                dt=dt)

verbose=true
opts_ilqr = iLQRSolverOptions(verbose=verbose,iterations=500)
opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=10,
    penalty_scaling=2.0,
    penalty_initial=10.)

# opts_altro = ALTROSolverOptions{T}(verbose=verbose,
#     opts_al=opts_al,
#     R_inf=1.0e-8,
#     resolve_feasible_problem=false,
#     projected_newton=false)

@time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift,prob_load,n_slack,:sequential,opts_al)

# max_violation(slift_al[3])
# max_violation(sload_al)

# plift_al[1]
vis = Visualizer()
open(vis)
visualize_lift_system(vis,plift_al,pload_al,r_lift,r_load)

plot(plift_al[1].U,1:3)
plot(pload_al.U,1:3)
