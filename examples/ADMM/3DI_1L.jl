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

    # intialize system
    for i = 1:num_lift
        setobject!(vis["lift$i"],HyperSphere(Point3f0(0), convert(Float32,r_lift)) ,MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))

        cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d[i]),convert(Float32,0.01))
        setobject!(vis["cable"]["$i"],cable,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    end
    setobject!(vis["load"],HyperSphere(Point3f0(0), convert(Float32,r_load)) ,MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))

    addcylinders!(vis,_cyl,3.)

    anim = MeshCat.Animation(24)
    for k = 1:prob_lift[1].N
        MeshCat.atframe(anim,vis,k) do frame
            # cables
            x_load = prob_load.X[k][1:n_slack]
            for i = 1:num_lift
                x_lift = prob_lift[i].X[k][1:n_slack]
                settransform!(frame["cable"]["$i"], cable_transform(x_lift,x_load))
                settransform!(frame["lift$i"], Translation(x_lift...))
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
                c[1:n_slack] = u[n_slack .+ (1:n_slack)] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
            else
                c[1] = norm(x[1:n_slack] - X_load[k][1:n_slack])^2 - d^2
                if k < N
                    c[1 .+ (1:n_slack)] = u[n_slack .+ (1:n_slack)] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
                end
            end
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            if k == 1
                C[1:n_slack,(n+n_slack) .+ (1:n_slack)] = Is
            else
                C[1,1:n_slack] = 2*dif
                if k < N
                    C[1 .+ (1:n_slack),(n+n_slack) .+ (1:n_slack)] = Is
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
                    c[_shift .+ (1:n_slack)] = U_lift[i][k][n_slack .+ (1:n_slack)] + u[(i-1)*n_slack .+ (1:n_slack)]
                    _shift += n_slack
                end
            else
                for i = 1:num_lift
                    c[i] = norm(X_lift[i][k][1:n_slack] - x[1:n_slack])^2 - d[i]^2
                end

                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        c[_shift .+ (1:n_slack)] = U_lift[i][k][n_slack .+ (1:n_slack)] + u[(i-1)*n_slack .+ (1:n_slack)]
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

function gen_self_collision_constraints(X_lift,agent,n,m,sep,n_slack=3)
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
    d = [norm(prob_lift[i].x0 - prob_load.x0) for i = 1:num_lift]

    # initial rollout
    for i = 1:num_lift
        # rollout!(prob_lift[i])
        solve!(prob_lift[i],opts_al)
    end
    rollout!(prob_load)

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

        self_col = [gen_self_collision_constraints(X_lift,i,n_lift,m_lift,2*r_lift,n_slack) for i = 1:num_lift]

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
            solve!(prob_lift_al[i],solver_lift_al[i].solver_uncon)
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
        solve!(prob_load_al,solver_load_al.solver_uncon)
        X_load .= prob_load_al.X
        U_load .= prob_load_al.U

        # update lift agents: constraints, dual update, penalty update
        for i = 1:num_lift
            update_constraints!(prob_lift_al[i].obj.C,prob_lift_al[i].obj.constraints, prob_lift_al[i].X, prob_lift_al[i].U)
            update_active_set!(prob_lift_al[i].obj)
            cost(prob_lift_al[i])

            dual_update!(prob_lift_al[i], solver_lift_al[i])
            penalty_update!(prob_lift_al[i], solver_lift_al[i])
            copyto!(solver_lift_al[i].C_prev,solver_lift_al[i].C)
        end

        # update load: constraints, dual update, penalty update
        update_constraints!(prob_load_al.obj.C,prob_load_al.obj.constraints, prob_load_al.X, prob_load_al.U)
        update_active_set!(prob_load_al.obj)
        cost(prob_load_al)

        dual_update!(prob_load_al, solver_load_al)
        penalty_update!(prob_load_al, solver_load_al)
        copyto!(solver_load_al.C_prev, solver_load_al.C)

        if max([max_violation(solver_lift_al[i]) for i = 1:num_lift]...,max_violation(solver_load_al)) < opts.constraint_tolerance
            @info "ADMM problem solved"
            break
        else
            for i = 1:num_lift
                reset!(solver_lift_al[i].solver_uncon)
            end
            reset!(solver_load_al.solver_uncon)
        end
    end

    return prob_lift_al, prob_load_al, solver_lift_al, solver_load_al
end

# Set up lift (3x) and load (1x) models
num_lift = 3
num_load = 1

n_slack = 3
n_lift = Dynamics.doubleintegrator3D.n
m_lift = Dynamics.doubleintegrator3D.m + n_slack

function double_integrator_3D_dynamics_lift!(ẋ,x,u) where T
    u_input = u[1:3]
    u_slack = u[4:6]
    Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_input+u_slack)
end

doubleintegrator3D_lift = Model(double_integrator_3D_dynamics_lift!,n_lift,m_lift)

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
r_lift = 0.1
r_load = 0.1

# Control limits for lift robots
u_lim = Inf*ones(m_lift)
u_lim[1:3] .= 20.
bnd1 = BoundConstraint(n_lift,m_lift,u_min=-1.0*u_lim,u_max=u_lim)

x_lim_lift_l = -Inf*ones(n_lift)
x_lim_lift_l[3] = 0.
bnd2 = BoundConstraint(n_lift,m_lift,u_min=-1.0*u_lim,u_max=u_lim,x_min=x_lim_lift_l)
# bnd2 = BoundConstraint(n_lift,m_lift,x_min=x_lim_lift_l)

x_lim_load_l = -Inf*ones(n_load)
x_lim_load_l[3] = 0.
bnd3 = BoundConstraint(n_load,m_load,x_min=x_lim_load_l)


# Obstacle constraints
# r_cylinder = 0.5
r_cylinder = 0.75

_cyl = []
# l1 = 6

push!(_cyl,(5.,1.,r_cylinder))
push!(_cyl,(5.,-1.,r_cylinder))
# for i = range(4.75,stop=5.25,length=l1)
#     push!(_cyl,(i, 1.25,r_cylinder))
# end
# for i = range(4.75,stop=5.25,length=l1)
#     push!(_cyl,(i, -1.25,r_cylinder))
# end
#
# for i = range(4.75,stop=5.25,length=l1)
#     push!(_cyl,(i, 1.,r_cylinder))
# end
# for i = range(4.75,stop=5.25,length=l1)
#     push!(_cyl,(i, -1.,r_cylinder))
# end
#
# for i = range(4.75,stop=5.25,length=l1)
#     push!(_cyl,(i, .75,r_cylinder))
# end
# for i = range(4.75,stop=5.25,length=l1)
#     push!(_cyl,(i, -.75,r_cylinder))
# end


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


# initial state
scaling = 1.

shift_ = zeros(n_lift)
shift_[1:3] = [0.0;0.0;1.]
x10 = zeros(n_lift)
x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
x10 += shift_
x20 = zeros(n_lift)
x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
x20 += shift_
x30 = zeros(n_lift)
x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
x30 += shift_
xload0 = zeros(n_load)
xload0 += shift_

xlift0 = [x10, x20, x30]

# norm(xload0[1:3]-x10[1:3])
# norm(xload0[1:3]-x20[1:3])
# norm(xload0[1:3]-x30[1:3])

# goal state
_shift = zeros(n_lift)
_shift[1:3] = [10.;0.0;0.0]

x1f = x10 + _shift
x2f = x20 + _shift
x3f = x30 + _shift
xloadf = xload0 + _shift

xliftf = [x1f, x2f, x3f]

d1 = norm(xloadf[1:3]-x1f[1:3])
d2 = norm(xloadf[1:3]-x2f[1:3])
d3 = norm(xloadf[1:3]-x3f[1:3])

d = [d1, d2, d3]

# discretization
N = 21
dt = 0.1

# objective
Q_lift = 1.0e-2*Diagonal(I,n_lift)
Qf_lift = 1000.0*Diagonal(I,n_lift)
R_lift = 1.0e-4*Diagonal(I,m_lift)

Q_load = 1.0e-2*Diagonal(I,n_load)
Qf_load = 1000.0*Diagonal(I,n_load)
R_load = 1.0e-6*Diagonal(I,m_load)

obj_lift = [LQRObjective(Q_lift,R_lift,Qf_lift,xliftf[i],N) for i = 1:num_lift]
obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N)

# constraints
constraints_lift = []
for i = 1:num_lift
    con = Constraints(N)
    for k = 1:N-1
        con[k] += bnd2 + obs_lift
    end
    # con[N] += goal_constraint(xliftf[i])
    push!(constraints_lift,copy(con))
end

constraints_load = Constraints(N)
for k = 1:N-1
    constraints_load[k] += bnd3 + obs_load
end
# constraints_load[N] += goal_constraint(xloadf)


u_ = [0.;0.;9.81 + 9.81/num_lift;0.;0.;-9.81/num_lift]
u_load = [0.;0.;9.81/num_lift;0.;0.;9.81/num_lift;0.;0.;9.81/num_lift]
U0_lift = [u_ for k = 1:N-1]
U0_load = [u_load for k = 1:N-1]

# create problems
prob_lift = [Problem(doubleintegrator3D_lift,
                obj_lift[i],
                U0_lift,
                integration=:midpoint,
                constraints=constraints_lift[i],
                x0=xlift0[i],
                xf=xliftf[i],
                N=N,
                dt=dt)
                for i = 1:num_lift]

prob_load = Problem(doubleintegrator3D_load,
                obj_load,
                U0_load,
                integration=:midpoint,
                constraints=constraints_load,
                x0=xload0,
                xf=xloadf,
                N=N,
                dt=dt)

verbose=true
opts_ilqr = iLQRSolverOptions(verbose=verbose)
opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=50,
    penalty_scaling=2.0,
    penalty_initial=1.0)

plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift,prob_load,n_slack,:sequential,opts_al)

max_violation(slift_al[1])
max_violation(plift_al[1])

plift_al[1]
vis = Visualizer()
open(vis)
visualize_lift_system(vis,prob_lift,prob_load,r_lift,r_load)
#
# plot(prob_lift[1].U,4:6)
