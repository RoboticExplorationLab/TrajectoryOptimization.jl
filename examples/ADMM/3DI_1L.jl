using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO

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
    Dynamics.double_integrator_3D_dynamics!(ẋ,x,-u_slack1-u_slack2-u_slack3)
end

n_load = Dynamics.doubleintegrator3D.n
m_load = n_slack*num_lift
doubleintegrator3D_load = Model(double_integrator_3D_dynamics_load!,n_load,m_load)

# Robot sizes
r_lift = 0.2
r_load = 0.2

# Control limits for lift robots
u_lim = Inf*ones(m_lift)
u_lim[1:3] .= 15.
bnd = BoundConstraint(n_lift,m_lift,u_min=-1.0*u_lim,u_max=u_lim)

# Obstacle constraints
r_cylinder = 0.5
_cyl = []
l1 = 3

for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, .75,r_cylinder))
end
for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, -.75,r_cylinder))
end
for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, 1.,r_cylinder))
end
for i = range(4,stop=5,length=l1)
    push!(_cyl,(i, -1.,r_cylinder))
end

function cI_cylinder_lift(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + r_lift)
    end
end
obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

function cI_cylinder_load(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + r_load)
    end
end
obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)


# initial state
scaling = 1.

shift_ = zeros(n_lift)
shift_[1:3] = [0.0;0.0;1.0]
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
_shift[1:3] = [10.0;0.0;0.0]

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
N = 51
dt = 0.1

# objective
Q_lift = 1.0*Diagonal(I,n_lift)
Qf_lift = 1.0*Diagonal(I,n_lift)
R_lift = 1.0e-1*Diagonal(I,m_lift)

Q_load = 1.0*Diagonal(I,n_load)
Qf_load = 1.0*Diagonal(I,n_load)
R_load = 1.0e-1*Diagonal(I,m_load)

obj_lift = [LQRObjective(Q_lift,R_lift,Qf_lift,xliftf[i],N) for i = 1:num_lift]
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

# create problems
prob_lift = [Problem(doubleintegrator3D_lift,
                obj_lift[i],
                [0.01*rand(m_lift) for k = 1:N-1],
                integration=:rk3,
                constraints=constraints_lift[i],
                x0=xlift0[i],
                xf=xliftf[i],
                N=N,
                dt=dt)
                for i = 1:num_lift]

prob_load = Problem(doubleintegrator3D_load,
                obj_load,
                [ones(m_load) for k = 1:N-1],
                integration=:rk3,
                constraints=constraints_load,
                x0=xload0,
                xf=xloadf,
                N=N,
                dt=dt)


function gen_lift_cable_constraints(X_load,U_load,agent,n,m,d,n_slack=3)
    N = length(X_load)
    con_cable_lift = []
    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            c[1] = norm(x[1:n_slack] - X_load[k][1:n_slack])^2 - d^2
            # if k < N
            #     c[1 .+ (1:n_slack)] = u[n_slack .+ (1:n_slack)] - U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
            # end
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            C[1,1:n_slack] = 2*dif
            # if k < N
            #     C[1 .+ (1:n_slack),(n+n_slack) .+ (1:n_slack)] = Is
            # end
        end
        # k < N ? p_con = 1+n_slack : p_con = 1
        p_con = 1
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
            for i = 1:num_lift
                c[i] = norm(X_lift[i][k][1:n_slack] - x[1:n_slack])^2 - d[i]^2
            end

            # if k < N
            #     _shift = num_lift
            #     for i = 1:num_lift
            #         c[_shift .+ (1:n_slack)] = U_lift[i][k][n_slack .+ (1:n_slack)] - u[(i-1)*n_slack .+ (1:n_slack)]
            #         _shift += n_slack
            #     end
            # end
        end

        function ∇con(C,x,u=zeros())
            for i = 1:num_lift
                x_pos = X_lift[i][k][1:n_slack]
                x_load_pos = x[1:n_slack]
                dif = x_pos - x_load_pos
                C[i,1:n_slack] = -2*dif
            end
            # if k < N
            #     _shift = num_lift
            #     for i = 1:num_lift
            #         u_idx = ((i-1)*n_slack .+ (1:n_slack))
            #         C[_shift .+ (1:n_slack),n .+ u_idx] = -Is
            #         _shift += n_slack
            #     end
            # end
        end
        # k < N ? p_con = num_lift*(1 + n_slack) : p_con = num_lift
        p_con = num_lift
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_load)
        push!(con_cable_load,cc)
    end

    return con_cable_load
end

function solve_admm!(prob_lift,prob_load,n_slack,opts)
    admm_type = :sequential

    num_lift = length(prob_lift)
    n_lift = prob_lift[1].model.n
    m_lift = prob_lift[1].model.m
    n_load = prob_load.model.n
    m_load = prob_load.model.m

    # calculate cable lengths based on initial configuration
    d = [norm(prob_lift[i].x0 - prob_load.x0) for i = 1:num_lift]

    println(d)

    # generate cable constraints
    X_lift = [deepcopy(prob_lift[i].X) for i = 1:num_lift]
    U_lift = [deepcopy(prob_lift[i].U) for i = 1:num_lift]

    X_load = deepcopy(prob_load.X)
    U_load = deepcopy(prob_load.U)

    cable_lift = [gen_lift_cable_constraints(X_load,
                    U_load,
                    i,
                    n_lift,
                    m_lift,
                    d[i],
                    n_slack) for i = 1:num_lift]

    cable_load = gen_load_cable_constraints(X_lift,U_lift,n_load,m_load,d,n_slack)

    for i = 1:num_lift
        for k = 1:N
            prob_lift[i].constraints[k] += cable_lift[i][k]
        end
    end

    for k = 1:N
        prob_load.constraints[k] += cable_load[k]
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

    cost(prob_lift_al[1])
    error("hi")
    for ii = 1:opts.iterations
        # solve lift agents using iLQR
        for i = 1:num_lift
            solve!(prob_lift_al[i],solver_lift_al[i].solver_uncon)
            if admm_type == :sequential
                X_lift[i] .= prob_lift_al[i].X
                U_lift[i] .= prob_lift_al[i].U
            end
            # reset!(solver_lift_al[i].solver_uncon)
        end

        # if admm_type == :parallel
        #     for i = 1:num_lift
        #         X_lift[i] .= prob_lift_al[i].X
        #         U_lift[i] .= prob_lift_al[i].U
        #     end
        # end

        # solve load using iLQR
        solve!(prob_load_al,solver_load_al.solver_uncon)
        X_load .= prob_load_al.X
        U_load .= prob_load_al.U
        # reset!(solver_load.solver_uncon)

        # update lift agents: constraints, dual update, penalty update
        for i = 1:num_lift
            update_constraints!(prob_lift_al[i].obj.C,prob_lift_al[i].obj.constraints, prob_lift_al[i].X, prob_lift_al[i].U)
            update_active_set!(prob_lift_al[i].obj)
            # cost(prob_lift_al[i])

            dual_update!(prob_lift_al[i], solver_lift_al[i])
            penalty_update!(prob_lift_al[i], solver_lift_al[i])
            copyto!(solver_lift_al[i].C_prev,solver_lift_al[i].C)
        end

        # update load: constraints, dual update, penalty update
        update_constraints!(prob_load_al.obj.C,prob_load_al.obj.constraints, prob_load_al.X, prob_load_al.U)
        update_active_set!(prob_load_al.obj)
        # cost(prob_load_al)

        dual_update!(prob_load_al, solver_load_al)
        penalty_update!(prob_load_al, solver_load_al)
        copyto!(solver_load_al.C_prev, solver_load_al.C)

        if max([max_violation(prob_lift_al[i]) for i = 1:num_lift]...,max_violation(prob_load)) < opts.constraint_tolerance
            @info "ADMM problem solved"
            break
        end
    end

    return prob_lift_al, prob_load_al
end

solve_admm!(prob_lift,prob_load,n_slack,AugmentedLagrangianSolverOptions{Float64}())

rollout!(prob_load)
plot(prob_load.X)

X_load = deepcopy(prob_load.X)
U_load = deepcopy(prob_load.U)
ccc = gen_lift_cable_constraints(X_load,U_load,1,n_lift,m_lift,d[1],n_slack)

cc1 = zeros(1+n_slack)
cc2 = zeros(1+n_slack)
cc3 = zeros(1+n_slack)
xx = rand(n_lift)
uu = rand(m_lift)

ccc[1].c(cc1,xx,uu)
ccc[2].c(cc2,xx,uu)
ccc[N-1].c(cc2,xx,uu)
ccc[N].c(cc3,xx,uu)

norm(xx - X_load[N])^2

prob_load.X[1] .= 0
ccc[1].c(cc1,xx,uu)

X_ll = [prob_lift[i].X for i = 1:num_lift]
U_ll = [prob_lift[i].U for i = 1:num_lift]

ddd = gen_load_cable_constraints(X_ll,U_ll,n_load,m_load,d,n_slack)


dd1 = zeros(num_lift*(1+n_slack))
dd2 = zeros(num_lift*(1+n_slack))
dd3 = zeros(num_lift*(1+n_slack))
xx = rand(n_load)
uu = rand(m_load)

ddd[1].c(dd1,xx,uu)
ddd[2].c(dd2,xx,uu)
ddd[3].c(dd3,xx,uu)

dd1
dd2
dd3


X_ll[1][1] .= 0
ddd[1].c(dd1,xx,uu)
dd1
# solve!(prob_lift[1],ALTROSolverOptions{Float64}())
#
# using Plots
#
# plot(prob_lift[1].X)


# prob_lift[1].constraints[1]


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

function visualize_lift_system(vis,prob_lift,prob_load,n_slack=3)
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

vis = Visualizer()
open(vis)
visualize_lift_system(vis,prob_lift,prob_load)
