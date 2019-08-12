import TrajectoryOptimization.Dynamics: Quaternion, Lmult
using Random, StaticArrays
using CoordinateTransformations, MeshCat, GeometryTypes, Colors
using FileIO

function animate_free_body(vis,prob)

    x0 = @SVector [0., 0, 0.0]
    widths = @SVector [1., 1, 3]
    offset = @SVector [0,0,2]
    robot_obj = HyperRectangle(Vec(x0), Vec(widths))
    robot = vis["robot"]
    setobject!(vis["robot"]["free_body"],robot_obj,MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0)));
    settransform!(vis["robot"]["free_body"], Translation(-widths/2))

    # settransform!(vis["/Cameras/default"], compose(Translation(0., 72., 60.),LinearMap(RotX(pi/7.5)*RotZ(pi/2))))
    anim = MeshCat.Animation(24)
    for i = 1:prob.N
        MeshCat.atframe(anim,vis,i) do frame
            settransform!(frame["robot"], compose(Translation(offset),LinearMap(Quat(prob.X[i][4:7]...))))
        end
    end
    MeshCat.setanimation!(vis,anim)
end
vis = Visualizer()
open(vis)

model = Dynamics.free_body_model
n,m = model.n, model.m

Random.seed!(1)
x0 = [zeros(3); normalize(rand(4))]
qf = Quaternion(RotX(pi/4)*RotZ(pi/4))
xf = zeros(n)
xf[4] = 1
xf[4:7] = SVector(qf)

Q = Diagonal(I,n)*1e-2
R = Diagonal(I,m)*1e-1
Qf = Diagonal(I,n)*1.0

N = 31
tf = 5.0
costfun = QuadraticCost(Q,R)
costfun_term = QuadraticCost(Qf,zeros(n), 0.)
obj = Objective(costfun, costfun_term, N)
obj = LQRObjective(Q,R,Qf,xf,N)

prob = Problem(rk3(model), obj, x0=x0, xf=xf, N=N, tf=tf)
initial_controls!(prob, zeros(m,N))

opts_ilqr = iLQRSolverOptions()
@time sol, = solve(prob,opts_ilqr)
animate_free_body(vis,sol)


model_quat = Model(Dynamics.free_body_dynamics!,n,m,Dynamics.free_body_params,Dict{Symbol,Any}(:quat=>4:7))
@test TO.has_quat(model_quat)
Gf = TO.state_diff_jacobian(model_quat, xf)
Q_diag = diag(Q)[1:7 .!= 4]
Q_quat = Gf'Diagonal(Q_diag)*Gf
Qf_diag = diag(Qf)[1:7 .!= 4]
Qf_quat = Gf'Diagonal(Qf_diag)*Gf
costfun = QuadraticCost(Q_quat,R)
costfun_term = QuadraticCost(Qf_quat,zeros(n), 0.)
obj_quat = Objective(costfun, costfun_term, N)

prob_quat = update_problem(prob, obj=obj_quat, model=rk3(model_quat))
@test TO.has_quat(prob_quat.model)
@time sol_quat, = solve(prob_quat, opts_ilqr)
animate_free_body(vis,sol_quat)

@btime solve($prob, $opts_ilqr)
@btime solve($prob_quat, $opts_ilqr)



prob = copy(Problems.quadrotor)

max_con_viol = 1e-6
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-8,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3)

# @time sol, = solve(prob, opts_altro)
@time sol, = solve(prob, opts_ilqr)
@time sol_con, = solve(prob, opts_al)


T = Float64

# model
d = Dict{Symbol,Any}(:quat=>4:7)
model = Model(Dynamics.quadrotor_dynamics!,13,4,quad_params,d)
@test TO.has_quat(model)
model_d = rk3(model)
@test TO.has_quat(model_d)

prob_quat = update_problem(prob, model=model_d)
@test TO.has_quat(prob_quat.model)
@time sol_quat, = solve(prob_quat, opts_ilqr)
@time sol_quat_con, = solve(prob_quat, opts_al)
max_violation(sol_quat_con)

vis_quad = Visualizer()
open(vis_quad)

@btime solve($prob, $opts_ilqr)
@btime solve($prob_quat, $opts_ilqr)

animate_quadrotor_line(vis_quad, sol)
animate_quadrotor_line(vis_quad, sol_quat)

animate_quadrotor_line(vis_quad, sol_con)
animate_quadrotor_line(vis_quad, sol_quat_con)


# Quadrotor Maze
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-8,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4)

prob_altro = copy(Problems.quadrotor_maze)
prob_altro.model.quat
@time p1, s1 = solve(prob_altro, opts_altro)

prob_quat = update_problem(prob_altro, model=model_d)
@time p1_quat, s1_quat = solve(prob_quat, opts_altro)

animate_quadrotor_maze(vis_quad, p1)
animate_quadrotor_maze(vis_quad, p1_quat)


settransform!(vis_quad["/Cameras/default"], compose(Translation(0., 72., 60.),LinearMap(RotX(pi/7.5)*RotZ(pi/2))))
