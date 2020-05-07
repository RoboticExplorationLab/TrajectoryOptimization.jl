using TrajectoryOptimization
import TrajectoryOptimization: stage_cost, cost_expansion
using TrajOptPlots
const TO = TrajectoryOptimization

# using PlanningWithAttitude
# import PlanningWithAttitude: cost_gradient, stage_cost, cost_hessian
# include("visualization.jl")
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Plots
using ForwardDiff
using Random
using MeshCat



# Dynamics
model = Dynamics.Satellite()
N = 101
tf = 5.0

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

# Objective
struct SatCost <: CostFunction
    Q::Diagonal{Float64,SVector{3,Float64}}
    R::Diagonal{Float64,SVector{3,Float64}}
    q::SVector{3,Float64}
    q0::SVector{4,Float64}
    Qq::Float64
    c::Float64
end

function SatCost(Q::Diagonal,R::Diagonal,q0::SVector, Qq=0.0, ω0=@SVector zeros(3))
    q = -Q*ω0
    c = 0.5*ω0'Q*ω0
    SatCost(Q,R,q,q0,Qq,c)
end

function stage_cost(cost::SatCost, x::SVector, u::SVector=(@SVector zeros(3)))
    ω = @SVector [x[1], x[2], x[3]]
    q = @SVector [x[4], x[5], x[6], x[7]]
    J = 0.5*(ω'cost.Q*ω + u'cost.R*u) + cost.q'ω + cost.c
    d = cost.q0'q
    if d ≥ 0
        J += cost.Qq*(1-d)
    else
        J += cost.Qq*(1+d)
    end
    return J
end


function cost_expansion(cost::SatCost, model::Dynamics.Satellite, z::KnotPoint{T,N,M}, G) where {T,N,M}
    Q = cost.Q
    q0 = cost.q0
    u = control(z)
    ω = @SVector [z.z[1], z.z[2], z.z[3]]
    q = @SVector [z.z[4], z.z[5], z.z[6], z.z[7]]
    d = cost.q0'q
    Qω = Q*ω + cost.q
    if d ≥ 0
        Qq = -cost.Qq*cost.q0'Lmult(q)*Vmat()'
    else
        Qq =  cost.Qq*cost.q0'Lmult(q)*Vmat()'
    end
    Qx = [Qω; Qq']
    Qu = cost.R*u

    Qxx = Diagonal(@SVector [Q[1,1], Q[2,2], Q[3,3], 0,0,0])
    d = q0'q*cost.Qq
    if d ≤ 0
        d *= -1
    end
    Qqq = I(3)*d
    Qxx = @SMatrix [
        Q[1,1] 0 0 0 0 0;
        0 Q[2,2] 0 0 0 0;
        0 0 Q[3,3] 0 0 0;
        0 0 0 Qqq[1] Qqq[4] Qqq[7];
        0 0 0 Qqq[2] Qqq[5] Qqq[8];
        0 0 0 Qqq[3] Qqq[6] Qqq[9];
    ]
    Quu = cost.R
    Qux = @SMatrix zeros(M,N-1)
    return Qxx, Quu, Qux, Qx, Qu
end

Q_diag = @SVector [1e-3,1e-3,1e-3, 1e-2,1e-2,1e-2,1e-2]
R_diag = @SVector fill(1e-4,3)
θs = range(0,2pi,length=N)
u = normalize([1,0,0])
iω = @SVector [1,2,3]
iq = @SVector [4,5,6,7]

Xref = map(1:N) do k
    θ = θs[k] #+ deg2rad(185)
    ωf = [2pi/5,0,0]
    qf = @SVector [cos(θ/2), u[1]*sin(θ/2), u[2]*sin(θ/2), u[3]*sin(θ/2)]
    xf = @SVector [ωf[1], ωf[2], ωf[3], qf[1], qf[2], qf[3], qf[4]]
end

costs = map(1:N) do k
    ωf = Xref[k][iω]
    qf = Xref[k][iq]
    k < N ? s = 1 : s = 1
    # LQRCost(Q_diag*s, R_diag, xf)
    SatCost(Diagonal(Q_diag[iω]), Diagonal(R_diag), qf, 1.0, ωf)
    # xf = Xref[k]
    # QuatLQRCost(Diagonal(Q_diag), Diagonal(R_diag), xf)
end
obj = Objective(costs)

# Initial Condition
x0 = @SVector [0,0,0, 1,0,0,0.]
u0 = @SVector [0.3, 0, 0]
Random.seed!(1)
U0 = [u0 + randn(3)*4e-1 for k = 1:N-1]

# Final Condition
θ = θs[end]
xf = @SVector [0,0,0, cos(θ/2), u[1]*sin(θ/2), u[2]*sin(θ/2), u[3]*sin(θ/2)]

# Solver
prob = Problem(model, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob)
initial_controls!(solver, U0)
rollout!(solver)
visualize!(vis, model, solver.Z)
cost(obj, solver.Z)

solver.opts.verbose = true
initial_controls!(solver, U0)

cost(solver)
TO.state_diff_jacobian!(solver.G, solver.model, solver.Z)
TO.discrete_jacobian!(solver.∇F, solver.model, solver.Z)
TO.cost_expansion!(solver.Q, solver.G, solver.obj, solver.model, solver.Z)
ΔV = TO.backwardpass!(solver)
TO.forwardpass!(solver, ΔV, cost(solver))
for k = 1:N
    solver.Z[k].z = solver.Z̄[k].z
end


initial_controls!(solver, U0)
solve!(solver)
solver.stats.iterations
states(solver)[end][4:7]'xf[4:7]
visualize!(vis, model, solver.Z)
Z = solver.Z

solver.opts.verbose = false
@btime begin
    initial_controls!($solver, $U0)
    solve!($solver)
end



############################################################################################
#                                 Intermediate Keyframes
############################################################################################

Q_nom = @SVector [1e-3,1e-3,1e-3, 1e-4,1e-4,1e-4,1e-4]
R_nom = @SVector fill(1e-4,3)

Q_key = @SVector [1e-3,1e-3,1e-3, 1e-2,1e-2,1e-2,1e-2]
R_key = @SVector fill(1e-4,3)

keyframes = Int.(range(1,N, length=3))
xf = @SVector [0,0,0, cos(θ/2), u[1]*sin(θ/2), u[2]*sin(θ/2), u[3]*sin(θ/2)]

costs = map(1:N) do k
    if k ∈ keyframes
        ωf = Xref[k][iω]
        qf = Xref[k][iq]
        SatCost(Diagonal(Q_key[iω]), Diagonal(R_key), qf, 1.0, ωf)
    else
        ωf = xf[iω]
        qf = xf[iq]
        SatCost(Diagonal(Q_nom[iω]), Diagonal(R_nom), qf, 0.0, ωf)
    end
end
obj = Objective(costs)

# Initial Condition
x0 = @SVector [0,0,0, 1,0,0,0.]
u0 = @SVector [0.3, 0, 0]
Random.seed!(1)
U0 = [u0 + randn(3)*4e-1 for k = 1:N-1]

# Solver
prob = Problem(model, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob)
initial_controls!(solver, U0)
rollout!(solver)
visualize!(vis, model, solver.Z)
cost(obj, solver.Z)

initial_controls!(solver, U0)
# initial_controls!(solver, controls(Z_sol))
solver.opts.verbose = true
solve!(solver)
solver.stats.iterations
states(solver)[end][4:7]'xf[4:7]
visualize!(vis, model, solver.Z)
cost(solver)
Z = solver.Z

Zsol = deepcopy(Z)

iq = @SVector [4,5,6,7]
n =  13
function to_diag(iq,::Val{N}) where N
    D = @MVector zeros(N)
end
@btime to_diag($iq, Val($n))
@btime @MVector zeros(3)
