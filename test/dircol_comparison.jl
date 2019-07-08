import TrajectoryOptimization: to_dvecs
const TO = TrajectoryOptimization
using LinearAlgebra
using PartedArrays
using SparseArrays
using ForwardDiff
using Plots
using Ipopt

function unpack(Z::Vector{<:Real}, part_z::NamedTuple)
    N, uN = size(part_z.X,2), size(part_z.U,2)
    X = [view(Z,part_z.X[:,k]) for k = 1:N]
    U = [view(Z,part_z.U[:,k]) for k = 1:uN]
    return X, U
end

function pack(X,U, part_z)
    n,m = length(X[1]), length(U[1])
    N, uN = length(X), length(U)
    Z = zeros(eltype(X[1]), N*n + uN*m)
    for k = 1:N
        Z[part_z.X[:,k]] = X[k]
        Z[part_z.U[:,k]] = U[k]
    end
    return Z
end

function get_rc(A::SparseMatrixCSC)
    row,col,inds = findnz(A)
    v = sortperm(inds)
    row[v],col[v]
end

function PartedArrays.create_partition(n::Int,m::Int,N::Int,uN=N-1)
    Nx = N*n
    Nu = uN*m
    Nz = Nx+Nu
    ind_x = zeros(Int,n,N)
    ind_u = zeros(Int,m,uN)
    ix = 1:n
    iu = n .+ (1:m)
    for k = 1:uN
        ind_x[:,k] = ix .+ (k-1)*(n+m)
        ind_u[:,k] = iu .+ (k-1)*(n+m)
    end
    if uN == N-1
        ind_x[:,N] = ix .+ (N-1)*(n+m)
    end
    return (X=ind_x, U=ind_u)
end

function gen_usrfun(model, cost, xf, N, dt)

    Q = cost.Q
    R = cost.R
    Qf = cost.Qf

    function eval_f(Z)
        X,U = unpack(Z, part_z)
        N = length(X)

        J = 0.0
        for k = 1:N-1
            J += (X[k]-xf)'Q*(X[k]-xf) + U[k]'R*U[k]
        end
        J = J/(N-1)
        J += (X[N] - xf)'Qf*(X[N] - xf)
        return J/2
    end

    function eval_g(Z::Vector{T},g) where T
        X,U = unpack(Z, part_z)
        N = length(X)
        n,m = length(X[1]), length(U[1])

        fVals = [zeros(T,n) for k = 1:N]
        for k = 1:N
            model.f(fVals[k], X[k], U[k])
        end
        g_ = reshape(g, n, N-1)
        for k = 1:N-1
            xm = 0.5*(X[k] + X[k+1]) + dt/8*(fVals[k] - fVals[k+1])
            um = 0.5*(U[k] + U[k+1])
            fValm = zero(X[k])
            model.f(fValm, xm, um)
            g_[:,k] = X[k] - X[k+1] + dt*(fVals[k] + 4fValm + fVals[k+1])/6
        end
    end

    eval_g2(g, Z) = eval_g(Z, g)

    function eval_grad_f(Z, grad_f)
        X,U = unpack(Z, part_z)
        N = length(X)
        n,m = length(X[1]), length(U[1])

        grad = reshape(grad_f, n+m, N)
        for k = 1:N-1
            grad[:,k] = [Q*(X[k] - xf); R*U[k]] ./ (N-1)
        end
        grad[:,N] = [Qf*(X[N] - xf); zeros(m)]
    end

    function eval_jac_g(Z, mode, row, col, val)
        if mode == :Structure
            copyto!(row, r)
            copyto!(col, c)
        else
            jac = zeros(p_colloc, length(Z))
            g = zeros(p_colloc)
            ForwardDiff.jacobian!(jac, eval_g2, g, Z)
            copyto!(val, jac[CartesianIndex.(r,c)])
        end
    end

    function jac_g_structure(n,m,N)
        p_colloc = (N-1)*n
        NN = N*(n+m)
        jac = spzeros(p_colloc, NN)

        n_blk = 2(n+m)n
        blk = reshape(1:n_blk, n, 2(n+m))
        off1 = 0
        off2 = 0
        off = 0
        for k = 1:N-1
            jac[off1 .+ (1:n), off2 .+ (1:2(n+m))] = blk .+ off
            off1 += n
            off2 += n+m
            off  += n_blk
        end
        return jac
    end


    jac_struct = jac_g_structure(n,m,N)
    r,c = get_rc(jac_struct)

    part_z = create_partition(n,m,N,N)

    return eval_f, eval_g, eval_grad_f, eval_jac_g
end


x0 = [0.;0.;0.]
xf = [0.;1.;0.]
N = 51
n,m = 3,2
U0 = ones(m,N)
p_colloc = (N-1)*n
NN = N*(n+m)

# New Method
model = Dynamics.car_model
costfun = Dynamics.car_costfun
prob = Problem(rk4(model), Objective(costfun,N), N=N, tf=3.)
initial_controls!(prob,U0)
X0 = rollout(prob)
dt = prob.dt

ilqr = iLQRSolverOptions()
res = solve(prob, ilqr)
plot()
plot_trajectory!(res.X)

# Old Method
method = :hermite_simpson
model, obj = Dynamics.pendulum
costfun = obj.cost
solver = Solver(model, ConstrainedObjective(obj), N=N)
X0 = to_dvecs(rollout(solver, U0))
dt = solver.dt

res, = solve(solver, U0)
plot(res.X)



# New Ipopt Functions
part_z = create_partition(n,m,N,N)
Z0 = pack(X0,to_dvecs(U0),part_z)
eval_f2, eval_g2, eval_grad_f2, eval_jac_g2 = gen_usrfun(model,costfun,xf,N,dt)
g2 = zeros(p_colloc)
nG = p_colloc*2(n+m)
row,col,val = zeros(nG), zeros(nG), zeros(nG)

eval_f2(Z0)
eval_g2(Z0,g2)
g2

eval_jac_g2(Z0,:Vals,row,col,val)

# Z_L, Z_U, G_L, G_U = TrajectoryOptimization.get_bounds(solver, method)
z_L = ones(NN)*-1.1e20
z_U = ones(NN)*1.1e20
g_L = zeros(p_colloc)
g_U = zeros(p_colloc)
z_L[1:n] = x0
z_U[1:n] = x0
z_L[(N-1)*(n+m) .+ (1:n)] = xf
z_U[(N-1)*(n+m) .+ (1:n)] = xf

problem = createProblem(NN, z_L, z_U, p_colloc, g_L, g_U, nG, 0,
    eval_f2, eval_g2, eval_grad_f2, eval_jac_g2)
problem.x = copy(Z0)

opt_file = joinpath(TrajectoryOptimization.root_dir(),"ipopt.opt");
addOption(problem,"option_file_name",opt_file)
solveProblem(problem)
Zsol = problem.x
Xsol,Usol = unpack(Zsol,part_z)
plot_trajectory!(Xsol)

# Try Ipopt again
prob = Problem(rk4(model), Objective(costfun,N), N=N, tf=3.)
prob = TrajectoryOptimization.update_problem(prob, model=model, constraints=Constraints(N))
prob.constraints[N] += goal_constraint(xf)
eval_f, eval_g, eval_grad_f, eval_jac_g = TrajectoryOptimization.gen_dircol_functions2(prob)
Z0
eval_f(Z0)
eval_f2(Z0)
eval_g2(Z0,g2)
eval_g(Z0,g)
g == g2

prob0 = copy(prob)
solver = TO.DIRCOLSolver(prob0)
bnds = TO.remove_bounds!(prob0)
z_U, z_L, g_U, g_L = TO.get_bounds(prob0, solver, bnds)

z_U
problem = createProblem(NN, z_L, z_U, p_colloc, g_L, g_U, nG, 0,
    eval_f, eval_g, eval_grad_f, eval_jac_g)
problem.x = Z0
opt_file = joinpath(TrajectoryOptimization.root_dir(),"ipopt.opt");
addOption(problem,"option_file_name",opt_file)
solveProblem(problem)

Zsol = problem.x
Xsol,Usol = unpack(Zsol,part_z)
plot_trajectory!(Xsol)


prob0 = copy(prob)
bnds = TrajectoryOptimization.remove_bounds!(prob0)
Z_u, Z_l, G_u, G_l = TrajectoryOptimization.get_bounds(prob, dircol, bnds)

opts = TrajectoryOptimization.DIRCOLSolverOptions{Float64}()
dircol = TrajectoryOptimization.DIRCOLSolver(prob, opts)
problem = solve!(prob, dircol)
solveProblem(problem)

problem = TrajectoryOptimization.gen_ipopt_prob(prob)
problem.x = copy(Z0)
solveProblem(problem)

Zsol = problem.x
Xsol,Usol = unpack(Zsol,part_z)
plot_trajectory!(Xsol)
