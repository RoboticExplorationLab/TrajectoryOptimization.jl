using TrajectoryOptimization

"""@(SIGNATURES) Time Varying Discrete Linear Quadratic Regulator (TVLQR)"""
function lqr(A::Array{Float64,3}, B::Array{Float64,3}, Q::AbstractArray{Float64,2}, R::AbstractArray{Float64,2}, Qf::AbstractArray{Float64,2})::Array{Float64,3}
    n,m,N = size(B)
    N += 1
    K = zeros(m,n,N-1)
    S = zeros(n,n)

    # Boundary conditions
    S .= Qf
    for k = 1:N-1
        # Compute control gains
        K[:,:,k] = (R + B[:,:,k]'*S*B[:,:,k])\(B[:,:,k]'*S*A[:,:,k])
        # Calculate cost-to-go for backward propagation
        S .= Q + A[:,:,k]'*S*A[:,:,k] - A[:,:,k]'*S*B[:,:,k]*K[:,:,k]
    end
    K
end

function lqr(results::SolverResults, solver::Solver)
    n, m, N = get_sizes(solver)
    Q, R, Qf, xf = get_cost_matrices(solver)
    A, B = results.fx, results.fu
    lqr(A,B,Q,R,Qf)
end

"""@(SIGNATURES) Simulate discrete system using LQR tracker"""
function simulate_lqr_tracking(fd::Function,X::Matrix,U::Matrix,K::Array{Float64,3})::Tuple{Array{Float64,2},Array{Float64}}
    # get state, control, horizon dimensions
    m, n, N = size(K)
    N += 1

    # allocate memory for state and control trajectories
    X_ = zero(X)
    X_[:,1] = X[:,1]

    U_ = zero(U)

    for k = 1:N-1
        U_[:,k] = -K[:,:,k]*(X_[:,k] - X[:,k]) + U[:,k]
        fd(view(X_,:,k+1), X_[:,k], U_[:,k])
    end
    X_, U_
end

# Pendulum test case
n = 2
m = 1
model, obj = TrajectoryOptimization.Dynamics.pendulum!
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true

obj.Qf = 30.0*Diagonal(I,n)
obj.Q = 0.1*Diagonal(I,n)
obj.R = 0.01*Diagonal(I,m)
obj.x0 = [0.0; 0.0]
obj.xf = [pi; 0.0]
obj.tf = 5.0
dt = 0.1

# ilQR solve
solver = TrajectoryOptimization.Solver(model,obj,dt=dt,opts=opts)
U = zeros(solver.model.m, solver.N)
results, = TrajectoryOptimization.solve(solver,U)

# LQR tracker
K = lqr(results,solver)
X, U = simulate_lqr_tracking(solver.fd,results.X,results.U,K)

# using Plotly
# plot(results.X)
# plot(X)
