using Ipopt
using ForwardDiff

model, obj0 = Dynamics.cartpole_analytical
obj = copy(obj0)
n,m = model.n, model.m
dt = 0.1

# Set up problem
method = :trapezoid
solver = Solver(model,ConstrainedObjective(obj),dt=dt,integration=:rk3_foh)
N,N_ = get_N(solver,method)
NN = N*(n+m)
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)
Z = packZ(X0,U0)

# SNOPT method
results = DircolResults(get_sizes(solver)...,method)
results.Z .= Z
update_derivatives!(solver,results,method)
get_traj_points!(solver,results,method)
get_traj_points_derivatives!(solver,results,method)
J_snopt = cost(solver,results)
g_snopt = collocation_constraints(solver,results,method)
grad_f_snopt = cost_gradient(solver,results,method)

struct DircolVars
    Z::Vector{Float64}
    X::SubArray{Float64}
    U::SubArray{Float64}
end

function DircolVars(Z::Vector,n::Int,m::Int,N::Int)
    z = reshape(Z,n+m,N)
    X = view(z,1:n,:)
    U = view(z,n+1:n+m,:)
    DircolVars(Z,X,U)
end

function init_traj_points(solver,X,U,fVal,method)
    N,N_ = get_N(solver,method)
    if method == :trapezoid || method == :hermite_simpson_separated
        X_,U_,fVal_ = X,U,fVal
    else
        X_,U_,fVal_ = zeros(n,N_),zeros(m,N_),zeros(n,N_)
    end
    return X_,U_,fVal_
end

function get_traj_points(solver,X,U,gfVal,gX_,gU_,method::Symbol,cost_only::Bool=false)
    if !cost_only
        update_derivatives!(solver,X,U,gfVal)
    end
    if method == :trapezoid || method == :hermite_simpson_separated
        X_,U_ = X,U
    else
        if cost_only
            update_derivatives!(solver,X,U,gfVal)
        end
        get_traj_points!(solver,X,U,gX_,gU_,gfVal,method)
        X_,U_ = gX_,gU_
    end
    return X_, U_
end



#################
# COST FUNCTION #
#################

fVal = zeros(X0)
N,N_ = get_N(solver,method)
gX_,gU_,fVal_ = init_traj_points(solver,X0,U0,fVal,method)
weights = get_weights(method,N_)

function eval_f(Z)
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X, vars.U
    X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method,true)
    cost(solver,X_,U_,weights)
end

function eval_f2(Z)
    X,U = unpackZ(Z)
    cost(solver,X,U,weights)
end

function eval_f3(Z)
    results.Z .= Z
    vars = DircolVars(Z,n,m,N)
    update_derivatives!(solver,results,method)
    get_traj_points!(solver,results,method)
    cost(solver,results)
end

@btime eval_f(Z) # == J_snopt
# @btime eval_f2(Z)
@btime eval_f3(Z)


###########################
# COLLOCATION CONSTRAINTS #
###########################

g0 = zeros((N-1)*n)
g1 = zeros((N-1)*n)
g2 = zeros(g0)

function eval_g(Z, g)
    X,U = unpackZ(Z)
    g = reshape(g,n,N-1)
    f1, f2 = zeros(n),zeros(n)
    for k = 1:N-1
        solver.fc(f1,X[:,k+1],U[:,k+1])
        solver.fc(f2,X[:,k],U[:,k])
        g[:,k] = dt*( f1 + f2 )/2 - X[:,k+1] + X[:,k]
    end
    # reshape(g,n*(N-1),1)
    return nothing
end

function eval_g1(Z, g)
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X,vars.U
    X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
    get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
    collocation_constraints!(solver::Solver, X_, U_, fVal_, g, method::Symbol)
    # reshape(g,n*(N-1),1)
    return nothing
end

function eval_g2(Z, g)
    results.Z .= Z
    update_derivatives!(solver,results,method)
    get_traj_points!(solver,results,method)
    get_traj_points_derivatives!(solver,results,method)
    g .= collocation_constraints(solver::Solver,results, method::Symbol)
    # reshape(g,n*(N-1),1)
    return nothing
end

# @btime eval_g(Z,g0)
@btime eval_g1(Z,g1)
@btime eval_g2(Z,g2)
# g0 == g1
g2 == g1



#################
# COST GRADIENT #
#################

grad_f = zeros(NN)

function eval_grad_f(Z, grad_f)
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X, vars.U
    grad_f = reshape(grad_f, n+m, N)
    for k = 1:N
        grad_f[1:n,k] = weights[k]*Q*X[:,k]
        grad_f[n+1:end,k] = weights[k]*R*U[:,k]
    end
    grad_f[1:n,N] += Qf*X[:,N]
    return nothing
end

function eval_jac_g(x, mode, rows, cols, vals)
    z = reshape(x,n+m,N)
    if mode == :Structure
        ind = 1
        for k = 1:N-1 # loop over blocks
            for j = 1:2(n+m) # loop over columns of each block
                for i = 1:n # loop over rows of each block
                    row[ind] = i + (k-1)*n
                    col[ind] = j + (k-1)*(n+m)
                    ind += 1
                end
            end
        end
    else
        n_blk = n*2(n+m)
        n_blk2 = n*(n+m)
        fz = F(z[:,1])
        vals[1:n_blk2] .= vec(fz+eye(n,n+m))
        for k = 2:N-1
            fz = F(z[:,k])
            vals[(1:n_blk2) + (k-1)*n_blk - n_blk2] .= vec(fz+eye(n,n+m))
            vals[(1:n_blk2) + (k-1)*n_blk] .= vec(fz-eye(n,n+m))
        end
        fz = F(z[:,N])
        vals[end-n_blk2+1:end] .= vec(fz-eye(n,n+m))
    end
end

function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
    if mode == :Structure
        ind = 1
        for k = 1:N
            n_blk = (n^2 + m^2)
            for j = 1:n
                for i = 1:n
                    row[ind] = i + (k-1)*n_blk
                    col[ind] = j + (k-1)*n_blk
                    ind += 1
                end
            end
            for j = 1:m
                for i = 1:m
                    row[ind] = i + (k-1)*n_blk + n^2
                    col[ind] = j + (k-1)*n_blk + n^2
                    ind += 1
                end
            end
        end
    end
end

X = rand(n,N)
U = rand(m,N)
Z = packZ(X,U)

X2,U2 = unpackZ(Z)
g = zeros(n*(N-1))
eval_g(Z,g)
g
grad_f = zeros((n+m)*N)
eval_grad_f(Z,grad_f)


nnz_jac_g = (N-1)*n*2(n+m)
row = zeros(Int,nnz_jac_g)
col = zeros(Int,nnz_jac_g)
val = zeros(nnz_jac_g)
eval_jac_g(Z,:Structure, row, col, val)

jac_g0 = zeros(Int,(N-1)*n,N*(n+m))
inds = sub2ind(jac_g0,row,col)
jac_g0[inds] .= 1:nnz_jac_g
jac_g0

eval_jac_g(Z,:None,row,col,val)
val

nnz_h = (n^2 + m^2)*N
h = zeros(N*(n+m), N*(n+m))
row = zeros(Int,nnz_h)
col = zeros(Int,nnz_h)
vals = zeros(nnz_h)
lambda = zeros(n, N-1)
eval_h(Z, :Structure, row, col, 1, lambda, vals)

row

col
inds = CartesianIndex.(row[1:10],col[1:10])
h[inds] .= 1:10
h


function packZ(X,U)
    n, N = size(X)
    m = size(U,1)
    Z = zeros(n+m,N)
    Z[1:n,:] .= X
    Z[n+1:end,1:end] .= U
    Z = reshape(Z,1,(n+m)N)
end

function unpackZ(Z)
    Z = reshape(Z,n+m,N)
    X = Z[1:n,:]
    U = Z[n+1:end,:]
    return X,U
end

function cost(X,U)
    J = 0
    for k = 1:N-1
      J += dt/2*(cost_k(X[:,k],U[:,k]) + cost_k(X[:,k+1],U[:,k+1]))
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
end

function cost_k(x,u)
    0.5*(x-xf)'Q*(x-xf) + 0.5*u'R*u
end
