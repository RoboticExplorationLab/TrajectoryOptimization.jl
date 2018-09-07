using Ipopt
using ForwardDiff

function init_jacobians(solver,method)
    N,N_ = get_N(solver,method)
    if method == :trapezoid || method == :hermite_simpson_separated
        A = zeros(n,n+m,N_)
        B = zeros(0,0,N_)
    else
        A = zeros(n,n,N_)
        B = zeros(n,m,N_)
    end
    return A,B
end

model, obj0 = Dynamics.cartpole_analytical
obj = copy(obj0)
n,m = model.n, model.m
dt = 0.1

# Set up problem
method = :midpoint
solver = Solver(model,ConstrainedObjective(obj),dt=dt,integration=:rk3_foh)
N,N_ = TrajectoryOptimization.get_N(solver,method)
NN = N*(n+m)
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)
Z = TrajectoryOptimization.packZ(X0,U0)

# Init Variables
results = DircolResults(n,m,solver.N,method)
results.Z .= Z
fVal = zeros(X0)
N,N_ = get_N(solver,method)
gX_,gU_,fVal_ = init_traj_points(solver,X0,U0,fVal,method)
weights = get_weights(method,N_)
A,B = init_jacobians(solver,method)









#################
# COST FUNCTION #
#################


function eval_f(Z)
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X, vars.U
    X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method,true)
    cost(solver,X_,U_,weights)
end

function eval_f2(Z)
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X, vars.U
    cost(solver,X,U,weights)
end

function eval_f3(Z)
    results.Z .= Z
    vars = DircolVars(Z,n,m,N)
    update_derivatives!(solver,results,method)
    get_traj_points!(solver,results,method)
    cost(solver,results)
end




###########################
# COLLOCATION CONSTRAINTS #
###########################

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



#################
# COST GRADIENT #
#################

grad_f = zeros(NN)
grad_f1 = zeros(NN)
grad_f2 = zeros(NN)

function eval_grad_f(Z, grad_f)
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X, vars.U
    X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
    get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
    update_jacobians!(solver,X_,U_,A,B,method,true)
    cost_gradient!(solver, X_, U_, fVal, A, B, weights, grad_f, method)
    return nothing
end
eval_grad_f(Z, grad_f)
@btime eval_grad_f(Z, grad_f1)

function eval_grad_f1(Z, grad_f)
    results.Z .= Z
    update_derivatives!(solver,results,method)
    get_traj_points!(solver,results,method)
    get_traj_points_derivatives!(solver,results,method)
    grad_f .= cost_gradient(solver,results,method)
    return nothing
end
eval_grad_f1(Z, grad_f1)
grad_f == grad_f1
@btime eval_grad_f1(Z, grad_f1)


#######################
# CONSTRAINT JACOBIAN #
#######################

function get_nG(solver::Solver,method::Symbol)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    if method == :trapezoid || method == :hermite_simpson
        return 2(n+m)*(N-1)n
    elseif method == :hermite_simpson_separated
        return 3(n+m)*(N-1)n
    elseif method == :midpoint
        return (2n+m)*(N-1)n
    end
end
nG = get_nG(solver,method)
jacob_g = constraint_jacobian_sparsity(solver,method)
Array(jacob_g)


function eval_jac_g(Z, mode, rows, cols, vals)
    if mode == :Structure

    else
        vars = DircolVars(Z,n,m,N)
        X,U, = vars.X,vars.U
        X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
        get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
        update_jacobians!(solver,X_,U_,A,B,method)
        jacob_g = constraint_jacobian(solver,X_,U_,A,B,method)
    end
end
jacob_g = eval_jac_g(Z,:vals,[],[],[])


function eval_jac_g1(Z, mode, rows, cols, vals)
    if mode == :Structure

    else
        vars = DircolVars(Z,n,m,N)
        X,U, = vars.X,vars.U
        X_,U_ = get_traj_points(solver,X,U,fVal,gX_,gU_,method)
        get_traj_points_derivatives!(solver,X_,U_,fVal_,fVal,method)
        update_jacobians!(solver,X_,U_,A,B,method)
        constraint_jacobian!(solver,X_,U_,A,B,vals,method)
    end
    return nothing
end
n_blk = 6(n+m)n
vals = zeros(nG)
eval_jac_g1(Z,:vals,[],[],vals)
jacob_g2 = sparse(rows,cols,vals)
jacob_g2 == jacob_g



b3 = Array(jacob_g[4n+(1:2n),4(n+m)+(1:3(n+m))])
b1 == b3

Array(jacob_g1 - jacob_g)



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
