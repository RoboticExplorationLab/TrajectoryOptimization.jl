function cost(solver::Solver,X::Matrix,U::Matrix,weights::Vector,method::Symbol)
    obj = solver.obj
    f = solver.fc
    Q = obj.Q; xf = obj.xf; Qf = obj.Qf; R = obj.R;
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    N,N_ = get_N(solver,method)
    dt = solver.dt
    XM = zeros(eltype(X),n,N-1)
    UM = zeros(eltype(X),m̄,N-1)
    fVal = zeros(eltype(X),n,N)

    if method == :hermite_simpson
        # pull out solver/objective values
        J = zeros(eltype(X),N-1)
        f1 = zeros(eltype(X),n)
        f2 = zeros(eltype(X),n)
        for k = 1:N-1
            f(f1,X[:,k], U[1:m,k])
            f(f2,X[:,k+1], U[1:m,k+1])
            x1 = X[:,k]
            x2 = X[:,k+1]

            solver.opts.minimum_time ? dt = U[m̄,k] : nothing

            Xm = (x1+x2)/2 + dt/8*(f1-f2)
            Um = (U[1:m,k] + U[1:m,k+1])/2


            XM[:,k] = Xm
            UM[:,k] = [Um;dt]
            fVal[:,k+1] = f2
            if k > 1 && fVal[:,k] != f1
                error("Something happened")
            end
            fVal[:,k] = f1


            J[k] = dt/6*(ℓ(X[:,k],U[1:m,k],Q,R,xf) + 4*ℓ(Xm,Um,Q,R,xf) + ℓ(X[:,k+1],U[1:m,k+1],Q,R,xf)) # rk3 foh stage cost (integral approximation)
        end
        J = sum(J)
        J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
        return J#, XM, UM, fVal
    elseif method == :midpoint
        Xm = zeros(eltype(X),n,N-1)
        for k = 1:N-1
            Xm[:,k] = (X[:,k] + X[:,k+1])/2
        end
        J = zeros(eltype(Xm),N)
        for k = 1:N-1
            J[k] = stage_cost(Xm[:,k],U[:,k],Q,R,xf,0.)
        end
        J = weights'J
        J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
        return J

    else
        J = zeros(eltype(X),N)
        for k = 1:N
            J[k] = stage_cost(X[:,k],U[:,k],Q,R,xf,0.)
        end
        J = weights'J
        J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
        return J
    end
end

function cost_gradient!(solver::Solver, vars::DircolVars, weights::Vector{Float64}, vals::Vector{Float64}, method::Symbol)::Nothing
    # println("simple")
    n,m = get_sizes(X,U)
    N,N_ = get_N(solver,method)
    dt = solver.dt

    obj = solver.obj
    Q = obj.Q; xf = obj.xf; R = obj.R; Qf = obj.Qf;
    # X,U = res.X_, res.U_
    grad_f = reshape(vals, n+m, N)

    for k = 1:N
        grad_f[1:n,k] = weights[k]*Q*(X[:,k] - xf)*dt
        grad_f[n+1:end,k] = weights[k]*R*U[:,k]*dt
    end
    grad_f[1:n,N] += Qf*(X[:,N] - xf)
    return nothing
end

function get_traj_points!(solver::Solver,res::DircolResults,method::Symbol)
    get_traj_points!(solver,res.X,res.U,res.X_,res.U_,res.fVal,method)
end

function get_traj_points_derivatives!(solver::Solver,res::DircolResults,method::Symbol)
    get_traj_points_derivatives!(solver::Solver,res.X_,res.U_,res.fVal_,res.fVal,method::Symbol)
end

function update_derivatives!(solver::Solver,res::DircolResults,method::Symbol)
    # Calculate derivative
    if method != :midpoint
        update_derivatives!(solver,res.X,res.U,res.fVal)
    end
end

function update_jacobians!(solver::Solver,res::DircolResults,method::Symbol,cost_only::Bool=false)
    update_jacobians!(solver,res.X_,res.U_,res.A,res.B,method,cost_only) # TODO: pass in DircolVar
end
