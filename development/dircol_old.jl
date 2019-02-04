function cost(solver::Solver,X::Matrix,U::Matrix,weights::Vector,method::Symbol)
    obj = solver.obj
    f = solver.model.f
    Q = obj.cost.Q; xf = obj.xf; Qf = obj.cost.Qf; R = obj.cost.R;
    n,m = get_sizes(solver)
    m̄, = get_num_controls(solver)
    N,N_ = get_N(solver,method)
    dt = solver.dt

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

            solver.state.minimum_time ? dt = U[m̄,k] : nothing

            Xm = (x1+x2)/2 + dt/8*(f1-f2)
            Um = (U[1:m,k] + U[1:m,k+1])/2

            J[k] = dt/6*(stage_cost(obj.cost,X[:,k],U[1:m,k]) + 4*stage_cost(obj.cost,Xm,Um) + stage_cost(obj.cost,X[:,k+1],U[1:m,k+1])) # rk3 foh stage cost (integral approximation
            solver.state.minimum_time ? J[k] += solver.opts.R_minimum_time*dt : nothing
        end
        J = sum(J)
        J += stage_cost(obj.cost,X[:,N])
        return J#, XM, UM, fVal
    elseif method == :midpoint
        Xm = zeros(eltype(X),n,N-1)
        for k = 1:N-1
            Xm[:,k] = (X[:,k] + X[:,k+1])/2
        end
        J = zeros(eltype(Xm),N)
        for k = 1:N-1
            J[k] = stage_cost(obj.cost,Xm[:,k],U[:,k])
        end
        J = weights'J
        J += stage_cost(obj.cost,X[:,N])
        return J

    else
        J = zeros(eltype(X),N)
        for k = 1:N
            J[k] = stage_cost(obj.cost,X[:,k],U[:,k])
        end
        J = weights'J
        J += stage_cost(obj.cost,X[:,N])
        return J
    end
end

function cost_gradient(solver::Solver, res::DircolResults, method::Symbol)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    grad_f = zeros((n+m)N)
    cost_gradient!(solver,res.X_,res.U_,res.fVal_,res.A,res.B,res.weights,grad_f,method)
    return grad_f
end


function cost_gradient!(solver::Solver, vars::DircolVars, weights::Vector{Float64}, vals::Vector{Float64}, method::Symbol)::Nothing
    # println("simple")
    n,m = get_sizes(X,U)
    N,N_ = get_N(solver,method)
    dt = solver.dt

    obj = solver.obj
    Q = obj.cost.Q; xf = obj.xf; R = obj.cost.R; Qf = obj.cost.Qf;
    # X,U = res.X_, res.U_
    grad_f = reshape(vals, n+m, N)

    for k = 1:N
        grad_f[1:n,k] = weights[k]*Q*(X[:,k] - xf)*dt
        grad_f[n+1:end,k] = weights[k]*R*U[:,k]*dt
    end
    grad_f[1:n,N] += Qf*(X[:,N] - xf)
    return nothing
end









function collocation_constraints(solver::Solver, res::DircolResults, method::Symbol)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver,method)
    g = zeros(eltype(res.X),(N-1)*n)
    collocation_constraints!(solver,res.X_,res.U_,res.fVal_,g,method)
    return g
end

function collocation_constraints(X,U,method,dt,f::Function)
    n,m,N = get_sizes(X,U)
    g = zeros(eltype(X),(N-1)*n)
    g = reshape(g,n,N-1)

    fVal = zeros(eltype(X),n,N)
    for k = 1:N
        f(view(fVal,:,k),X[:,k],U[:,k])
    end

    if method == :trapezoid
        for k = 1:N-1
            # Collocation Constraints
            g[:,k] = dt*( fVal[:,k+1] + fVal[:,k] )/2 - X[:,k+1] + X[:,k]
        end
    elseif method == :hermite_simpson_separated
        iLow = 1:2:N-1
        iMid = iLow .+ 1
        iUpp = iMid .+ 1

        midpoints =  - X[:,iMid] + (X[:,iLow]+X[:,iUpp])/2 + dt*(fVal[:,iLow] - fVal[:,iUpp])/8
        collocation = - X[:,iUpp] + X[:,iLow] + dt*(fVal[:,iLow] + 4*fVal[:,iMid] + fVal[:,iUpp])/6
        g[:,iLow] = collocation
        g[:,iMid] = midpoints

    elseif method == :hermite_simpson
        fm = zeros(eltype(X),n)
        for k = 1:N-1
            x1 = X[:,k]
            x2 = X[:,k+1]
            xm = (x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            um = (U[:,k] + U[:,k+1])/2
            f(fm, xm, um)
            g[:,k] = -x2 + x1 + dt*(fVal[:,k] + 4*fm + fVal[:,k+1])/6
        end

    elseif method == :midpoint
        fm = zeros(eltype(X),n)
        for k = 1:N-1
            x1 = X[:,k]
            x2 = X[:,k+1]
            xm = (x1+x2)/2
            f(fm,xm,U[:,k])
            g[:,k] = dt*fm - x2 + x1
        end
    end
    return vec(g)
end
