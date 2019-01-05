function cost(solver::Solver,X::Matrix,U::Matrix,weights::Vector,method::Symbol)
    obj = solver.obj
    f = solver.fc
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

            solver.opts.minimum_time ? dt = U[m̄,k] : nothing

            Xm = (x1+x2)/2 + dt/8*(f1-f2)
            Um = (U[1:m,k] + U[1:m,k+1])/2

            J[k] = dt/6*(stage_cost(obj.cost,X[:,k],U[1:m,k]) + 4*stage_cost(obj.cost,Xm,Um) + stage_cost(obj.cost,X[:,k+1],U[1:m,k+1])) # rk3 foh stage cost (integral approximation
            solver.opts.minimum_time ? J[k] += solver.opts.R_minimum_time*dt : nothing
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


function constraint_jacobian(solver::Solver, res::DircolResults, method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    jacob_g = spzeros((N-1)*n,N*(n+m))
    jacob_g = constraint_jacobian(solver,res.X_,res.U_,res.A,res.B,method)
    return jacob_g
end

function constraint_jacobian(solver::Solver, X, U, A, B, method::Symbol)
    N,N_ = get_N(solver,method)
    n,m = get_sizes(solver)
    jacob_g = spzeros((N-1)*n,N*(n+m))
    Inm = Matrix(I,n,n+m)
    dt = solver.dt

    if method == :trapezoid
        Z = packZ(X,U)
        z = reshape(Z,n+m,N)

        solver.opts.minimum_time ? dt = U[m̄:m̄,:] : dt = ones(1,N_)*solver.dt

        # First time step
        fz = A[:,:,1]
        jacob_g[1:n,1:n+m] .= dt[1]*fz/2+Inm

        # Loop over time steps
        for k = 2:N-1
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            # Calculate (n,n+m) Jacobian of both states and controls
            fz = A[:,:,k]  #F(z[:,k])
            jacob_g[off_1.-n.+(1:n),off_2.+(1:n+m)] .= dt[k]*fz/2 - Inm
            jacob_g[off_1  .+ (1:n),off_2.+(1:n+m)] .= dt[k]*fz/2 + Inm
        end

        # Last time step
        fz = A[:,:,N]  # F(z[:,N])
        jacob_g[end-n+1:end,end-n-m+1:end] = dt[N-1]*fz/2-Inm

    elseif method == :hermite_simpson_separated
        nSeg = Int((N-1)/2)
        Z = packZ(X,U)
        z = reshape(Z,n+m,N)

        fz1 = A[:,:,1]  # F(z[:,1])

        function calc_block(k)
            vals = zeros(2n,3(n+m))
            fz = A[:,:,k]  # F(z[:,k])
            vals[   (1:n),(1:n+m)] .= dt*fz/6 + Inm
            vals[n.+(1:n),(1:n+m)] .= dt*fz/8 + Inm/2
            fm = A[:,:,k+1]  # F(z[:,k+1])
            vals[   (1:n),n.+m.+(1:n+m)] .= 2*dt*fm/3
            vals[n.+(1:n),n.+m.+(1:n+m)] .= -Inm
            fz1 .= A[:,:,k+2]  # F(z[:,k+2])
            vals[   (1:n),2(n+m).+(1:n+m)] .=  dt*fz1/6 - Inm
            vals[n.+(1:n),2(n+m).+(1:n+m)] .= -dt*fz1/8 + Inm/2
            return vals
        end


        for i = 1:nSeg
            off_1 = 2(i-1)*(n)
            off_2 = 2(i-1)*(n+m)
            k = 2i-1

            jacob_g[off_1.+(1:2n), off_2.+(1:3(n+m))] = calc_block(k)
        end
    elseif method == :hermite_simpson
        nSeg = N-1

        Xk = view(X,:,1:2:N_)
        Uk = view(U,:,1:2:N_)
        Xm = view(X,:,2:2:N_-1)
        Um = view(U,:,2:2:N_-1)
        Ak = view(A,:,:,1:2:N_)
        Bk = view(B,:,:,1:2:N_)
        AM = view(A,:,:,2:2:N_-1)
        BM = view(B,:,:,2:2:N_-1)

        function calc_jacob_block(k::Int)::Matrix
            x1,u1 = Xk[:,k],Uk[:,k]
            x2,u2 = Xk[:,k+1],Uk[:,k+1]
            A1,A2 = Ak[:,:,k],Ak[:,:,k+1]
            B1,B2 = Bk[:,:,k],Bk[:,:,k+1]
            xm = Xm[:,k] #(x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            um = Um[:,k] # (u1+u2)/2
            Am,Bm = AM[:,:,k],BM[:,:,k]
            In = Matrix(I,n,n)
            Im = Matrix(I,m,m)

            vals = zeros(n,2(n+m))
            vals[:,1:n] =          dt/6*(A1 + 4Am*( dt/8*A1 + In/2)) + In    # ∇x1
            vals[:,n.+(1:m)] =     dt/6*(B1 + 4Am*( dt/8*B1) + 4Bm*(Im/2))   # ∇u1
            vals[:,n.+m.+(1:n)] =  dt/6*(A2 + 4Am*(-dt/8*A2 + In/2)) - In    # ∇x2
            vals[:,2n.+m.+(1:m)] = dt/6*(B2 + 4Am*(-dt/8*B2) + 4Bm*(Im/2))   # ∇u2
            return vals
        end

        for k = 1:nSeg
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            jacob_g[off_1.+(1:n), off_2.+(1:2(n+m))] = calc_jacob_block(k)
        end

    elseif method == :midpoint
        nSeg = N-1
        In = Matrix(I,n,n)

        function calc_jacob_block_midpoint(k)
            vals = zeros(n,2(n+m))
            vals[:,1:n] =          In + dt*A[:,:,k]/2    # ∇x1
            vals[:,n.+(1:m)] =     dt*B[:,:,k]           # ∇u1
            vals[:,n.+m.+(1:n)] = -In + dt*A[:,:,k]/2    # ∇x2
            vals[:,2n.+m.+(1:m)] = zeros(n,m)            # ∇u2
            return vals
        end

        for k = 1:nSeg
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            jacob_g[off_1.+(1:n), off_2.+(1:2(n+m))] = calc_jacob_block_midpoint(k)
        end
    end

    return jacob_g
end
