
function get_sizes(obj::Objective)
    n = size(obj.Q,1)
    m = size(obj.R,1)
    return n,m
end

function get_sizes(X::Matrix,U::Matrix)
    n,N = size(X)
    m = size(U,1)
    return n,m,N
end

function convertInf!(A::Matrix,infbnd=1.1e20)
    infs = isinf.(A)
    A[infs] = sign.(A[infs])*infbnd
    return nothing
end


function convert_N(N::Int,method::Symbol)::Int
    nSeg = N-1
    if method == :hermite_simpson_separated
        N = 2nSeg + 1
    else
        N = nSeg + 1
    end
    return N
end

function get_weights(method,N::Int)
    if method == :trapezoid
        weights = ones(N)
        weights[[1,end]] = 0.5
    elseif method == :hermite_simpson_separated ||
            method == :hermite_simpson
        weights = ones(N)*2/6
        weights[2:2:end] = 4/6
        weights[[1,end]] = 1/6
    elseif method == :midpoint
        weights = ones(N)
        weights[end] = 0
    end
    return weights
end

function get_bounds(obj::UnconstrainedObjective,N::Int)
    get_bounds(ConstrainedObjective(obj),N::Int)
end

function get_bounds(obj::ConstrainedObjective,N::Int)
    n,m = get_sizes(obj)
    lb = zeros((n+m),N)
    ub = zeros((n+m),N)

    lb[1:n,:] .= obj.x_min
    ub[1:n,:] .= obj.x_max
    lb[n+(1:m),:] .= obj.u_min
    ub[n+(1:m),:] .= obj.u_max

    # Initial Condition
    lb[1:n,1] .= obj.x0
    ub[1:n,1] .= obj.x0

    # Terminal Constraint
    lb[1:n,N] .= obj.xf
    ub[1:n,N] .= obj.xf

    # Convert Infinite bounds
    convertInf!(lb)
    convertInf!(ub)
    return vec(lb), vec(ub)
end


"""
Stack state and controls for all time steps into a single vector of variables
Z = [X1,U1,X2,U2,..,] (all transposed)
"""
function packZ(X,U)
    n, N = size(X)
    m = size(U,1)
    Z = zeros(n+m,N)
    Z[1:n,:] .= X
    Z[n+1:end,1:end] .= U
    Z = vec(Z)
end

function unpackZ(Z, sze)
    n,m,N = sze
    Z = reshape(Z,n+m,N)
    X = Z[1:n,:]
    U = Z[n+1:end,:]
    return X,U
end

function get_initial_state(obj::Objective, N::Int)
    n,m = get_sizes(obj)
    X0 = line_trajectory(obj.x0, obj.xf, N)
    U0 = zeros(m,N)
    return X0, U0
end


function cost(obj::Objective,f::Function,X::Array{Float64,2},U::Array{Float64,2})
    # pull out solver/objective values
    N = size(X,2); Q = obj.Q; xf = obj.xf; Qf = obj.Qf; R = obj.R;
    dt = obj.tf/(N-1)
    n,m = get_sizes(obj)

    J = 0.0
    f1 = zeros(n)
    f2 = zeros(n)
    for k = 1:N-1
        f(f1,X[:,k], U[:,k])
        f(f2,X[:,k+1], U[:,k+1])
        x1 = X[:,k]
        x2 = X[:,k+1]

        Xm = (x1+x2)/2 + dt/8*(f1-f2)
        Um = (U[:,k] + U[:,k+1])/2

        J += dt/6*(stage_cost(X[:,k],U[:,k],Q,R,xf) + 4*stage_cost(Xm,Um,Q,R,xf) + stage_cost(X[:,k+1],U[:,k+1],Q,R,xf)) # rk3 foh stage cost (integral approximation)
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)

    return J
end

"""
Evaluate Objective Value
"""
function cost(X,U,weights,obj::Objective)
    N = size(X,2)
    Qf = obj.Qf; xf = obj.xf;
    J = zeros(eltype(X),N)
    for k = 1:N
        J[k] = cost_k(X[:,k],U[:,k],obj)
    end
    J = weights'J
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
end

function cost_k(x,u,obj::Objective)
    xf = obj.xf
    0.5*(x-xf)'obj.Q*(x-xf) + 0.5*u'obj.R*u
end

"""
Gradient of Objective
"""
function cost_gradient(X,U,weights,obj::Objective)
    Q = obj.Q; xf = obj.xf; R = obj.R; Qf = obj.Qf;
    n,m,N = get_sizes(X,U)
    grad_f = zeros(eltype(X),(n+m)*N)
    grad_f = reshape(grad_f, n+m, N)
    for k = 1:N
        grad_f[1:n,k] = weights[k]*Q*(X[:,k] - xf) # weight includes dt
        grad_f[n+1:end,k] = weights[k]*R*U[:,k]
    end
    grad_f[1:n,N] += Qf*(X[:,N] - xf)
    return vec(grad_f)
end


"""
Evaluate constraint values
[ dt*(f(x1)+f(x2))/2 - x2 + x1     ] Collocation Constraints
[ dt*(f(x2)+f(x3))/2 - x3 + x2     ]
[           ...                 ]
[ dt*(f(xN-1)+f(xN))/2 - xN + xN-1 ]
"""
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
        iMid = iLow + 1
        iUpp = iMid + 1

        midpoints =  - X[:,iMid] + (X[:,iLow]+X[:,iUpp])/2 + dt*(fVal[:,iLow] - fVal[:,iUpp])/8
        collocation = - X[:,iUpp] + X[:,iLow] + dt*(fVal[:,iLow] + 4*fVal[:,iMid] + fVal[:,iUpp])/6
        g[:,iLow] = collocation
        g[:,iMid] = midpoints

    elseif method == :hermite_simpson
        fm = zeros(n)
        for k = 1:N-1
            x1 = X[:,k]
            x2 = X[:,k+1]
            xm = (x1+x2)/2 + dt/8*(fVal[:,k]-fVal[:,k+1])
            um = (U[:,k] + U[:,k+1])/2
            f(fm, xm, um)
            g[:,k] = -x2 + x1 + dt*(fVal[:,k] + 4*fm + fVal[:,k+1])/6
        end

    elseif method == :midpoint
        fm = zeros(n)
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


"""
Constraint Jacobian
"""
function constraint_jacobian(X,U,dt,method,F::Function)
    n,m,N = get_sizes(X,U)
    Z = packZ(X,U)
    jacob_g = zeros((N-1)*n,N*(n+m))
    z = reshape(Z,n+m,N)

    if method == :trapezoid
        nm = n+m
        # First time step
        fz = F(z[:,1])
        jacob_g[1:n,1:n+m] .= dt*fz/2+eye(n,n+m)

        # Loop over time steps
        for k = 2:N-1
            off_1 = (k-1)*(n)
            off_2 = (k-1)*(n+m)
            # Calculate (n,n+m) Jacobian of both states and controls
            fz = F(z[:,k])
            jacob_g[off_1-n+(1:n),off_2+(1:n+m)] .= dt*fz/2 - eye(n,n+m)
            jacob_g[off_1 + (1:n),off_2+(1:n+m)] .= dt*fz/2 + eye(n,n+m)
        end

        # Last time step
        fz = F(z[:,N])
        jacob_g[end-n+1:end,end-n-m+1:end] = dt*fz/2-eye(n,n+m)

    elseif method == :hermite_simpson_separated
        nSeg = Int((N-1)/2)
        jacob_g = zeros(n*(N-1),(n+m)N)

        fz1 = F(z[:,1])
        Inm = Matrix(I,n,n+m)

        function calc_block(k)
            vals = zeros(2n,3(n+m))
            fz = F(z[:,k])
            vals[0+(1:n),(1:n+m)] .= dt*fz/6 + Inm
            vals[n+(1:n),(1:n+m)] .= dt*fz/8 + Inm/2
            fm = F(z[:,k+1])
            vals[0+(1:n),n+m+(1:n+m)] .= 2*dt*fm/3
            vals[n+(1:n),n+m+(1:n+m)] .= -Inm
            fz1 .= F(z[:,k+2])
            vals[0+(1:n),2(n+m)+(1:n+m)] .=  dt*fz1/6 - Inm
            vals[n+(1:n),2(n+m)+(1:n+m)] .= -dt*fz1/8 + Inm/2
            return vals
        end


        for i = 1:nSeg
            off_1 = 2(i-1)*(n)
            off_2 = 2(i-1)*(n+m)
            k = 2i-1

            jacob_g[off_1+(1:2n), off_2+(1:3(n+m))] = calc_block(k)
        end
    end


    return jacob_g
end


"""
$(SIGNATURES)
Interpolate a trajectory using cubic interpolation
"""
function interp_traj(N::Int,tf::Float64,X::Matrix,U::Matrix)::Tuple{Matrix,Matrix}
    X2 = interp_rows(N,tf,X)
    U2 = interp_rows(N,tf,U)
    return X2, U2
end

"""
$(SIGNATURES)
Interpolate the rows of a matrix using cubic interpolation
"""
function interp_rows(N::Int,tf::Float64,X::Matrix)::Matrix
    n,N1 = size(X)
    t1 = linspace(0,tf,N1)
    t2 = linspace(0,tf,N)
    X2 = zeros(n,N)
    for i = 1:n
        interp_cubic = CubicSplineInterpolation(t1, X[i,:])
        X2[i,:] = interp_cubic(t2)
    end
    return X2
end


# JUNK FUNCTIONS

function rollout_midpoint(solver::Solver,U::Matrix)
    N = solver.N
    N = convert_N(N,method)
    n,m = solver.model.n,solver.model.m
    nSeg = N-1
    N_ = 2*nSeg + 1
    X_ = zeros(n,N_)
    U_ = zeros(size(U,1),N_)
    X_[:,1] = solver.obj.x0
    U_[:,1] = U[:,1]

    for k = 1:N_-1
        if isodd(k)
            j = (k+1)÷2
            U_[:,k+2] = U[:,j+1]
            solver.fd(view(X_,:,k+2), X_[:,k], U_[1:m,k], U_[1:m,k+2])
        else
            Ac1, Bc1 = solver.Fc(X_[:,k-1],U_[:,k-1])
            Ac2, Bc2 = solver.Fc(X_[:,k+1],U_[:,k+1])
            M = [(0.5*eye(n) + dt/8*Ac1) (dt/8*Bc1) (0.5*eye(n) - dt/8*Ac2) (-dt/8*Bc2)]

            Xm = M*[X_[:,k-1]; U_[:,k-1]; X_[:,k+1]; U_[:,k+1]]
            Um = (U_[:,k-1] + U_[:,k+1])/2

            X_[:,k] = Xm
            U_[:,k] = Um
        end
    end
    return X_,U_
end

function calc_midpoints(X::Matrix, U::Matrix, solver::Solver)
    N = solver.N
    N = convert_N(N,method)
    n,m = solver.model.n,solver.model.m
    nSeg = N-1
    N_ = 2*nSeg + 1
    X_ = zeros(n,N_)
    U_ = zeros(size(U,1),N_)
    X_[:,1:2:end] = X
    U_[:,1:2:end] = U

    f1 = zeros(n)
    f2 = zeros(n)
    for k = 2:2:N_

        f(f1,X_[:,k-1], U_[:,k-1])
        f(f2,X_[:,k+1], U_[:,k+1])
        x1 = X_[:,k-1]
        x2 = X_[:,k+1]
        Xm = (x1+x2)/2 + dt/8*(f1-f2)

        Um = (U_[:,k-1] + U_[:,k+1])/2

        X_[:,k] = Xm
        U_[:,k] = Um
    end
    return X_,U_
end

function interp(t,T,X,U,F)
    k = findlast(t .> T)
    τ = t-T[k]
    if method == :trapezoid
        u = U[:,k] + τ/dt*(F[:,k+1]-F[:,k])
        x = X[:,k] + F[:,k]*τ + τ^2/(2*dt)*(F[:,k+1]-F[:,k])
    elseif method == :hermite_simpson || method == :hermite_simpson_separated
        x1,x2 = X[:,k], X[:,k+1]
        u1,u2 = U[:,k], U[:,k+1]
        f1,f2 = F[:,k], F[:,k+1]
        xm = (x1+x2)/2 + dt/8*(f1-f2)
        um = (U[:,k] + U[:,k+1])/2

        u = (2(τ-dt/2)(τ-dt)U[:,k] - 4τ*(τ-dt)Um + 2τ*(τ-dt/2)*U[:,k+1])/dt^2
        x = X[:,k] + F[:,k]*τ/dt + 1/2*(-3F[:,])
    end
    return x,u
end
