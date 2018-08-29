
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

function get_weights(method,N::Int)
    if method == :trapezoid
        weights = ones(N)
        weights[[1,end]] = 0.5
    elseif method == :hermite_simpson_separated
        weights = ones(N)*2/3
        weights[2:2:end] = 4/3
        weights[[1,end]] = 1/3
    end
    return weights
end

function get_bounds(obj::Objective,N::Int)
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

function get_pack(Z, obj)
    n,m = get_sizes(obj)
    return (n,m, length(Z)//(n*m))
end

function get_initial_state(obj::Objective, N::Int)
    n,m = get_sizes(obj)
    X0 = line_trajectory(obj.x0, obj.xf, N)
    U0 = zeros(m,N)
    Z0 = packZ(X0,U0)
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
    end
    # reshape(g,n*(N-1),1)
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
