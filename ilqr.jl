module iLQR
using RigidBodyDynamics
using ForwardDiff

struct Model
    f::Function
    n::Int
    m::Int

    function Model(f::Function, n::Int64, m::Int64)
        new(f,n,m)
    end

    function Model(mech::Mechanism)
        n = num_positions(mech) + num_velocities(mech) + num_additional_states(mech)
        num_joints = length(joints(mech))-1  # subtract off joint to world
        m = num_joints # Default to number of joints

        function fc(x,u)
            state = MechanismState{eltype(x)}(mech)

            # set the state variables:
            q = x[1:num_joints]
            qd = x[(1:num_joints)+num_joints]
            set_configuration!(state, q)
            set_velocity!(state, qd)

            [qd; Array(mass_matrix(state))\u - Array(mass_matrix(state))\Array(dynamics_bias(state))]
        end
        new(fc, n, m)
    end
end

function Model(urdf::String)
    mech = parse_urdf(Float64,urdf)
    Model(mech)
end

struct Objective
    Q::Array{Float64,2}
    R::Array{Float64,2}
    Qf::Array{Float64,2}
    tf::Float64
    x0::Array{Float64,1}
    xf::Array{Float64,1}
end

struct Solver
    model::Model
    obj::Objective
    dt::Float64
    fd::Function  # discrete dynamics
    F::Function
    N::Int
    function Solver(model::Model, obj::Objective, dt::Float64)
        obj_n = size(obj.Q, 1)
        obj_m = size(obj.R, 1)
        @assert obj_n == model.n
        @assert obj_m == model.m

        # RK4 integration
        fd = rk4(model.f, dt)
        F(x,u) = Jacobian(fd,x,u)
        N = Int(floor(obj.tf/dt))
        new(model, obj, dt, fd, F, N)
    end

    # function Solver(model, obj, dt=0.1)
    #     n, m = model.n, model.m
    #     fd = f_midpoint(model.f, dt)     # Discrete dynamics
    #     f_aug = f_augmented(model)  # Augmented continuous dynamics
    #     fd_aug = f_midpoint(f_aug)  # Augmented discrete dynamics
    #
    #     out = zeros(n+m+1)
    #     Df(S::Array) = ForwardDiff.jacobian(fd_aug, S)
    #
    #     function f_jacobian(x::Array,u::Array,dt::Float64)
    #         Df_aug = Df([x;u;dt])
    #         A = Df_aug[1:n,1:n]
    #         B = Df_aug[1:n,n+1:n+m]
    #         return A,B
    #     end
    #
    #     N = Int(floor(obj.tf/dt));
    #     new(model, obj, dt, fd, f_jacobian, N)
    # end
end

# function f_midpoint(f::Function, dt::Float64)
#     dynamics_midpoint(x,u)  = x + f(x + f(x,u)*dt/2, u)*dt
# end
#
# function f_midpoint(f::Function)
#     dynamics_midpoint(S::Array)  = S + f(S + f(S)*S[end]/2)*S[end]
# end
#
# function f_midpoint!(f_aug!::Function)
#
#     function dynamics_midpoint(out::AbstractVector, S::Array)
#         # out = zeros(7)
#         f_aug!(out, S)
#         f_aug!(out, S + out*S[end]/2)
#         copy!(out, S + out*S[end])
#     end
# end
#
#
# function f_augmented(model::Model)
#     n, m = model.n, model.m
#     f_aug = f_augmented(model.f, n, m)
#     f(S::Array) = [f_aug(S); zeros(m+1)]
# end
#
# function f_augmented!(model::Model)
#     n, m = model.n, model.m
#     f_aug! = f_augmented!(model.f, n, m)
#     f!(out::AbstractVector, S::Array) = [f_aug!(out, S); zeros(m+1)]
# end
#
# function f_augmented(f::Function, n::Int, m::Int)
#     f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
# end
#
# function f_augmented!(f::Function, n::Int, m::Int)
#     f_aug!(out::AbstractVector, S::Array) = copy!(out, f(S[1:n], S[n+(1:m)]))
# end

function rk4(f::Function,dt::Float64)
    # Runge-Kutta 4
    k1(x,u) = dt*f(x,u)
    k2(x,u) = dt*f(x + k1(x,u)/2.,u)
    k3(x,u) = dt*f(x + k2(x,u)/2.,u)
    k4(x,u) = dt*f(x + k3(x,u), u)
    fd(x,u) = x + (k1(x,u) + 2.*k2(x,u) + 2.*k3(x,u) + k4(x,u))/6.
end

function midpoint(f::Function,dt::Float64)
    fd(x,u) = x + f(x + f(x,u)*dt/2., u)*dt
end

function Jacobian(f::Function,x::Array{Float64,1},u::Array{Float64,1})
    f1 = a -> f(a,u)
    f2 = b -> f(x,b)
    fx = ForwardDiff.jacobian(f1,x)
    fu = ForwardDiff.jacobian(f2,u)
    return fx, fu
end

#iLQR
function rollout(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        X[:,k+1] = solver.fd(X[:,k],U[:,k])
    end
    return X
end

# function rollout(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},X_::Array{Float64,2},U_::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2},alpha::Float64)
#     X_prev = copy(X)
#     X[:,1] = solver.obj.x0
#     for k = 1:solver.N-1
#       U_[:,k] = U[:,k] - K[:,:,k]*(X[:,k] - X_prev[:,k]) - alpha*d[:,k]
#       X[:,k+1] = solver.fd(X[:,k],U_[:,k]);
#     end
#     return X, U_
# end

# function rollout!(solver::Solver, X::Array{Float64,2}, U::Array{Float64,2}, K::Array{Float64,3}, d::Array{Float64,2}, alpha::Float64, X_::Array{Float64,2}, U_::Array{Float64,2})
#     N = solver.N
#     X_[:,1] = solver.obj.x0;
#     for k = 2:N
#         a = alpha*(d[:,k-1]);
#         delta = (X_[:,k-1] - X[:,k-1])

#         U_[:, k-1] = U[:, k-1] - K[:,:,k-1]*delta - a;
#         X_[:,k] = solver.fd(X_[:,k-1], U_[:,k-1]);
#     end
# end

function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    N = solver.N
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf

    J = 0.0
    for k = 1:N-1
      J += 0.5*(X[:,k] - xf)'*Q*(X[:,k] - xf) + 0.5*U[:,k]'*R*U[:,k]
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

function backwardpass(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2})
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf

    S = zeros(n,n,N)
    s = zeros(n,N)

#     K = zeros(m,n,N-1)
#     d = zeros(m,N-1)

    S = Qf
    s = Qf*(X[:,N] - xf)
    v1 = 0.0
    v2 = 0.0

    mu = 0.0
    k = N-1

    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R
        fx, fu = solver.F(X[:,k],U[:,k])

        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = luu + fu'*(S + mu*eye(n))*fu
        Qux = fu'*(S + mu*eye(n))*fx

        # regularization
        if any(x->x < 0.0, (eigvals(Quu)))
            mu = mu + 1.0;
            k = N-1;
            println("regularized")
        end

        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)'
        S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]

        # terms for line search
        v1 += d[:,k]'*Qu
        v2 += d[:,k]'*Quu*d[:,k]

        k = k - 1;
    end
    return K, d, v1, v2
end

function forwardpass(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2},J::Float64,v1,v2,c1::Float64=0.0,c2::Float64=1.0)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    X_prev = copy(X)
    J_prev = copy(J)
    U_ = zeros(m,N-1)
    J = Inf
    dV = 0.0
    dJ = 0.0
    z = 0.0

    alpha = 1.0

    while J > J_prev || z < c1 || z > c2
        X[:,1] = solver.obj.x0
        for k = 1:N-1
            U_[:,k] = U[:,k] - K[:,:,k]*(X[:,k] - X_prev[:,k]) - alpha*d[:,k]
            X[:,k+1] = solver.fd(X[:,k],U_[:,k]);
        end
#         X_, U_ = rollout(solver,X,U,X_,U_,K,d,alpha)

         J = cost(solver,X,U_)
#        J = cost(solver,X_,U_)
        dV = alpha*v1 + (alpha^2)*v2/2.0
        dJ = J_prev - J
        z = dJ/dV[1]

        alpha = alpha/2.0;
    end

    println("New cost: $J")
    println("- Expected improvement: $(dV[1])")
    println("- Actual improvement: $(dJ)")
    println("- (z = $z)\n")

      return X, U_, J
#     return X_, U_, J
end

function solve(solver::Solver,iterations::Int64=100,eps::Float64=1e-3;control_init::String="random")
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    X = zeros(n,N)
    X_ = zeros(n,N)

    if control_init == "random"
        U = 10.0*rand(m,N-1)
    else
        U = zeros(m,N-1)
    end
    U_ = zeros(m,N-1)

    K = zeros(m,n,N-1)
    d = zeros(m,N-1)

    X = rollout(solver, X, U)
    J_prev = cost(solver, X, U)
    println("Initial Cost: $J_prev\n")

    for i = 1:iterations
        println("*** Iteration: $i ***")
        K, d, v1, v2 = backwardpass(solver,X,U,K,d)
        X, U, J = forwardpass(solver,X,U,K,d,J_prev,v1,v2)

        if abs(J-J_prev) < eps
          println("-----SOLVED-----")
          println("eps criteria met at iteration: $i")
          break
        end
        J_prev = copy(J)
    end

    return X, U
end

end
