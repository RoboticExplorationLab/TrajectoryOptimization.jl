module iLQR
using RigidBodyDynamics
using ForwardDiff

struct Model
    f::Function
    n::Int
    m::Int

    function Model(f::Function, n::Int, m::Int)
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

            # return momentum converted to an `Array` (as this is the format that ForwardDiff expects)
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
    fd::Function  # Discrete dynamics
    f_jacobian::Function
    N::Int
    function Solver(model, obj, f_jacobian, dt)
        obj_n = size(obj.Q, 1)
        obj_m = size(obj.R, 1)
        @assert obj_n == model.n
        @assert obj_m == model.m

        f_mid = f_midpoint(model.f, dt)
        N = Int(floor(obj.tf/dt));
        new(model, obj, dt, f_mid, f_jacobian, N)
    end
    function Solver(model, obj, dt=0.1)
        n, m = model.n, model.m
        fd = f_midpoint(model.f, dt)     # Discrete dynamics
        f_aug = f_augmented(model)  # Augmented continuous dynamics
        fd_aug = f_midpoint(f_aug)  # Augmented discrete dynamics

        out = zeros(n+m+1)
        Df(S::Array) = ForwardDiff.jacobian(fd_aug, S)

        function f_jacobian(x::Array,u::Array,dt::Float64)
            Df_aug = Df([x;u;dt])
            A = Df_aug[1:n,1:n]
            B = Df_aug[1:n,n+1:n+m]
            return A,B
        end

        N = Int(floor(obj.tf/dt));
        new(model, obj, dt, fd, f_jacobian, N)
    end
end

function f_midpoint(f::Function, dt::Float64)
    dynamics_midpoint(x,u)  = x + f(x + f(x,u)*dt/2, u)*dt
end

function f_midpoint(f::Function)
    dynamics_midpoint(S::Array)  = S + f(S + f(S)*S[end]/2)*S[end]
end

function f_midpoint!(f_aug!::Function)

    function dynamics_midpoint(out::AbstractVector, S::Array)
        # out = zeros(7)
        f_aug!(out, S)
        f_aug!(out, S + out*S[end]/2)
        copy!(out, S + out*S[end])
    end
end


function f_augmented(model::Model)
    n, m = model.n, model.m
    f_aug = f_augmented(model.f, n, m)
    f(S::Array) = [f_aug(S); zeros(m+1)]
end

function f_augmented!(model::Model)
    n, m = model.n, model.m
    f_aug! = f_augmented!(model.f, n, m)
    f!(out::AbstractVector, S::Array) = [f_aug!(out, S); zeros(m+1)]
end

function f_augmented(f::Function, n::Int, m::Int)
    f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
end

function f_augmented!(f::Function, n::Int, m::Int)
    f_aug!(out::AbstractVector, S::Array) = copy!(out, f(S[1:n], S[n+(1:m)]))
end



function rollout!(solver::Solver, x::Array{Float64,2}, u::Array{Float64,2})
    N = size(x, 2)
    for k = 2:N
        x[:, k] = solver.fd(x[:,k-1], u[:,k-1])
    end
end

function rollout!(solver::Solver, x::Array{Float64,2}, u::Array{Float64,2}, K::Array{Float64,3}, lk::Array{Float64,2}, alpha::Float64)
    N = solver.N
    x_ = zeros(solver.model.n, N);
    u_ = zeros(solver.model.m, N)
    rollout!(solver::Solver, x::Array{Float64,2}, u::Array{Float64,2}, K::Array{Float64,3}, lk::Array{Float64,2},
        alpha::Float64, x_::Array{Float64,2}, u_::Array{Float64,2})
    return x_, u_
end

function rollout!(solver::Solver, x::Array{Float64,2}, u::Array{Float64,2}, K::Array{Float64,3}, lk::Array{Float64,2}, alpha::Float64, x_::Array{Float64,2}, u_::Array{Float64,2})
    N = solver.N
    x_[:,1] = solver.obj.x0;
    for k = 2:N
        a = alpha*(lk[:,k-1]);
        delta = (x_[:,k-1] - x[:,k-1])

        u_[:, k-1] = u[:, k-1] - K[:,:,k-1]*delta - a;
        x_[:,k] = solver.fd(x_[:,k-1], u_[:,k-1]);
    end
end


function computecost(obj::Objective, x::Array{Float64,2}, u::Array{Float64,2})
    Q = obj.Q
    R = obj.R
    Qf = obj.Qf
    xf = obj.xf
    N = size(x, 2)

    J = 0;
    for k = 1:N-1
        J = J + (x[:,k] - xf)'*Q*(x[:,k] - xf) + u[:,k]'*R*u[:,k];
    end
    J = 0.5*(J + (x[:,N] - xf)'*Qf*(x[:,N] - xf));
end

function backward_pass!(solver::Solver, x::Array{Float64,2}, u::Array{Float64,2}, K::Array{Float64, 3}, lk::Array{Float64,2})
    n = solver.model.n
    m = solver.model.m
    N = solver.N
    Q = solver.obj.Q
    R = solver.obj.R
    Qf = solver.obj.Qf

    S_prev = Qf
    s_prev = Qf*(x[:,N] - solver.obj.xf)
    s_prev = reshape(s_prev, 1, n)
    # S = zeros(n,n,N)
    # S[:,:,N] = Qf
    # s = zeros(n,N)
    # s[:,N] = Qf*(x[:,N] - solver.obj.xf)
    # K = zeros(m,n,N)
    # lk = zeros(m,N)
    vs1 = 0
    vs2 = 0

    mu_reg = 0;

    for k = N-1:-1:1
        q = Q*(x[:,k] - solver.obj.xf);
        r = R*(u[:,k]);
        A, B = solver.f_jacobian(x[:,k], u[:,k], solver.dt);
        # A = solver.fx(x[:,k], u[:,k], solver.dt)
        # B = solver.fu(x[:,k], u[:,k], solver.dt)
        C1 = q' + s_prev*A;  # 1 x n
        C2 = r' + s_prev*B;  # 1 x m
        C3 = Q + A'*S_prev*A; # n x n
        C4 = R + B'*(S_prev + mu_reg*eye(n))*B; # m x m
        C5 = Array(B'*(S_prev + mu_reg*eye(n))*A);  # m x n

        # regularization
        if any(eigvals(C4).<0)
            mu_reg = mu_reg + 1;
            k = N-1;
            println("REG")
        end

        K[:,:,k] = C4\C5;
        lk[:,k] = C4\C2';
        s_prev = C1 - C2*K[:,:,k] + lk[:,k]'*C4*K[:,:,k] - lk[:,k]'*C5;
        S_prev = C3 + K[:,:,k]'*C4*K[:,:,k] - K[:,:,k]'*C5 - C5'*K[:,:,k];

        vs1 = vs1 + float(lk[:,k]'*C2')[1];
        vs2 = vs2 + float(lk[:,k]'*C4*lk[:,k]);

    end

    return K, lk, vs1, vs2

end

function forwardpass!(x_, u_, solver::Solver, x::Array{Float64,2}, u::Array{Float64,2}, K::Array{Float64,3}, lk::Array{Float64,2}, vs1::Float64, vs2::Float64, c1::Float64=0.25, c2::Float64=0.75)

    # Compute original cost
    J_prev = computecost(solver.obj, x, u)

    # update control, roll out new policy, calculate new cost
    # u_ = zeros(solver.model.m, solver.N)
    # x_ = similar(x);
    # J_prev = copy(J);
    J = Inf;
    alpha = 1.0;
    iter = 0;
    dV = Inf;
    z = 0;

    while J > J_prev || z < c1 || z > c2
        # x_, u_ = iLQR.rollout!(solver, x, u, K, lk, alpha)
        iLQR.rollout!(solver, x, u, K, lk, alpha, x_, u_)

        # Calcuate cost
        J = iLQR.computecost(solver.obj, x_, u_)

        dV = alpha*vs1 + (alpha^2)*vs2/2
        z = (J_prev - J)/dV[1,1]
        alpha = alpha/2;
        iter = iter + 1;

        if iter > 200
            println("max iterations")
            break
        end
    end

    return J

end

function solve(solver::Solver; iterations=10)
    u = 10*rand(m,N-1)
    solve(solver, u, iterations=iterations)
end

function solve(solver::Solver, u::Array; iterations=10)
    n = solver.model.n
    m = solver.model.m
    N = solver.N

    x = zeros(n,N)
    x_ = similar(x)
    u_ = similar(u)
    x[:,1] = solver.obj.x0;

    K = zeros(m,n,N)
    lk = zeros(m,N)

    # first roll-out
    iLQR.rollout!(solver, x, u)

    ## iterations of iLQR using my derivation
    # improvement criteria
    c1 = 0.25;
    c2 = 0.75;

    for i = 1:iterations
        K, lk, vs1, vs2 = iLQR.backward_pass!(solver, x, u, K, lk)
        J = iLQR.forwardpass!(x_, u_, solver, x, u, K, lk, vs1, vs2)

        x = copy(x_)
        u = copy(u_)
        println("Cost: $J")
    end

    return x, u
end

end
