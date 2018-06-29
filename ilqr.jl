module iLQR

struct Model
    f::Function
    n::Int
    m::Int
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
    f_midpoint::Function
    fx::Function
    fu::Function
    N::Int
    function Solver(model, obj, fx, fu, dt)
        obj_n = size(obj.Q, 1)
        obj_m = size(obj.R, 1)
        @assert obj_n == model.n
        @assert obj_m == model.m

        f_mid = f_midpoint(model.f, dt)
        N = Int(floor(obj.tf/dt));
        new(model, obj, dt, f_mid, fx, fu, N)

    end
end


function f_midpoint(f::Function, dt::Float64)
    dynamics_midpoint(x,u)  = x + f(x + f(x,u)*dt/2, u)*dt
end

function rollout!(solver::Solver, x::Array{Float64,2}, u::Array{Float64,2})
    N = size(x, 2)
    for k = 2:N
        x[:, k] = solver.f_midpoint(x[:,k-1], u[:,k-1])
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
        x_[:,k] = solver.f_midpoint(x_[:,k-1], u_[:,k-1]);
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
        A = solver.fx(x[:,k], solver.dt);
        B = solver.fu(x[:,k], solver.dt)
        C1 = q' + s_prev*A;  # 1 x n
        C2 = r' + s_prev*B;  # 1 x m
        C3 = Q + A'*S_prev*A; # n x n
        C4 = R + B'*(S_prev + mu_reg*eye(n))*B; # m x m
        C5 = Array(B'*(S_prev + mu_reg*eye(n))*A);  # m x n

        # regularization
        if any(eigvals(C4).<0)
            mu_reg = mu_reg + 1;
            k = N-1;
        end

        K[:,:,k] = C4\C5';
        lk[:,k] = C4\C2';
        s_prev = C1 - C2*K[:,:,k] + lk[:,k]'*C4*K[:,:,k] - lk[:,k]'*C5;
        S_prev = C3 + K[:,:,k]'*C4*K[:,:,k] - K[:,:,k]'*C5 - C5'*K[:,:,k];

        vs1 = vs1 + float(lk[:,k]'*C2');
        vs2 = vs2 + float(lk[:,k]'*C4*lk[:,k]);

    end

    return K, lk

end

function forwardpass!(solver::Solver, x, u, K, lk, x_, u_)

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

    while J > J_prev
        # x_, u_ = iLQR.rollout!(solver, x, u, K, lk, alpha)
        iLQR.rollout!(solver, x, u, K, lk, alpha, x_, u_)

        # Calcuate cost
        J = iLQR.computecost(solver.obj, x_, u_)

        alpha = alpha/2;
        iter = iter + 1;

        if iter > 200
            println("max iterations")
            break
        end
    end

    return J

end

end
