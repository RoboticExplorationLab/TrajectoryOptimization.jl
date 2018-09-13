# Pendulum
n = 2
m = 1
Q = 0.3*Matrix(I,n,n)
R = 0.3*Matrix(I,m,m)
Qf = 30.0*Matrix(I,n,n)
dt = 0.001
x0 = [0.0; 0.0]
xf = [pi; 0.0]
tf = 5.0

N = 50

function f(x::Vector,u::Vector)
    m = 1;
    l = .5;
    b = 0.1;
    lc = .5;
    I = .25;
    g = 9.81;

    q = x[1];
    qd = x[2];

    qdd = (u .- m*g*lc*sin(q) .- b*qd)/I;

    [qd; qdd]
end

# Discrete RK4
function rk4(f::Function,dt::Float64)
    fd(x,u) = begin
        # xdot1 = f(x,u);
        # xdot2 = f(x+.5*dt*xdot1,u)
        # xdot3 = f(x+.5*dt*xdot2,u);
        # xdot4 = f(x+dt*xdot3,u);
        #
        # x + (dt/6)*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4);
        k1 = dt*f(x,u)
        k2 = dt*f(x + k1/2.0,u)
        k3 = dt*f(x + k2/2.0,u)
        k4 = dt*f(x + k3, u)
        x + 1.0/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)

    end
end

function Jacobians(f::Function, x::Vector, u::Vector)
    f1(a) = f(a,u)
    f2(b) = f(x,b)

    A = ForwardDiff.jacobian(f1,x)
    B = ForwardDiff.jacobian(f2,u)
    A, B
end

function cost(x::Matrix, u::Matrix, N::Int64, Q::Matrix, R::Matrix, Qf::Matrix, xf::Vector, dt::Float64)
    J = 0.0
    for k = 1:N-1
        J += dt*0.5*((x[:,k] - xf)'*Q*(x[:,k] - xf) + u[:,k]'*R*u[:,k])
    end
    J += 0.5*(x[:,N] - xf)'*Qf*(x[:,N] - xf)
    J
end

function rollout(fd::Function, x::Matrix, u::Matrix, x0::Vector, n::Int64, m::Int64, N::Int64, K::Array{Float64,3}=zeros(m,n,N), d::Matrix=zeros(m,N), alpha::Float64=0.0)
    x_ = zero(x)
    u_ = zero(u)
    x_[:,1] = x0
    for k = 1:N-1
        Δx = x_[:,k] - x[:,k]
        u_[:,k] = u[:,k] + K[:,:,k]*Δx + alpha*d[:,k]
        x_[:,k+1] = fd(x_[:,k],u_[:,k])
    end
    x_, u_
end

x = zeros(n,N)
x_ = zeros(n,N)
u = zeros(m,N-1)
u_ = zeros(m,N-1)
fd = rk4(f,dt)

x_, u_ = rollout(fd,x,u,x_,u_,x0,n,m,N)
x_



function backwardpass(fd::Function,jac::Function, x::Matrix, u::Matrix, n::Int64, m::Int64, N::Int64, Q::Matrix, R::Matrix, Qf::Matrix, xf::Vector, dt::Float64, lambda::Float64)
    diverge = false

    K = zeros(m,n,N-1)
    d = zeros(m,N-1)

    S = zeros(n,n,N)
    s = zeros(n,N)

    S[:,:,N] = Qf
    s[:,N] = Qf*(x[:,N] - xf)

    Δv = [0.0 0.0]

    k = N - 1
    while k >= 1
        A, B = jac(fd,x[:,k],u[:,k])
        # if k == N-1
        #     println("A: $A\n B: $B")
        # end

        Qx = dt*Q*(x[:,k] - xf) + A'*vec(s[:,k+1])
        Qu = dt*R*u[:,k] + B'*vec(s[:,k+1])
        Qxx = dt*Q + A'*S[:,:,k+1]*A
        Quu = dt*R + B'*S[:,:,k+1]*B + lambda*Matrix(I,m,m)
        Qxu = dt*zeros(n,m) + A'*S[:,:,k+1]*B

        if !isposdef(Quu)
            #regularize
            println("regularization need on bp")
            diverge = true
            return diverge, K, d, Δv
        end

        d[:,k] = -Quu\Qu
        K[:,:,k] = -Quu\Qxu'

        Δv += [vec(d[:,k])'*vec(Qu) 0.5*vec(d[:,k])'*Quu*vec(d[:,k])]

        s[:,k] = Qx + K[:,:,k]'*Quu*vec(d[:,k]) + K[:,:,k]'*vec(Qu) + Qxu*vec(d[:,k])
        S[:,:,k] = Qxx + K[:,:,k]'*Quu*K[:,:,k] + K[:,:,k]'*Qxu' + Qxu*K[:,:,k]
        S[:,:,k] = 0.5*(S[:,:,k] + S[:,:,k]')

        k -= 1
    end
    diverge, K, d, Δv
end

diverge, K, d, Δv = backwardpass(fd,Jacobians, x, u, n, m, N, Q, R, Qf, xf, dt, lambda)


function forwardpass(fd::Function,x::Matrix, u::Matrix, n::Int64,m::Int64,N::Int64,Q::Matrix, R::Matrix, Qf::Matrix, x0::Vector, xf::Vector, dt::Float64, K::Array{Float64,3},d::Matrix,Δv::Array{Float64,2},z_min::Float64=0.0)

    Jprev = cost(x, u, N, Q, R, Qf, xf, dt)
    Alpha = 10.0.^linspace(0,-3,11)

    Jbest = Inf
    α = 1.0
    for i = 1:size(Alpha,1)
        x_,u_ = rollout(fd, x, u, x0, n, m, N, K, d, Alpha[i])
        expected = -Alpha[i]*(Δv[1] + Alpha[i]*Δv[2])
        J = cost(x_, u_, N, Q, R, Qf, xf, dt)
        z = (Jprev - J)/expected

        if z > z_min
            return x_, u_, J, true
        elseif expected <= 0
            println("$expected")
            println("warning - no cost improvement")
        end
    end
    println("no cost improvement")
    return x, u, Jprev, false
end

x_,u_,Jprev, fp_status = forwardpass(fd, x, u, n, m, N, Q, R, Qf, x0, xf, dt, K,d,Δv)
x


function solve(fd::Function,jac::Function,u::Matrix,n::Int64,m::Int64,N::Int64,Q::Matrix, R::Matrix, Qf::Matrix, x0::Vector, xf::Vector, dt::Float64, lambda::Float64,dlambda::Float64,lambdaFactor::Float64,lambdaMax::Float64,lambdaMin::Float64, eps_func::Float64=1.0e-5, eps_grad::Float64=1e-5, max_iter::Int64=250)
    x = zeros(n,N)

    iter = 0
    # initial rollout
    x, = rollout(fd,x,u,x0,n,m,N)
    Jprev = cost(x, u, N, Q, R, Qf, xf, dt)
    println("Original cost: $Jprev")

    for i = 1:max_iter
        iter += 1

        # backward pass
        diverge, K, d, Δv = backwardpass(fd,Jacobians, x, u, n, m, N, Q, R, Qf, xf, dt, lambda)

        # check for backward pass divergence
        if diverge
            dlambda = max(dlambda*lambdaFactor, lambdaFactor)
            lambda = max(lambda*dlambda, lambdaMin)
            if lambda > lambdaMax
                println("max lambda reached - terminating solve")
                return x_, u_, J
            end
            continue
        end

        # check for gradient convergence
        grad = mean(maximum(abs.(d[:])./(abs.(u[:]) .+ 1)))
        if grad < eps_grad && lambda < 1e-5
            dlambda = min(dlambda/lambdaFactor, 1.0/lambdaFactor)
            lambda = lambda*dlambda*(lambda>lambdaMin)
            println("gradient tolerance met - solve complete")
            return x_, u_, J
        end

        # forward pass
        x_, u_, J, fp_status = forwardpass(fd,x, u, n, m, N, Q, R, Qf, x0, xf, dt, K, d, Δv)

        if fp_status # cost improvement
            # decrease lambda
            dlambda = min(dlambda/lambdaFactor, 1.0/lambdaFactor)
            lambda = lambda*dlambda*(lambda > lambdaMin)

            # check for cost convergence
            if (Jprev-J) < eps_func
                println("dJ tolerance met - solve complete")
                return x_, u_, J
            end
        else # no cost improvement
            dlambda = max(dlambda*lambdaFactor, lambdaFactor)
            lambda = max(lambda*dlambda, lambdaMin)

            if lambda > lambdaMax
                println("max lambda reached - terminating solve")
                break
            end
        end

        println("Iteration $i\n -cost: $J \n -grad: $grad \n -lambda: $lambda")


        x = copy(x_)
        u = copy(u_)
        Jprev = copy(J)

    end
    println("no solve")
end

lambda = 1.0
dlambda = 1.0

lambdaFactor = 1.6
lambdaMax = 1.0e10
lambdaMin = 1.0e-6

u = zeros(m,N-1)
solve(fd,Jacobians,u,n,m,N,Q, R, Qf, x0, xf, dt,lambda,dlambda,lambdaFactor,lambdaMax,lambdaMin)
