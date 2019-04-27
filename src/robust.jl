using ForwardDiff, Plots, LinearAlgebra, BenchmarkTools, MatrixCalculus,
    PartedArrays, TrajectoryOptimization

## DIRTREL tests
function pendulum_dynamics_uncertain!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},w::AbstractVector{T}) where T
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    ẋ[1] = x[2]
    ẋ[2] = (u[1] - (m + w[1])*g*lc*sin(x[1]) - b*x[2])/I
end

function rk4_uncertain!(f!::Function, dt::T) where T
    # Runge-Kutta 4
    fd!(xdot,x,u,w,dt=dt) = begin
        k1 = zero(xdot)
        k2 = zero(xdot)
        k3 = zero(xdot)
        k4 = zero(xdot)
        f!(k1, x, u, w);         k1 *= dt;
        f!(k2, x + k1/2, u, w); k2 *= dt;
        f!(k3, x + k2/2, u, w); k3 *= dt;
        f!(k4, x + k3, u, w);    k4 *= dt;
        copyto!(xdot, x + (k1 + 2*k2 + 2*k3 + k4)/6)
    end
end


function midpoint_uncertain!(f!::Function, dt::T) where T
    fd!(xdot,x,u,w,dt=dt) = begin
        f!(xdot,x,u,w)
        xdot .*= dt/2.
        f!(xdot, x + xdot, u, w)
        copyto!(xdot,x + xdot*dt)
    end
end

function f_augmented_uncertain!(f!::Function, nx::Int, nu::Int, nw::Int)
    f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1:nx], S[nx .+ (1:nu)], S[(nx+nu) .+ (1:nw)])
end

function pack(X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(X)
    Z̄ = [k != N ? [X[k];U[k]] : X[N] for k = 1:N]
    Z = vcat(Z̄...)
end

function unpack(Z::Vector{T},nx::Int,nu::Int,N::Int) where T
    idx = nx+nu

    X = Vector{T}[]
    U = Vector{T}[]

    for k = 1:N-1
        z = Z[((k-1)*idx+1):k*idx]
        push!(X,z[1:nx])
        push!(U,z[nx .+ (1:nu)])
    end
    push!(X,Z[end-nx+1:end])

    return X,U
end

function robust_cost(Z::Vector{T},∇f::Function,E1::AbstractArray,
        D::AbstractArray,Q::AbstractArray,R::AbstractArray,Qf::AbstractArray,
        Qr::AbstractArray,Rr::AbstractArray,Qfr::AbstractArray,nx::Int,nu::Int,
        nw::Int,N::Int) where T

    TT = eltype(Z)
    A = zeros(TT,nx,nx,N-1)
    B = zeros(TT,nx,nu,N-1)
    G = zeros(TT,nx,nw,N-1)

    K = zeros(TT,nu,nx,N-1)

    E = zeros(TT,nx,nx,N)
    H = zeros(TT,nx,nw)


    idx = nx+nu
    z = zeros(TT,nx+nu+nw)

    for k = 1:N-1
        z[1:nx] = Z[((k-1)*idx) .+ (1:nx)]
        z[nx .+ (1:nu)] = Z[(nx + (k-1)*idx) .+ (1:nu)]

        F = ∇f(z)

        A[:,:,k] = F[:,1:nx]
        B[:,:,k] = F[:,nx .+ (1:nu)]
        G[:,:,k] = F[:,nx+nu .+ (1:nw)]

    end

    P = Qf
    for k = N-1:-1:1
        K[:,:,k] = (R + B[:,:,k]'*P*B[:,:,k])\(B[:,:,k]'*P*A[:,:,k])
        P = Q + K[:,:,k]'*R*K[:,:,k] + (A[:,:,k] - B[:,:,k]*K[:,:,k])'*P*(A[:,:,k] - B[:,:,k]*K[:,:,k]);
    end

    for k = 1:N-1
        E[:,:,k+1] = (A[:,:,k] - B[:,:,k]*K[:,:,k])*E[:,:,k]*(A[:,:,k] - B[:,:,k]*K[:,:,k])' + (A[:,:,k] - B[:,:,k]*K[:,:,k])*H*G[:,:,k]' + G[:,:,k]*H'*(A[:,:,k] - B[:,:,k]*K[:,:,k])' + G[:,:,k]*D*G[:,:,k]'
        H = (A[:,:,k] - B[:,:,k]*K[:,:,k])*H + G[:,:,k]*D
    end

    ℓ = 0.
    for k = 1:N-1
        ℓ += tr((Qr + K[:,:,k]'*Rr*K[:,:,k])*E[:,:,k])
    end
    ℓ += tr(Qfr*E[:,:,N])

    return ℓ
end

function gen_robust_cost(_rcf::Function,_dynamics::Function,
        D::AbstractArray,E1::AbstractArray,Q::AbstractArray,R::AbstractArray,
        Qf::AbstractArray,Qr::AbstractArray,Rr::AbstractArray,Qfr::AbstractArray,nx::Int,nu::Int,nw::Int,N::Int)

    dynamics = f_augmented_uncertain!(_dynamics,nx,nu,nw)

    _∇f(z,nx) = ForwardDiff.jacobian(dynamics,zeros(eltype(z),nx),z)
    function gen_∇f(_∇f::Function,nx::Int)
        ∇f(z) = _∇f(z,nx)
    end
    ∇f = gen_∇f(_∇f,nx)

    function rcf(Z::Vector{T}) where T
        _rcf(Z,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
    end

    return rcf
end

N = 5
nx,nu,nw = 2,1,1;

dt = 0.1
x0 = [0.;0.]

# allocate trajectories
X = [zeros(nx) for k = 1:N]
U = [ones(nu) for k = 1:N-1]

# cost functions
R = Rr = [0.1]
Q = Qr = [10. 0.; 0. 1.]
Qf = Qfr = [100. 0.; 0. 100.]

# uncertainty
D = [0.2^2]
E1 = zeros(nx,nx)

# discrete dynamics
pendulum_discrete! = rk4_uncertain!(pendulum_dynamics_uncertain!,dt)

# rollout initial state trajectory
X[1] .= x0
for k = 1:N-1
    pendulum_discrete!(X[k+1],X[k],U[k],zeros(nw),dt)
end
X
Z = pack(X,U)
xx,uu = unpack(Z,nx,nu,N)

rcf = gen_robust_cost(robust_cost,pendulum_discrete!,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)

c_fd = rcf(Z)
dc_fd = ForwardDiff.gradient(rcf,Z)

###

x_part = NamedTuple{(:x,:δx)}((1:nx,nx .+ (1:nx^2)))
Xr = [BlockArray(zeros(nx+nx^2),x_part) for k = 1:N]

u_part = NamedTuple{(:u,:δu)}((1:nu,nu .+ (1:nu^2)))
Ur = [BlockArray(zeros(nu+nu^2),u_part) for k = 1:N-1]

Xr[1].δx
Ur[1].δu

function set_controls!(Ur::PartedVecTrajectory{T},U0::VectorTrajectory{T}) where T
    N = length(Ur)
    for k = 1:N
        Ur[k].u .= U0[k]
    end
end

set_controls!(Ur,U)

function rollout_uncertain!(Xr::PartedVecTrajectory{T},Ur::PartedVecTrajectory{T},x0::AbstractArray=zeros(T,length(Xr[1].x)),dt::T=1.0) where T
    Xr[1].x .= x0
    for k = 1:N-1
        pendulum_discrete!(Xr[k+1].x,Xr[k].x,Ur[k].u,zeros(nw),dt)
    end
end

rollout_uncertain!(Xr,Ur,x0,dt)
