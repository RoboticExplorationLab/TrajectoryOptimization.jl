using ForwardDiff, Plots, LinearAlgebra, BenchmarkTools, MatrixCalculus,
    PartedArrays, TrajectoryOptimization

## DIRTREL tests
# function pendulum_dynamics_uncertain!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},w::AbstractVector{T}) where T
#     m = 1.
#     l = 0.5
#     b = 0.1
#     lc = 0.5
#     I = 0.25
#     g = 9.81
#     ẋ[1] = x[2]
#     ẋ[2] = (u[1] - (m + w[1])*g*lc*sin(x[1]) - b*x[2])/I
# end

# function rk4_uncertain!(f!::Function, dt::T) where T
#     # Runge-Kutta 4
#     fd!(xdot,x,u,w,dt=dt) = begin
#         k1 = zero(xdot)
#         k2 = zero(xdot)
#         k3 = zero(xdot)
#         k4 = zero(xdot)
#         f!(k1, x, u, w);         k1 *= dt;
#         f!(k2, x + k1/2, u, w); k2 *= dt;
#         f!(k3, x + k2/2, u, w); k3 *= dt;
#         f!(k4, x + k3, u, w);    k4 *= dt;
#         copyto!(xdot, x + (k1 + 2*k2 + 2*k3 + k4)/6)
#     end
# end


# function midpoint_uncertain!(f!::Function, dt::T) where T
#     fd!(xdot,x,u,w,dt=dt) = begin
#         f!(xdot,x,u,w)
#         xdot .*= dt/2.
#         f!(xdot, x + xdot, u, w)
#         copyto!(xdot,x + xdot*dt)
#     end
# end

# function f_augmented_uncertain!(f!::Function, nx::Int, nu::Int, nw::Int)
#     f_aug!(dS::AbstractArray, S::Array) = f!(dS, S[1:nx], S[nx .+ (1:nu)], S[(nx+nu) .+ (1:nw)])
# end

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

function unpack_cost_hessian(Z::Matrix{T},nx::Int,nu::Int,N::Int) where T
    """
    Z = [ℓw_xx ℓw_ux'; ℓw_ux ℓw_uu]

    """
    idx = nx+nu

    ℓw_xx = Matrix{T}[]
    ℓw_ux = Matrix{T}[]
    ℓw_uu = Matrix{T}[]

    for k = 1:N-1
        _idx = ((k-1)*idx+1):k*idx
        z = Z[_idx,_idx]
        push!(ℓw_xx,z[1:nx,1:nx])
        push!(ℓw_ux,z[nx .+ (1:nu),1:nx])
        push!(ℓw_uu,z[nx .+ (1:nu),nx .+ (1:nu)])
    end
    push!(ℓw_xx,Z[end-nx+1:end,end-nx+1:end])

    return ℓw_xx,ℓw_ux,ℓw_uu
end

function robust_cost(Z::Vector{T},∇f::Function,E1::AbstractArray,
        D::AbstractArray,Q::AbstractArray,R::AbstractArray,Qf::AbstractArray,
        Qr::AbstractArray,Rr::AbstractArray,Qfr::AbstractArray,nx::Int,nu::Int,
        nw::Int,N::Int) where T

    A = zeros(T,nx,nx,N-1)
    B = zeros(T,nx,nu,N-1)
    G = zeros(T,nx,nw,N-1)

    K = zeros(T,nu,nx,N-1)

    E = zeros(T,nx,nx,N)
    H = zeros(T,nx,nw)


    idx = nx+nu
    z = zeros(T,nx+nu+nw)

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
        P = Q + A[:,:,k]'*P*A[:,:,k] - A[:,:,k]'*P*B[:,:,k]*K[:,:,k]
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

function robust_cost(X::VectorTrajectory{T},U::VectorTrajectory{T},∇f::Function,E1::AbstractArray,
        D::AbstractArray,Q::AbstractArray,R::AbstractArray,Qf::AbstractArray,
        Qr::AbstractArray,Rr::AbstractArray,Qfr::AbstractArray,nx::Int,nu::Int,
        nw::Int,N::Int) where T

    A = [zeros(nx,nx) for k = 1:N-1]
    B = [zeros(nx,nu) for k = 1:N-1]
    G = [zeros(nx,nw) for k = 1:N-1]

    K = [zeros(nu,nx) for k = 1:N-1]

    E = [zeros(nx,nx) for k = 1:N]
    H = zeros(nx,nw)


    idx = nx+nu
    z = zeros(nx+nu+nw)

    for k = 1:N-1
        z[1:nx] = X[k][1:nx]
        z[nx .+ (1:nu)] = U[k][1:nu]
        F = ∇f(z)

        A[k] .= F[:,1:nx]
        B[k] .= F[:,nx .+ (1:nu)]
        G[k] .= F[:,nx+nu .+ (1:nw)]

    end

    P = Qf
    for k = N-1:-1:1
        K[k] .= (R + B[k]'*P*B[k])\(B[k]'*P*A[k])
        P = Q + A[k]'*P*A[k] - A[k]'*P*B[k]*K[k]
    end

    for k = 1:N-1
        E[k+1] = (A[k] - B[k]*K[k])*E[k]*(A[k] - B[k]*K[k])' + (A[k] - B[k]*K[k])*H*G[k]' + G[k]*H'*(A[k] - B[k]*K[k])' + G[k]*D*G[k]'
        H = (A[k] - B[k]*K[k])*H + G[k]*D
    end

    ℓ = 0.
    for k = 1:N-1
        ℓ += tr((Qr + K[k]'*Rr*K[k])*E[k])
    end
    ℓ += tr(Qfr*E[N])

    return ℓ
end

struct RobustCost <: CostFunction
    ℓw::Function
    ∇ℓw::AbstractArray
    ∇²ℓw::AbstractArray
    D::AbstractArray
    E1::AbstractArray
    Q::AbstractArray
    R::AbstractArray
    Qf::AbstractArray
    Qr::AbstractArray
    Rr::AbstractArray
    Qfr::AbstractArray
    nx::Int
    nu::Int
    nw::Int
    N::Int

    function RobustCost(dynamics::Function,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx::Int,nu::Int,nw::Int,N::Int,rcf::Function=robust_cost)

        # augment dynamics
        dynamics_aug = f_augmented_uncertain!(dynamics,nx,nu,nw)

        # create uncertain dynamics jacobian that takes a single input (and works with ForwardDiff)
        _∇f(z,nx) = ForwardDiff.jacobian(dynamics_aug,zeros(eltype(z),nx),z) #TODO see if this can be taken from model instead of reconstructed
        function gen_∇f(_∇f::Function,nx::Int)
            ∇f(z) = _∇f(z,nx)
        end
        ∇f = gen_∇f(_∇f,nx)

        # create robust cost function that takes a single input
        function ℓw(Z::Vector{T}) where T
            rcf(Z,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
        end

        function ℓw(X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
            rcf(X,U,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
        end

        # allocate memory for robust cost expansion
        s = (N-1)*(nx+nu) + nx
        ∇ℓw = zeros(s)
        ∇²ℓw = zeros(s,s)
        K = [zeros(nu,nx) for k = 1:N-1]

        new(ℓw,∇ℓw,∇²ℓw,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N,K)
    end
end

function cost(c::RobustCost, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
    s = (c.N-1)*(c.nx+c.nu) + c.nx
    Z = pack(X,U)
    out = DiffResults.HessianResult(Z)
    ForwardDiff.hessian!(out,c.ℓw,Z)
    c.∇²ℓw .= DiffResults.hessian(out)
    c.∇ℓw .= DiffResults.gradient(out)
    return DiffResults.value(out)
end

function cost_expansion!(Q::ExpansionTrajectory{T},robust_cost::RobustCost,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = robust_cost.N
    Qx,Qu = unpack(robust_cost.∇ℓw,robust_cost.nx,robust_cost.nu,N)
    Qxx,Qux,Quu = unpack_cost_hessian(robust_cost.∇²ℓw,robust_cost.nx,robust_cost.nu,N)

    for k = 1:N-1
        Q[k].x .+= Qx[k]
        Q[k].u .+= Qu[k]
        Q[k].xx .+= Qxx[k]
        Q[k].ux .+= Qux[k]
        Q[k].uu .+= Quu[k]
    end
    Q[N].x .+= Qx[N]
    Q[N].xx .+= Qxx[N]

    return nothing
end

struct RobustObjective <: AbstractObjective
    obj::AbstractObjective
    robust_cost::RobustCost
end

function cost(robust_obj::RobustObjective, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
    J = cost(robust_obj.obj,X,U)
    J += cost(robust_obj.robust_cost,X,U)
    return J
end

function cost_expansion!(Q::ExpansionTrajectory{T},robust_obj::RobustObjective,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    cost_expansion!(Q,robust_obj.obj,X,U)
    cost_expansion!(Q,robust_obj.robust_cost,X,U)
end


# function gen_robust_cost(_rcf::Function,_dynamics::Function,
#         D::AbstractArray,E1::AbstractArray,Q::AbstractArray,R::AbstractArray,
#         Qf::AbstractArray,Qr::AbstractArray,Rr::AbstractArray,Qfr::AbstractArray,nx::Int,nu::Int,nw::Int,N::Int)
#
#     dynamics = f_augmented_uncertain!(_dynamics,nx,nu,nw)
#
#     _∇f(z,nx) = ForwardDiff.jacobian(dynamics,zeros(eltype(z),nx),z)
#     function gen_∇f(_∇f::Function,nx::Int)
#         ∇f(z) = _∇f(z,nx)
#     end
#     ∇f = gen_∇f(_∇f,nx)
#
#     function rcf(Z::Vector{T}) where T
#         _rcf(Z,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
#     end
#
#     return rcf
# end

# N = 5
# nx,nu,nw = 2,1,1;
#
# dt = 0.1
# x0 = [0.;0.]
#
# # allocate trajectories
# X = [zeros(nx) for k = 1:N]
# U = [ones(nu) for k = 1:N-1]
#
# # cost functions
# R = Rr = [0.1]
# Q = Qr = [10. 0.; 0. 1.]
# Qf = Qfr = [100. 0.; 0. 100.]
#
# # uncertainty
# D = [0.2^2]
# E1 = zeros(nx,nx)
#
# # discrete dynamics
# pendulum_discrete! = rk4_uncertain!(pendulum_dynamics_uncertain!,dt)
#
# # rollout initial state trajectory
# X[1] .= x0
# for k = 1:N-1
#     pendulum_discrete!(X[k+1],X[k],U[k],zeros(nw),dt)
# end
# X
# Z = pack(X,U)
# xx,uu = unpack(Z,nx,nu,N)
#
# rcf = gen_robust_cost(robust_cost,pendulum_discrete!,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
#
# c_fd = rcf(Z)
# dc_fd = ForwardDiff.gradient(rcf,Z)
#
# ###
#
# x_part = NamedTuple{(:x,:δx)}((1:nx,nx .+ (1:nx^2)))
# Xr = [BlockArray(zeros(nx+nx^2),x_part) for k = 1:N]
#
# u_part = NamedTuple{(:u,:δu)}((1:nu,nu .+ (1:nu^2)))
# Ur = [BlockArray(zeros(nu+nu^2),u_part) for k = 1:N-1]
#
# Xr[1].δx
# Ur[1].δu
#
function set_controls!(Ur::PartedVecTrajectory{T},U0::VectorTrajectory{T}) where T
    N = length(Ur)
    for k = 1:N
        Ur[k].u .= U0[k]
    end
end
#
# set_controls!(Ur,U)
#
function rollout_uncertain!(Xr::PartedVecTrajectory{T},Ur::PartedVecTrajectory{T},x0::AbstractArray=zeros(T,length(Xr[1].x)),dt::T=1.0) where T
    Xr[1].x .= x0
    for k = 1:N-1
        pendulum_discrete!(Xr[k+1].x,Xr[k].x,Ur[k].u,zeros(nw),dt)
    end
end
#
# rollout_uncertain!(Xr,Ur,x0,dt)
