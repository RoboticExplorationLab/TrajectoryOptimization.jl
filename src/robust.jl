using ForwardDiff, Plots, LinearAlgebra, BenchmarkTools, MatrixCalculus,
    PartedArrays, TrajectoryOptimization

"Time-varying Linear Quadratic Regulator; returns optimal linear feedback matrices and optimal cost-to-go"
function tvlqr_dis(prob::Problem{T},Q::AbstractArray{T},R::AbstractArray{T},Qf::AbstractArray{T}) where T
    n = prob.model.n; m = prob.model.m; N = prob.N
    dt = prob.dt

    K  = [zeros(T,m,n) for k = 1:N-1]
    ∇F = [BlockArray(zeros(T,n,length(prob.model)),create_partition2(prob.model)) for k = 1:N-1]
    P  = [zeros(T,n,n) for k = 1:N]

    jacobian!(∇F,prob.model,prob.X,prob.U,prob.dt)

    P[N] .= Qf

    for k = N-1:-1:1
        A, B = ∇F[k].xx, ∇F[k].xu
        K[k] .= (R*dt + B'*P[k+1]*B)\(B'*P[k+1]*A)
        P[k] .= Q*dt + K[k]'*R*K[k]*dt + (A - B*K[k])'*P[k+1]*(A - B*K[k])
    end

    return K, P
end

"Robust model from model"
function robust_model(model::Model{Uncertain,Discrete},E1::AbstractArray,H1::AbstractArray,D::AbstractArray)
    n = model.n; m = model.m; r = model.r
    n̄ = n*(1 + 2*n + r)

    fc! = prob.model.info[:fc]
    integration = prob.model.info[:integration]

    f_robust = robust_dynamics(fc!,E1,H1,D,n,m,r)

    discretize_model(UncertainModel(f_robust, n̄, m, r),integration,prob.dt)
end

function robust_model(model::Model{Uncertain,Continuous},E1::AbstractArray,H1::AbstractArray,D::AbstractArray)
    n = model.n; m = model.m; r = model.r
    n̄ = n*(1 + 2*n + r)

    fc! = prob.model.f

    f_robust = robust_dynamics(fc!,E1,H1,D,n,m,r)

    UncertainModel(f_robust, n̄, m, r)
end

"Robust dynamics, includes nominal, disturbance, and cost-to-go dynamics"
function robust_dynamics(fc::Function,fd::Function,E1::AbstractArray,H1::AbstractArray,
        D::AbstractArray,n::Int,m::Int,r::Int)

    fc_aug = f_augmented_uncertain!(fc,n,m,r) #TODO make sure existing ∇f is ForwardDiff-able, then use that
    ∇fc(z) = ForwardDiff.jacobian(fc_aug,zeros(eltype(z),n),z)

    fd_aug = f_augmented_uncertain!(fd,n,m,r) #TODO make sure existing ∇f is ForwardDiff-able, then use that
    ∇fd(z) = ForwardDiff.jacobian(fd_aug,zeros(eltype(z),n),z)

    idx = (x = 1:n, u = 1:m, w = 1:r, e = n .+ (1:n^2), h = (n+n^2) .+ (1:n*r), p = (n+n^2+n*r) .+ (1:n^2))

    function f_robust(ẏ::AbstractVector{T},y::AbstractVector{T},u::AbstractVector{T},w::AbstractVector{T}) where T
        "
            y = [x;E;H;P]
        "
        x = y[idx.x]

        # E = reshape(y[idx.e],n,n)
        # H = reshape(y[idx.h],n,r)
        P = reshape(y[idx.p],n,n)

        z = [x;u;w]

        Fc = ∇fc(z)
        Fd = ∇fd([z;dt])


        Ac = Fc[:,idx.x]; Bc = Fc[:,n .+ idx.u]; Gc = Fc[:,(n+m) .+ idx.w]
        Ad = Fd[:,idx.x]; Bd = Fd[:,n .+ idx.u]
        K = Kc(Bc,P,R)
        Accl = (Ac - Bc*K)

        println("K:")
        println(K)

        # nominal dynamics
        fc!(view(ẏ,1:n),x,u,w)

        # disturbances
        Ec!(view(ẏ,idx.e),E1,H1,Accl,Gc)
        Hc!(view(ẏ,idx.h),H1,Accl,Gc,D)

        # cost-to-go
        Pc!(view(ẏ,idx.p),P,Ac,Bc,Q,R)
    end

end

"Continuous-time TVLQR optimal feedback matrix"
function Kc(B::AbstractArray,P::AbstractArray,R::AbstractArray)
    R\(B'*P)
end

function Kc!(K::AbstractArray,B::AbstractArray,P::AbstractArray,R::AbstractArray)
    K .= Kc(B,P,R)
    return nothing
end

"Continuous-time TVLQR optimal cost-to-go"
function Pc!(Ṗ::AbstractVector,P::AbstractArray,A::AbstractArray,B::AbstractArray,
        Q::AbstractArray,R::AbstractArray)

        n = size(P,1)
        Ṗ .= -1*reshape(A'*P + P*A - P*B*(R\(B'*P)) + Q,n^2)
end

"Continuous-time disturbance dynamics"
function Ec!(Ė::AbstractVector,E1::AbstractArray,H1::AbstractArray,Acl::AbstractArray,
        G::AbstractArray)
        n = size(E1,1)
        Ė .= reshape(Acl*E1 + G*H1' + E1*Acl' + H1*G',n^2)
end

function Hc!(Ḣ::AbstractVector,H1::AbstractArray,Acl::AbstractArray,
        G::AbstractArray,D::AbstractArray)
        n, r = size(G)
        Ḣ = reshape(Acl*H1 + G*D,n,r)
end

mutable struct RobustCost{T} <: CostFunction
    Q::AbstractArray{T}
    R::AbstractArray{T}
    Qf::AbstractArray{T}

    Qr::AbstractArray{T}
    Rr::AbstractArray{T}
    Qrf::AbstractArray{T}

    ∇fd::Function
    n::Int
    m::Int
    r::Int
    idx::NamedTuple
end

function stage_cost(cost::RobustCost, y::Vector{T}, u::Vector{T}) where T
    idx = cost.idx; n = cost.n; m = cost.m; r = cost.r
    x = y[idx.x]

    E = reshape(y[idx.e],n,n)
    P = reshape(y[idx.p],n,n)
    Bd = ∇fd([x;u;zeros(r);0.])[:,n .+ (1:m)]
    K = K(Bd,P,R)

    tr((cost.Qr + K'*Rr*K)*E)
end

function stage_cost(cost::RobustCost, yN::Vector{T}) where T
    idx = cost.idx; n = cost.n
    E = reshape(y[idx.e],n,n)
    tr(cost.Qrf*E)
end

function cost(c::RobustCost,Y::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(Y)
    ℓ = 0.
    for k = 1:N-1
        ℓ += stage_cost(c,Y[k],U[k])
    end
    ℓ += stage_cost(c,Y[N])
    ℓ
end

# struct RobustCost <: CostFunction
#     ℓw::Function
#     ∇ℓw::AbstractArray
#     ∇²ℓw::AbstractArray
#     E1::AbstractArray
#     H1::AbstractArray
#     D::AbstractArray
#     Q::AbstractArray
#     R::AbstractArray
#     Qf::AbstractArray
#     Qr::AbstractArray
#     Rr::AbstractArray
#     Qfr::AbstractArray
#     nx::Int
#     nu::Int
#     nw::Int
#     N::Int
#
#     function RobustCost(dynamics::Function,E1,H1,D,Q,R,Qf,Qr,Rr,Qfr,nx::Int,nu::Int,nw::Int,N::Int,rcf::Function=robust_cost)
#
#         # augment dynamics
#         dynamics_aug = f_augmented_uncertain!(dynamics,nx,nu,nw)
#
#         # create uncertain dynamics jacobian that takes a single input (and works with ForwardDiff)
#         _∇f(z,nx) = ForwardDiff.jacobian(dynamics_aug,zeros(eltype(z),nx),z) #TODO see if this can be taken from model instead of reconstructed
#         function gen_∇f(_∇f::Function,nx::Int)
#             ∇f(z) = _∇f(z,nx)
#         end
#         ∇f = gen_∇f(_∇f,nx)
#
#         # create robust cost function that takes a single input
#         function ℓw(Z::Vector{T}) where T
#             rcf(Z,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
#         end
#
#         function ℓw(X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
#             rcf(X,U,∇f,E1,D,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
#         end
#
#         # allocate memory for robust cost expansion
#         s = (N-1)*(nx+nu) + nx
#         ∇ℓw = zeros(s)
#         ∇²ℓw = zeros(s,s)
#         K = [zeros(nu,nx) for k = 1:N-1]
#
#         new(ℓw,∇ℓw,∇²ℓw,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
#     end
# end
#
# function cost(c::RobustCost, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
#     return c.ℓw(X,U)
# end
#
# function cost_expansion!(Q::ExpansionTrajectory{T},robust_cost::RobustCost,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
#     N = robust_cost.N
#
#     s = (N-1)*(robust_cost.nx+robust_cost.nu) + robust_cost.nx
#     Z = pack(X,U)
#     out = DiffResults.HessianResult(Z)
#     ForwardDiff.hessian!(out,robust_cost.ℓw,Z)
#     robust_cost.∇²ℓw .= DiffResults.hessian(out)
#     robust_cost.∇ℓw .= DiffResults.gradient(out)
#
#     Qx,Qu = unpack(robust_cost.∇ℓw,robust_cost.nx,robust_cost.nu,N)
#     Qxx,Qux,Quu = unpack_cost_hessian(robust_cost.∇²ℓw,robust_cost.nx,robust_cost.nu,N)
#
#     for k = 1:N-1
#         Q[k].x .+= Qx[k]
#         Q[k].u .+= Qu[k]
#         Q[k].xx .+= Qxx[k]
#         Q[k].ux .+= Qux[k]
#         Q[k].uu .+= Quu[k]
#     end
#     Q[N].x .+= Qx[N]
#     Q[N].xx .+= Qxx[N]
#
#     return nothing
# end

struct RobustObjective <: AbstractObjective
    obj::AbstractObjective
    robust_cost::RobustCost
end

function cost(robust_obj::RobustObjective, X::VectorTrajectory{T}, U::VectorTrajectory{T})::T where T <: AbstractFloat
    J = cost(robust_obj.robust_cost,X,U)
    J += cost(robust_obj.obj,X,U)
    return J
end

function cost_expansion!(Q::ExpansionTrajectory{T},robust_obj::RobustObjective,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    cost_expansion!(Q,robust_obj.robust_cost,X,U)
    cost_expansion!(Q,robust_obj.obj,X,U)
end

function robust_problem(prob::Problem{T},E1::AbstractArray{T},
    H1::AbstractArray{T},D::AbstractArray{T},Q::AbstractArray{T},R::AbstractArray{T},
    Qf::AbstractArray{T},Qr::AbstractArray{T},Rr::AbstractArray{T},Qfr::AbstractArray{T}) where T
    N = prob.N
    @assert all([prob.obj[k] isa QuadraticCost for k = 1:N]) #TODO generic cost

    n = model.n; m = model.m; r = model.r
    n_robust = 2*n^2 + n*r # number of robust parameters in state vector

    # modify cost
    _cost = CostFunction[]
    for k = 1:N-1
        cost_robust = copy(prob.obj[k])
        cost_robust.Q = cat(cost_robust.Q,Diagonal(zeros(n_robust)),dims=(1,2))
        cost_robust.q = [cost_robust.q; zeros(n_robust)]
        cost_robust.H = [cost_robust.H zeros(m,n_robust)]
        push!(_cost,cost_robust)
    end

    cost_robust = copy(prob.obj[N])
    cost_robust.Qf = cat(cost_robust.Qf,Diagonal(zeros(n_robust)),dims=(1,2))
    cost_robust.qf = [cost_robust.qf; zeros(n_robust)]
    push!(_cost,cost_robust)

    _obj = Objective(_cost)

    idx = (x = 1:n, u = 1:m, w = 1:r, e = n .+ (1:n^2), h = (n+n^2) .+ (1:n*r), p = (n+n^2+n*r) .+ (1:n^2))

    model_uncertain = robust_model(prob.model,E1,H1,D)

    robust_cost = RobustCost(Q,R,Qf,Qr,Rr,Qfr,∇fc,n,m,r,idx)

    model_uncertain = robust_model(prob.model,E1,H1,D)

    # con_prob = AbstractConstraintSet[]
    # constrained = is_constrained(prob)
    # for k = 1:N-1
    #     con_uncertain = AbstractConstraint[]
    #     if constrained
    #         for cc in prob.constraints[k]
    #             push!(con_uncertain,robust_constraint(cc,prob.model.n,prob.model.m))
    #         end
    #     end
    #     push!(con_prob,con_uncertain)
    # end
    #
    # if constrained
    #     con_uncertain = AbstractConstraint[]
    #     for cc in prob.constraints[N]
    #         push!(con_uncertain,robust_constraint(cc,prob.model.n))
    #     end
    # else
    #     con_uncertain = Constraint[]
    # end
    # push!(con_prob,con_uncertain)
    rollout!(prob)
    K, P = tvlqr(prob,Q,R,Qf)

    update_problem(prob,model=model_uncertain,obj=RobustObjective(_obj,robust_cost),
        X=[[prob.X[k];ones(n_robust)*NaN32] for k = 1:prob.N],
        x0=[copy(prob.x0);reshape(E1,n^2);reshape(H1,n*r);reshape(P[1],n^2)])

end

"Modify constraint to evaluate all combinations of the disturbed state and control"
function robust_constraint(c::AbstractConstraint,n::Int,m::Int)
    p = c.p
    p_robust = p*(2*n+1)*(2*m+1)

    function rc(v,x,u)
        xw = Vector[]
        uw = Vector[]

        push!(xw,x[1:n])

        for i = 1:2*n
            push!(xw,x[n .+ (((i-1)*n+1):i*n)])
        end
        for j = 1:2*m
            push!(uw,x[(n+2*n^2) .+ (((j-1)*m+1):j*m)])
        end
        push!(uw,u[1:m])

        k = 1
        for _x in xw
            for _u in uw
                c.c(view(v,((k-1)*p+1:k*p)),_x,_u)
                k += 1
            end
        end
    end

    function ∇rc(V,x,u)
        xw = Vector[]
        uw = Vector[]

        push!(xw,x[1:n])

        for i = 1:2*n
            push!(xw,x[n .+ (((i-1)*n+1):i*n)])
        end
        for j = 1:2*m
            push!(uw,x[(n+2*n^2) .+ (((j-1)*m+1):j*m)])
        end
        push!(uw,u[1:m])

        k = 1
        for (i,_x) in enumerate(xw)
            for (j,_u) in enumerate(uw)
                _V = zeros(p,n+m)
                idx_p = (((k-1)*p+1):k*p)
                c.∇c(_V,_x,_u)
                idx = [((i-1)*n+1:i*n)...,((n+2*(n^2)) .+ ((j-1)*m+1:j*m))...]
                copyto!(view(V,idx_p,idx),_V)
                k += 1
            end
        end
    end
    typeof(c)(rc,∇rc,p_robust,c.label,c.inds)
end

function robust_constraint(c::AbstractConstraint,n::Int)
    p = c.p
    p_robust = p*(2*n+1)

    function rc(v,x)
        xw = Vector[]

        push!(xw,x[1:n])

        for i = 1:2*n
            push!(xw,x[n .+ (((i-1)*n+1):i*n)])
        end

        for (k,_x) in enumerate(xw)
            c.c(view(v,((k-1)*p+1:k*p)),_x)
        end
    end

    function ∇rc(V,x)
        xw = Vector[]

        push!(xw,x[1:n])

        for i = 1:2*n
            push!(xw,x[n .+ (((i-1)*n+1):i*n)])
        end

        for (k,_x) in enumerate(xw)
            _V = zeros(p,n)
            idx_p = (((k-1)*p+1):k*p)
            c.∇c(_V,_x)
            idx = (k-1)*n+1:k*n
            copyto!(view(V,idx_p,idx),_V)
        end
    end

    typeof(c)(rc,∇rc,p_robust,c.label,c.inds)
end

"Update jacobian for robust dynamics that include δx, δu terms in state vector"
function update_jacobian(_∇F::Function,n::Int,m::Int,r::Int)
    inds = (x=1:n, u=n .+ (1:m),w = (n+m) .+ (1:r), dt=n+m+r+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)),xw=(1:n,(n+m) .+ (1:r)), xdt=(1:n,(n+m+r) .+ (1:1)))
    nδ = 2*(n^2 + m^2)
    S0 = zeros(n,n+nδ+m+r+1)
    ẋ0 = zeros(n)

    idx_x = 1:n
    idx_u = 1:m
    idx_w = 1:r
    idx = [(idx_x)...,((n+nδ) .+ (idx_u))...,((n+nδ+m) .+ (idx_w))...,(n+nδ+m+r+1)]

    ∇F(S::AbstractMatrix,ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T= begin
        _∇F(view(S,idx_x,idx),ẋ,x,u,w,dt)
        return nothing
    end
    ∇F(S::AbstractMatrix,x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T = begin
        _∇F(view(S,idx_x,idx),ẋ0,x,u,w,dt)
        return nothing
    end
    ∇F(x::AbstractVector,u::AbstractVector,w::AbstractVector,dt::T) where T = begin
        _∇F(view(S0,idx_x,idx),ẋ0,x,u,w,dt)
        return S0
    end
    return ∇F
end
