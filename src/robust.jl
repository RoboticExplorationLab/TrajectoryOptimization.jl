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
        P[k] .= 0.5*(P[k] + P[k]')
    end

    return K, P
end

function tvlqr_con(prob::Problem{T},Q::AbstractArray{T},R::AbstractArray{T},Qf::AbstractArray{T}) where T
    n = prob.model.n; m = prob.model.m; r = prob.model.r; N = prob.N
    dt = prob.dt

    K  = [zeros(T,m,n) for k = 1:N-1]
    S  = [zeros(T,n^2) for k = 1:N]

    X_interp = gen_cubic_interp(prob.X,prob.dt)
    U_interp = gen_zoh_interp([prob.U...,prob.U[end]],prob.dt)

    function _Ac(t)
        Zc = zeros(n,n+m+r)
        p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t),zeros(r))
        Zc[:,1:n]
    end

    function _Bc(t)
        Zc = zeros(n,n+m+r)
        p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t),zeros(r))
        Zc[:,n .+ (1:m)]
    end

    function _Gc(t)
        Zc = zeros(n,n+m+r)
        p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t),zeros(r))
        Zc[:,(n + m) .+ (1:r)]
    end

    function r_dyn_sqrt(ṡ,s,t)
        SS = reshape(s,n,n)

        ss = inv(SS')
        ṡ[1:n^2] = reshape(-.5*Q*ss - _Ac(t)'*SS + .5*(SS*SS'*_Bc(t))*inv(R)*(_Bc(t)'*SS),n^2)
    end

    S[N] = vec(cholesky(Qf).U)

    _t = [prob.tf]

    for k = N:-1:2
        k1 = k2 = k3 = k4 = zero(S[k])
        x = S[k]
        println(k1)
        println(x)
        r_dyn_sqrt(k1, x, _t[1]);
        k1 *= -dt;
        r_dyn_sqrt(k2, x + k1/2, _t[1] - dt/2);
        k2 *= -dt;
        r_dyn_sqrt(k3, x + k2/2, _t[1] - dt/2);
        k3 *= -dt;
        r_dyn_sqrt(k4, x + k3, max(_t[1] - dt, 0.));
        k4 *= -dt;
        println(k)
        copyto!(S[k-1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
        _t[1] -= dt
    end

    return K, S
end

function gen_cubic_interp(X,dt)
    N = length(X); n = length(X[1])

    function interp(t)
        j = t/dt + 1.
        x = zeros(eltype(t),n)
        for i = 1:n
            x[i] = interpolate([X[k][i] for k = 1:N],BSpline(Cubic(Line(OnGrid()))))(j)
        end
        return x
    end
end

function gen_zoh_interp(X,dt)
    N = length(X); n = length(X[1])

    function interp(t)
        if (t/dt + 1.0)%floor(t/dt + 1.0) < 0.99999
            j = convert(Int64,floor(t/dt)) + 1
        else
            j = convert(Int64,ceil(t/dt)) + 1
        end

        interpolate(X,BSpline(Constant()))(j)

    end
end

"Robust model from model"
function robust_model(model::Model{Uncertain,Discrete},Q::AbstractArray{T},R::AbstractArray{T},D::AbstractArray) where T
    n = model.n; m = model.m; r = model.r

    fc! = prob.model.info[:fc]
    ∇fc! = prob.model.info[:∇fc]
    integration = prob.model.info[:integration]

    discretize_model(robust_model(fc!,∇fc!,Q,R,D,n,m,r),integration,prob.dt)
end

function robust_model(model::Model{Uncertain,Continuous},Q::AbstractArray{T},R::AbstractArray{T},D::AbstractArray) where T
    n = model.n; m = model.m; r = model.r

    robust_model(model.f,model.∇f,Q,R,D,n,m,r)
end

"Robust dynamics, includes: nominal, disturbance, and cost-to-go dynamics"
function robust_model(f::Function,∇f::Function,Q::AbstractArray{T},R::AbstractArray{T},D::AbstractArray{T},n::Int,m::Int,r::Int) where T
    idx = (x=1:n,u=1:m,w=1:r)
    z_idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)))

    Zc = zeros(n,n+m+r)
    n̄ = n + n^2 + n*r + n^2

    function robust_dynamics(ż,z,u,w)
        x = z[z_idx.x]
        E = reshape(z[z_idx.e],n,n)
        H = reshape(z[z_idx.h],n,r)
        S = reshape(z[z_idx.s],n,n)
        ss = inv(S')
        P = S*S'

        ∇f(Zc,x,u,w)
        Ac = Zc[:,idx.x]
        Bc = Zc[:,n .+ idx.u]
        Gc = Zc[:,(n+m) .+ idx.w]

        Kc = R\(Bc'*P)
        Acl = Ac - Bc*Kc

        f(view(ż,z_idx.x),x,u,w)
        ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
        ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
        ż[z_idx.s] = reshape(-.5*Q*ss - Ac'*S + .5*(P*Bc)*(R\(Bc'*S)),n^2)
    end

    UncertainModel(robust_dynamics, n̄, m, r)
end

mutable struct RobustCost{T} <: CostFunction
    Qr::AbstractArray{T}
    Rr::AbstractArray{T}
    Qrf::AbstractArray{T}

    Q::AbstractArray{T}
    R::AbstractArray{T}
    Qf::AbstractArray{T}

    K::Function

    n::Int
    m::Int
    r::Int
    idx::NamedTuple
end
#
# function RobustCost(model::Model{M,D},Qr::AbstractArray{T},Rr::AbstractArray{T},Qfr::AbstractArray{T},
#         Q::AbstractArray{T},R::AbstractArray{T},Qf::AbstractArray{T}) where {M,D,T}
#     n = model.n; m = model.m; r = model.r
#
#     idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)))
#
#     Zc = zeros(n,n+m+r)
#
#     if D <: Continuous
#         ∇fc = model.∇fc
#     else
#         ∇fc = model.info[:∇fc]
#     end
#
#     function K(z)
#         x = z[idx.x]
#         s = z[idx.s]
#         P = reshape(s,n,n)*reshape(s,n,n)'
#
#         ∇fc(Zc,x,u,zeros(r))
#         Bc = Zc[:,n .+ (1:m)]
#
#         R\(Bc*P)
#     end
#
#     RobustCost(Qr,Rr,Qfr,Q,R,Qf,K,n,m,r,idx)
# end


function stage_cost(cost::RobustCost, z::Vector{T}, u::Vector{T}) where T
    idx = cost.idx; n = cost.n; m = cost.m; r = cost.r

    x = z[idx.x]
    E = reshape(z[idx.e],n,n)

    Kc = K(z)

    tr((cost.Qr + K'*cost.Rr*K)*E)
end

function stage_cost(cost::RobustCost, zN::Vector{T}) where T
    idx = cost.idx; n = cost.n
    E = reshape(z[idx.e],n,n)
    tr(cost.Qrf*E)
end

function cost(c::RobustCost,Z::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(Z)
    ℓ = 0.
    for k = 1:N-1
        ℓ += stage_cost(c,Z[k],U[k])
    end
    ℓ += stage_cost(c,Z[N])
    ℓ
end

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
    H1::AbstractArray{T},D::AbstractArray{T},Qr::AbstractArray{T},Rr::AbstractArray{T},Qfr::AbstractArray{T},Q::AbstractArray{T},R::AbstractArray{T},
    Qf::AbstractArray{T}) where T
    N = prob.N
    @assert all([prob.obj[k] isa QuadraticCost for k = 1:N]) #TODO generic cost

    n = prob.model.n; m = prob.model.m; r = prob.model.r
    n_robust = 2*n^2 + n*r # number of robust parameters in state vector

    idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)))

    # generate optimal feedback matrix function
    Zc = zeros(n,n+m+r)

    ∇fc = prob.model.info[:∇fc]


    function K(z,u)
        x = z[idx.x]
        s = z[idx.s]
        P = reshape(s,n,n)*reshape(s,n,n)'

        ∇fc(Zc,x,u,zeros(r))
        Bc = Zc[:,n .+ (1:m)]

        R\(Bc*P)
    end

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

    model_uncertain = robust_model(prob.model,Q,R,D)

    robust_cost = RobustCost(Qr,Rr,Qfr,Q,R,Qf,K,n,m,r,idx)

    con_prob = AbstractConstraintSet[]
    constrained = is_constrained(prob)


    s1(c,z,u) = u[m .+ (1:n^2)] - z[idx.s]

    C = zeros(n^2,n + n^2 + n*r + n^2 + m + n^2)
    function ∇s1(C,z,u)
        C[:,(n+n^2+n*r) .+ (1:n^2)] = -1.0*Diagonal(ones(n^2))
        C[:,(n+n^2+n*r+n^2+m) .+ (1:n^2)] = 1.0*Diagonal(ones(n^2))
    end
    ctg_init = Constraint{Equality}(s1,∇s1,n + n_robust,m + n^2,n^2,:ctg)

    for k = 1:N-1
        con_uncertain = AbstractConstraint[]
        if k == 1
            push!(con_uncertain,ctg_init)
        end
        if constrained
            for cc in prob.constraints[k]
                push!(con_uncertain,robust_constraint(cc,K,idx,prob.model.n,prob.model.m))
            end
        end
        push!(con_prob,con_uncertain)
    end

    if constrained
        con_uncertain = AbstractConstraint[]
        for cc in prob.constraints[N]
            push!(con_uncertain,robust_constraint(cc,prob.model.n))
        end
    else
        con_uncertain = Constraint[]
    end

    push!(con_prob,con_uncertain)
    rollout_nominal!(prob)
    # K, P = tvlqr_dis(prob,Q,R,Qf)
    K, S = tvlqr_con(prob,Q,R,Qf)

    update_problem(prob,model=model_uncertain,obj=RobustObjective(_obj,robust_cost),
        constraints=ProblemConstraints(con_prob),
        X=[[prob.X[k];ones(n_robust)*NaN32] for k = 1:prob.N],
        U=[k == 1 ? [prob.U[k];S[1]] : prob.U[k] for k = 1:prob.N-1],
        x0=[copy(prob.x0);reshape(E1,n^2);reshape(H1,n*r);S[1]])

end

function E_to_δx(E,i)
    sqrt(E)[:,i]
end

function ∇E_to_δx(E,i)
    n = size(E,1)

    function f1(e)
        ee = eigen(E)

        b = Diagonal(sqrt.(ee.values))*ee.vectors'
        b[:,i]
    end

    ForwardDiff.jacobian(f1,vec(E))[:,((i-1)*n + 1):i*n]
end

function E_to_δu(K,z,u,idx,j)
    _K = K(z,u)
    n = size(_K,2)
    E = reshape(z[idx.e],n,n)
    sqrt(_K*E*_K')[:,j]
end

function ∇E_to_δu(K,z,u,idx,j)
    m = length(u)

    function f1(z)
        _K = K(z,u)
        n = size(_K,2)
        E = reshape(z[idx.e],n,n)
        ee = eigen(_K*E*_K')

        b = Diagonal(sqrt.(ee.values))*ee.vectors'
        b[:,j]
    end

    ForwardDiff.jacobian(f1,z)[:,((j-1)*m + 1):j*m]
end

"Modify constraint to evaluate all combinations of the disturbed state and control"
function robust_constraint(c::AbstractConstraint,K::Function,idx::NamedTuple,n::Int,m::Int)
    p = c.p
    p_robust = p*(2*n+1)*(2*m+1)

    function rc(v,z,u)
        xw = Vector[]
        uw = Vector[]

        x = z[idx.x]
        E = reshape(z[idx.e],n,n)
        Ex = sqrt(E)
        _K = K(z,u)
        Eu = sqrt(_K*E*_K')

        push!(xw,x)
        push!(uw,u)

        for i = 1:n
            δx = Ex[:,i]
            push!(xw,x + δx)
            push!(xw,x - dx)
        end
        for j = 1:m
            δu = Eu[:,j]
            push!(uw,u + du)
            push!(xw,u - du)
        end

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
        sx = []
        su = []

        x = z[idx.x]
        E = reshape(z[idx.e],n,n)
        Ex = sqrt(E)
        _K = K(z,u)
        Eu = sqrt(_K*E*_K')

        push!(xw,x)
        push!(sx,(0,0))

        push!(uw,u)
        push!(su,(0,0))

        for i = 1:n
            δx = Ex[:,i]
            push!(xw,x + δx)
            push!(sx,(i,1))
            push!(xw,x - dx)
            push!(sx,(i,-1))
        end
        for j = 1:m
            δu = Eu[:,j]
            push!(uw,u + δu)
            push!(su,(j,1))
            push!(uw,u - δu)
            push!(su,(j,-1))
        end

        k = 1
        for (i,_x) in enumerate(xw)
            for (j,_u) in enumerate(uw)
                _V = zeros(p,n+m)
                idx_p = (((k-1)*p+1):k*p)
                c.∇c(_V,_x,_u)

                idx = [(1:n)...,((n + n^2 + n*r + n^2) .+ (1:m))...]
                copyto!(view(V,idx_p,idx),_V)

                cx = _V[:,1:n]
                cu = _V[:,n .+ (1:m)]

                if i != 1
                    idx = (n .+ (sx[i][1]-1)*n + 1): sx[i][1]*n
                    copyto!(view(V,idx_p,idx),sx[i][2]*cx*∇E_to_δx(E,sx[i][1]))
                end
                if j != 1
                    idx = ((n + n^2 + n*r + n^2) .+ (su[i][1]-1)*m + 1): su[i][1]*m
                    copyto!(view(V,idx_p,idx),su[i][2]*cu*∇E_to_δu(K,z,u,idx,su[i][1]))
                end

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

        x = z[idx.x]
        E = reshape(z[idx.e],n,n)
        Ex = sqrt(E)

        push!(xw,x)

        for i = 1:n
            δx = Ex[:,i]
            push!(xw,x + δx)
            push!(xw,x - dx)
        end

        for (k,_x) in enumerate(xw)
            c.c(view(v,((k-1)*p+1:k*p)),_x)
        end
    end

    function ∇rc(V,x)
        xw = Vector[]
        sx = []

        x = z[idx.x]
        E = reshape(z[idx.e],n,n)
        Ex = sqrt(E)

        push!(xw,x)
        push!(sx,(0,0))

        for i = 1:n
            δx = Ex[:,i]
            push!(xw,x + δx)
            push!(sx,(i,1))
            push!(xw,x - dx)
            push!(sx,(i,-1))
        end

        for (k,_x) in enumerate(xw)
            _V = zeros(p,n)
            idx_p = (((k-1)*p+1):k*p)
            c.∇c(_V,_x)
            idx = (k-1)*n+1:k*n
            copyto!(view(V,idx_p,idx),_V)

            cx = _V[:,1:n]

            if k != 1
                idx = (n .+ (sx[k][1]-1)*n + 1): sx[k][1]*n
                copyto!(view(V,idx_p,idx),sx[k][2]*cx*∇E_to_δx(E,sx[k][1]))
            end
        end
    end

    typeof(c)(rc,∇rc,p_robust,c.label,c.inds)
end

"Update jacobian for robust dynamics that include δx, δu terms in state vector"
function update_jacobian(_∇F::Function,n::Int,m::Int,r::Int)
    inds = (x=1:n, u=n .+ (1:m),w = (n+m) .+ (1:r), dt=n+m+r+1, xx=(1:n,1:n),xu=(1:n,n .+ (1:m)),xw=(1:n,(n+m) .+ (1:r)), xdt=(1:n,(n+m+r) .+ (1:1)))
    # nδ = 2*(n^2 + m^2)
    n_robust = n^2 + n*r + n^2
    S0 = zeros(n,n+n_robust+m+r+1)
    ẋ0 = zeros(n)

    idx_x = 1:n
    idx_u = 1:m
    idx_w = 1:r
    idx = [(idx_x)...,((n+n_robust) .+ (idx_u))...,((n+n_robust+m) .+ (idx_w))...,(n+n_robust+m+r+1)]

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
