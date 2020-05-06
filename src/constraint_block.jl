
""" ```julia
ConstraintBlock{T,D}
```
A block of constraints for a single time-step. Stacked as follows
    [D1; # ∇f(z⁻,z) wrt z
     C;  # ∇c(z) for all stage-wise constraints
     D2] # ∇f(z,z⁺) wrt z
"""
struct ConstraintBlock{T,VT,VV,MT,MV,D}
    cons::Vector{<:AbstractConstraint}
    dynamics::D
    y::VT
    Y::MT
    YYt::Matrix{T}  # outer product => Shur compliment
    JYt::Matrix{T}   # partial Shur compliment
    YJ::Transpose{T,Matrix{T}}
    r::Vector{T}    # Shur compliment residual
    r_::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}

    D2::SubArray{T,2,MV,Tuple{UnitRange{Int},UnitRange{Int}},false}
    c::SubArray{T,1,VV,Tuple{UnitRange{Int}},true}
    C::SubArray{T,2,MV,Tuple{UnitRange{Int},UnitRange{Int}},false}
    c_::Vector{SubArray{T,1,VV,Tuple{UnitRange{Int}},true}}
    C_::Vector{SubArray{T,2,MV,Tuple{UnitRange{Int},UnitRange{Int}},false}}
    d::SubArray{T,1,VV,Tuple{UnitRange{Int}},true}
    D1::SubArray{T,2,MV,Tuple{UnitRange{Int},UnitRange{Int}},false}
    type::Symbol
end

function ConstraintBlock(dyn::DynamicsConstraint{<:Any,<:Any,T},
        cons, type=:stage; Y=zeros(T,0,0), y=zeros(T,0)) where T
    n,m = size(dyn.model)
    if type==:terminal
        n2,m = 0,0
    else
        n2,m2 = n,m
    end
    if type==:initial
        n1 = 0
    else
        n1 = n
    end

    if isempty(cons)
        cons = AbstractConstraint[]
        p = zeros(Int,0)
    else
        p = length.(cons)
    end
    bn,bm = n1 + sum(p) + n2, n+m

    if isempty(Y)
        Y = zeros(T, bn,bm)
    end
    if isempty(y)
        y = zeros(T, sum(p) + n2)
    end

    # Shur compliment pieces
    YYt = zeros(T, bn,bn)
    JYt = zeros(T, bm, bn)
    YJ = transpose(JYt)
    r = zeros(T, bn)
    r_ = [view(r,1:n1), view(r,n1 .+ (1:sum(p))), view(r, (n1 + sum(p)) .+ (1:n2))]

    # e_ = [view(y, e_inds[i] .+ (1:p_prev[i+1])) for i = 1:length(prev)]
    D2 = view(Y, 1:n1, 1:n1+m)
    if isempty(cons)
        c_ = SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}[]
        C_ = SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}[]
    else
        inds = pushfirst!(cumsum(p),0)
        c_ = [view(y, inds[i] .+ (1:p[i])) for i = 1:length(cons)]
        C_ = [view(Y, (inds[i] + n1) .+ (1:p[i]), zinds(cons[i],n,m)) for i = 1:length(cons)]
    end
    c = view(y, 1:sum(p))
    C = view(Y, n1 .+ (1:sum(p)), 1:n+m)
    d = view(y, sum(p) .+ (1:n2))
    D1 = view(Y, (sum(p) + n1) .+ (1:n2), 1:n+m)
    ConstraintBlock(cons,dyn,y,Y,YYt,JYt,YJ,r,r_, D2, c,C, c_,C_, d,D1, type)
end

Base.size(block::ConstraintBlock) = size(block.Y)
dims(block::ConstraintBlock) = size(block.D2,1), length(block.y) - size(block.D1,1), size(block.D1,1)

zinds(con::AbstractConstraint{<:Any,State},n,m) = 1:n
zinds(con::AbstractConstraint{<:Any,Control},n,m) = n .+ (1:m)
zinds(con::AbstractConstraint{<:Any,Stage},n,m) = 1:n+m

function evaluate!(block::ConstraintBlock, z1::KnotPoint, z2::KnotPoint)
    evaluate!(block, z1)
    block.d .= evaluate(block.dynamics, z1, z2)
end

function evaluate!(block::ConstraintBlock, z::KnotPoint)
    for i in eachindex(block.c_)
        evaluate!(block.c_[i], block.cons[i], z)
    end
end

evaluate!(val, con::AbstractConstraint, z::KnotPoint) = val .= evaluate(con, z)
evaluate!(val, con::AbstractConstraint{<:Any,<:Coupled}, z1::KnotPoint, z2::KnotPoint) =
    val .= evaluate(con, z1, z2)

function jacobian!(block::ConstraintBlock, z::KnotPoint)
    if block.type != :initial
        jacobian!(block.D2, block.dynamics, z, 2)
    end
    for i in eachindex(block.C_)
        jacobian!(block.C_[i], block.cons[i], z)
    end
    if block.type != :terminal
        jacobian!(block.D1, block.dynamics, z, 1)
    end
    return nothing
end


const ConstraintBlocks{T,VT,VV,MT,MV,D} = Vector{ConstraintBlock{T,VT,VV,MT,MV,D}}

function ConstraintBlocks(conSet::ConstraintSet)
    cons = copy(conSet.constraints)
    is_dyn = [con.con isa DynamicsConstraint for con in cons]
    dynamics = cons[is_dyn]
    if length(dynamics) == 0
        throw(ArgumentError("Constraint set must contain a dynamics constraint"))
    elseif length(dynamics) > 1
        throw(ArgumentError("Constraint set must contain only 1 dynamics constraint"))
    end
    cons = cons[.!is_dyn]
    dynamics = dynamics[1].con
    N = length(conSet.p)

    map(1:N) do k
        cons_ = map(con->con.con, filter(cons) do con
            k ∈ con.inds
        end)
        if k == 1
            type = :initial
        elseif k == N
            type = :terminal
        else
            type = :stage
        end
        ConstraintBlock(dynamics, cons_, type)
    end
end

function evaluate!(blocks::ConstraintBlocks, Z::Traj)
    N = length(Z)
    for k = 1:N-1
        evaluate!(blocks[k], Z[k], Z[k+1])
    end
    evaluate!(blocks[N], Z[N])
end

function jacobian!(blocks::ConstraintBlocks, Z::Traj)
    for k in eachindex(Z)
        jacobian!(blocks[k], Z[k])
    end
end

function ConstraintBlocks(D, d, blocks::ConstraintBlocks)
    N = length(blocks)
    off1, off2 = 0,0
    map(1:N) do k
        n1,p,n2 = dims(blocks[k])
        bn, bm = size(blocks[k])
        ip = off1 .+ (1:bn)
        iz = off2 .+ (1:bm)
        off1 += n1+p
        off2 += bm
        Y = view(D, ip, iz)
        y = view(d, ip[n1+1:end])
        ConstraintBlock(blocks[k].dynamics, blocks[k].cons, blocks[k].type, Y=Y, y=y)
    end
end
