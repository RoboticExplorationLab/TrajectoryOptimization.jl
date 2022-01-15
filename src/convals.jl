@inline get_data(A::AbstractArray) = A
@inline get_data(A::SizedArray) = A.data

"""
	AbstractConstraintValues{C<:AbstractConstraint}

An abstract type for working with and storing constraint values, such as current 
constraint values, Jacobians, dual variables, penalty parameters, etc. 
The information that is actually store, and the way it is stored, is up to the 
child type. However, at a minimum, it should store the following fields:
* `con::AbstractConstraint`: the actual constraint
* `vals::AbstractVector{<:AbstractVector}`: stores the constraint value for all time indices.
* `jac::AbstractMatrix{<:AbstractMatrix}`: stores the constraint Jacobian for all time indices.
* `inds::AbstractVector{Int}`: stores the time step indices.
The first dimension of all of these data fields should be the same 
(the number time indices).

With these fields, the following methods are implemented:
* `evaluate!(::AbstractConstraintValues, ::AbstractTrajectory)`
* `jacobian!(::AbstractConstraintValues, ::AbstractTrajectory)`
* `max_violation(::AbstractConstraintValues)`
"""
abstract type AbstractConstraintValues{C<:AbstractConstraint} end

"""
	ConVal{C,V,M,T}

Holds information about a constraint of type `C`. Allows for any type of
vector (`V`) or matrix (`M`) storage for constraint values and Jacobians
(allowing StaticArrays or views into a large, sparse matrix).
"""
struct ConVal{C,V,M,W} <: AbstractConstraintValues{C}
    con::C
	sig::FunctionSignature
	diffmethod::DiffMethod
    inds::Vector{Int}
    vals::Vector{V}
	vals2::Vector{V}
    jac::Matrix{M}
    ∇x::Matrix{W}
    ∇u::Matrix{W}
    c_max::Vector{Float64}
	is_const::BitArray{2}
	const_hess::BitVector
	iserr::Bool  # are the Jacobians on the error state
	function ConVal(n::Int, m::Int, con::AbstractConstraint, inds::Union{Vector{Int},UnitRange}, 
			jac, vals, iserr::Bool=false; 
			sig::FunctionSignature=StaticReturn(), diffmethod::DiffMethod=UserDefined()
	)
		if !iserr && (RD.output_dim(con), n+m) != size(jac[1])
			throw(DimensionMismatch("size of jac[i] $(size(jac[1])) does not match the expected size of $(size(gen_jacobian(con)))"))
		end
		vals2 = deepcopy(vals)
        p = RD.output_dim(con)
        P = length(vals)
        ix = 1:n
        iu = n .+ (1:m)
		∇x = [view(∇c, :, ix) for ∇c in jac]
		∇u = [view(∇c, :, iu) for ∇c in jac]
        c_max = zeros(P)
		is_const = BitArray(undef, size(jac)) 
		const_hess = BitVector(undef, P)
        new{typeof(con), eltype(vals), eltype(jac), eltype(∇x)}(con, sig, diffmethod,
			collect(inds), vals, vals2, jac, ∇x, ∇u, c_max, is_const, const_hess, iserr)
    end
end

function ConVal(n::Int, m::Int, cval::ConVal)
	# create a ConVal for the "raw" Jacobians, if needed
	# 	otherwise return the same ConVal
	if cval.iserr
		p = RD.output_dim(cval.con)
		ws = widths(cval.con, n, m)
		jac = [SizedMatrix{p,w}(zeros(p,w)) for k in cval.inds, w in ws]
		ConVal(n, m, cval.con, cval.inds, jac, cval.vals, false; 
		       sig=cval.sig, diffmethod=cval.diffmethod)
	else
		return cval
	end
end

function ConVal(n::Int, m::Int, con::AbstractConstraint, inds::UnitRange{Int}, 
	            iserr::Bool=false; kwargs...)
	C,c = gen_convals(n,m,con,inds)
	ConVal(n, m, con, inds, C, c; kwargs...)
end

function _index(cval::AbstractConstraintValues, k::Int)
	if k ∈ cval.inds
		return k - cval.inds[1] + 1
	else
		return 0
	end
end

@inline RD.output_dim(cval::AbstractConstraintValues) = RD.output_dim(cval.con) 
import Base.length
@deprecate length(cval::AbstractConstraintValues) RD.output_dim(cval)

function RD.evaluate!(cval::AbstractConstraintValues, Z::AbstractTrajectory)
	RD.evaluate!(cval.sig, cval.con, cval.vals, Z, cval.inds)
end

function RD.jacobian!(cval::AbstractConstraintValues, Z::AbstractTrajectory)
	if cval.iserr
		throw(ErrorException("Can't evaluate Jacobians directly on the error state Jacobians"))
	else
		RD.jacobian!(cval.sig, cval.diffmethod, cval.con, cval.jac, cval.vals, Z, cval.inds)
	end
end

function RD.∇jacobian!(G, cval::AbstractConstraintValues, Z::AbstractTrajectory, λ)
	RD.∇jacobian!(cval.sig, cval.diffmethod, cval.con, G, cval.λ, cval.vvals, Z, cval.inds)
end

function error_expansion!(errval::AbstractConstraintValues{C}, conval::AbstractConstraintValues, 
		model::AbstractModel, G) where C
	if errval.jac !== conval.jac
		n,m = size(model)
		ix = 1:n
		iu = 1:m
		for (i,k) in enumerate(conval.inds)
			if C <: StateConstraint
				mul!(errval.jac[i], conval.jac[i], get_data(G[k]))
			elseif C <: ControlConstraint
				errval.jac[i] .= conval.jac[i]
			else
				∇x  = view(errval.jac[i], :, ix)
				∇u  = view(errval.jac[i], :, iu)
				∇x0 = view(conval.jac[i], :, ix)
				∇u0 = view(conval.jac[i], :, iu)
				mul!(∇x, ∇x0, get_data(G[k]))
				∇u .= ∇u0
			end
		end
	end
end

##################################################
# Constraint Violations
##################################################

violation(::Equality, x) = x
violation(cone::Conic, x) = projection(cone, x) - x
∇violation!(::Equality, ∇v, ∇c, x, tmp=zeros(length(x), length(x))) = ∇v .= ∇c
function ∇violation!(cone::Conic, ∇v, ∇c, x, tmp=zeros(length(x), length(x))) 
	∇projection!(cone, tmp, x)
	for i = 1:length(x)
		tmp[i,i] -= 1
	end
	mul!(∇v, tmp, ∇c)
end

@inline max_violation(::Equality, v) = norm(v,Inf)
@inline max_violation(::Inequality, v) = max(0,maximum(v))
@inline max_violation(::Equality, v::Real) = abs(v)
@inline max_violation(::Inequality, v::Real) = v > 0 ? v : 0.0

function max_violation(cone::SecondOrderCone, x)
	proj = projection(cone, x)
	return norm(x - proj,Inf)
end

function max_violation(cval::AbstractConstraintValues)
	max_violation!(cval)
    return maximum(cval.c_max)
end

function max_violation!(cval::AbstractConstraintValues)
	s = sense(cval.con)
	P = output_dim(cval)
    for i in eachindex(cval.inds)
        cval.c_max[i] = max_violation(s, SVector{P}(cval.vals[i]))
    end
end

@inline norm_violation(::Equality, v, p=2) = norm(v,p)

@inline function norm_violation(::Inequality, v, p=2)
	# TODO: try this with LazyArrays?
	if p == 1
		a = zero(eltype(v))
		for x in v
			a += max(x,0)
		end
		return a
	elseif p == 2
		a = zero(eltype(v))
		for x in v
			a += max(x, 0)^2
		end
		return sqrt(a)
	elseif p == Inf
		return maximum(v)
	else
		throw(ArgumentError("$p is not a valid norm value. Must be 1,2 or Inf"))
	end
end

function norm_violation(cval::AbstractConstraintValues, p=2)
	norm_violation!(cval, p)
	return norm(cval.c_max, p)
end

function norm_violation!(cval::AbstractConstraintValues, p=2)
	s = sense(cval.con)
	for i in eachindex(cval.inds)
		cval.c_max[i] = norm_violation(s, cval.vals[i], p)
	end
end

function norm_dgrad!(cval::AbstractConstraintValues, Z::AbstractTrajectory, p=1)
	for (i,k) in enumerate(cval.inds)
		zs = RobotDynamics.get_z(cval.con, Z, k)
		mul!(cval.vals2[i], cval.jac[i,1], zs[1])
		if length(zs) > 1
			mul!(cval.vals2[i], cval.jac[i,2], zs[2], 1.0, 1.0)
		end
		cval.c_max[i] = norm_dgrad(cval.vals[i], cval.vals2[i], p)
	end
	return nothing
end
"""
	dgrad(x, dx, p=1)
Directional derivative of `norm(x, p)` in the direction `dx`
"""
function norm_dgrad(x, dx, p=1)
	g = zero(eltype(x))
	if p == 1
		@assert length(x) == length(dx)
		g = zero(eltype(x))
		for i in eachindex(x)
			if x[i] < 0
				g += -dx[i]
			elseif x[i] > 0
				g += dx[i]
			else
				g += abs(dx[i])
			end
		end
	else
		throw("Directional derivative of $p-norm isn't implemented yet")
	end
	return g
end

function norm_residual!(res, cval::AbstractConstraintValues, λ::Vector{<:AbstractVector}, p=2)
	for (i,k) in enumerate(cval.inds)
		mul!(res[i], cval.jac[i,1], λ[i])
		if size(cval.jac,2) > 1
			mul!(res[i], cval.jac[i,2], λ[i], 1.0, 1.0)
		end
		cval.c_max[i] = norm(res[i], p)
	end
	return nothing
end

function gen_convals(n̄::Int, m::Int, con::AbstractConstraint, inds)
    # n is the state diff size
    p = RD.output_dim(con)
	if con isa StageConstraint
		njac = 1
		len = length(inds)
	elseif con isa CoupledConstraint
		njac = 2
		len = length(inds) + 1
	else
		error("gen_convals only defined for Stage and Coupled Constraints.")
	end
    C = [SizedMatrix{p,n̄+m}(zeros(p,n̄+m)) for k in 1:len, w in 1:njac]
	c = [@MVector zeros(p) for k in 1:len]
    return C, c
end

function gen_convals(D::AbstractMatrix, d::AbstractVector, cinds, zinds, con::AbstractConstraint, inds)
    P = length(inds)
    p = RD.output_dim(con)
	n,m = get_dims(con, length(zinds[1]))
    ws = widths(con, n, m)

    C = [begin
		view(D, cinds[i], zinds[k+(j-1)][1:ws[j]])
	end for (i,k) in enumerate(inds), j = 1:length(ws)]
    c = [view(d, cinds[i]) for i in 1:P]
    return C,c
end

function gen_convals(blocks::Vector, cinds, con::AbstractConstraint, inds)
	# assumes all cinds are contiguous indices (i.e. can be represented as a UnitRange)
    C1 = map(enumerate(inds)) do (i,k)
        nm = size(blocks[k].Y,2)
		if con isa StateConstraint
			iz = 1:width(con)
		elseif con isa ControlConstraint
			m = control_dim(con)
			n = nm - m
			iz = n .+ (1:m)
		else
			iz = 1:nm
		end
		ic = cinds[i][1]:cinds[i][end]
		n1 = size(blocks[k].D2, 1)
        view(blocks[k].Y, n1 .+ (ic), iz)
    end
	C2 = map(enumerate(inds)) do (i,k)
		if con isa StageConstraint
			w = size(blocks[k].Y,2)
			view(blocks[k].Y,1:0,1:w)
		else
			w = size(blocks[k+1].Y,2)
			n = state_dim(con)
			view(blocks[k+1].Y,1:n,1:w)
		end
	end
	C = [C1 C2]
    c = map(enumerate(inds)) do (i,k)
		ic = cinds[i][1]:cinds[i][end]
        view(blocks[k].y, ic)
    end
    return C,c
end
