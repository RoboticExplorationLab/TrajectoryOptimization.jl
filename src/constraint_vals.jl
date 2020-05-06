# export
# 	ConstraintVals

mutable struct ConstraintParams{T}
	ϕ::T  	    # penalty scaling parameter
	μ0::T 	    # initial penalty parameter
	μ_max::T    # max penalty parameter
	λ_max::T    # max Lagrange multiplier
end
function ConstraintParams(ϕ::T1 = 10, μ0::T2 = 1.0, μ_max::T3 = 1e8, λ_max::T4 = 1e8) where {T1,T2,T3,T4}
	T = promote_type(T1,T2,T3,T4)
	ConstraintParams(T(ϕ), T(μ0), T(μ_max), T(λ_max))
end

""" $(TYPEDEF)
Struct that stores all of the values associated with a particular constraint.
Importantly, `ConstraintVals` stores the list of knotpoints to which the constraint
is applied. This type should be fairly transparent to the user, and only needs to be
directly dealt with when writing solvers or setting fine-tuned updates per constraint
(via the `.params` field).
"""
struct ConstraintVals{T,W,C,P,MT}
	con::C
	inds::UnitRange{Int}
	vals::Vector{SVector{P,T}}
	vals_prev::Vector{SVector{P,T}}
	∇c::Matrix{MT}
	λ::Vector{SVector{P,T}}
	μ::Vector{SVector{P,T}}
	active::Vector{SVector{P,Bool}}
	c_max::Vector{T}
	params::ConstraintParams{T}

	function ConstraintVals(con::AbstractConstraint{S, W},
			inds::UnitRange{Int}, vals::V, vals_prev,
			∇c::Matrix{MT}, λ::V, μ::V,
			active::Vector{SVector{P,Bool}}, c_max::Vector{T},
			params::ConstraintParams) where {S,W,T,P,N,V,MT}
		new{T,W,typeof(con),P,MT}(con,inds,vals,vals_prev,∇c,λ,μ,
			active,c_max, params)
	end
end

function ConstraintVals(con::C, inds::UnitRange, MT::Type=SizedMatrix, kwargs...) where C
	p = length(con)
	w = widths(con)
	P = length(inds)
	nw = length(w)
	λ    = [@SVector zeros(p) for k = 1:P]
	μ    = [@SVector ones(p)  for k = 1:P]
	atv  = [@SVector ones(Bool,p) for k = 1:P]
	vals = [@SVector zeros(p) for k = 1:P]
	if MT <: SizedArray
		∇c = [SizedMatrix{p,w[i]}(zeros(Float64,p,w[i])) for k = 1:P, i = 1:nw]
	elseif MT <: SubArray
		∇c = [view(zeros(Float64,p,w[i]),1:p,1:w[i]) for k = 1:P, i = 1:nw]
	end
	params = ConstraintParams(;kwargs...)
	ConstraintVals(con, inds, vals, deepcopy(vals), ∇c, λ, μ, atv, zeros(P),
		params)
end

function ConstraintVals(conVal::ConstraintVals, ∇c::Matrix{<:AbstractMatrix})
	ConstraintVals(conVal.con, conVal.inds, conVal.vals, conVal.vals_prev,
		∇c, conVal.λ, conVal.μ, conVal.active, conVal.c_max, conVal.params)
end

get_params(con::ConstraintVals)::ConstraintParams = con.params

function _index(con::ConstraintVals, k::Int)
	if k ∈ con.inds
		return k - con.inds[1] + 1
	else
		return 0
	end
end

Base.length(::ConstraintVals{T,W,C,P}) where {T,W,C,P} = P
Base.length(con::ConstraintVals, k::Int) = k ∈ con.inds ? length(con) : 0
state_dim(con::ConstraintVals) = state_dim(con.con)
control_dim(con::ConstraintVals) = control_dim(con.con)
check_dims(con::ConstraintVals,n,m) = check_dims(con.con,n,m)
constraint_type(con::ConstraintVals{T,W,C}) where {T,W,C} = C
is_bound(con::ConstraintVals) = is_bound(con.con)
duals(con::ConstraintVals) = con.λ
duals(con::ConstraintVals, k::Int) = con.λ[_index(con,k)]
penalty(con::ConstraintVals) = con.μ
penalty(con::ConstraintVals, k::Int) = con.μ[_index(con,k)]
penalty_matrix(con::ConstraintVals, i::Int) = Diagonal(con.active[i] .* con.μ[i])
lower_bound(con::ConstraintVals) = lower_bound(con.con)
upper_bound(con::ConstraintVals) = upper_bound(con.con)
contype(con::ConstraintVals) = contype(con.con)
sense(con::ConstraintVals) = sense(con.con)

evaluate!(con::ConstraintVals, Z::Traj) = evaluate!(con.vals, con.con, Z, con.inds)
jacobian!(con::ConstraintVals, Z::Traj) = jacobian!(con.∇c, con.con, Z, con.inds)


function update_active_set!(con::ConstraintVals{T,W,C}, ::Val{tol}) where
		{T,W,C<:AbstractConstraint{Inequality},tol}
	for i in eachindex(con.vals)
		con.active[i] = @. (con.vals[i] >= -tol) | (con.λ[i] > 0)
	end
	return nothing
end

update_active_set!(con::ConstraintVals{T,W,C}, ::Val{tol}) where
	{T,W,C<:AbstractConstraint{Equality},tol} = nothing

function viol_ineq(v::T, a)::T where T
	for i in eachindex(a)
		v = max(v, max(a[i], 0.0))
	end
	return v
end

function viol_eq(v::T, a)::T where T
	for i in eachindex(a)
		v = max(v, abs(a[i]))
	end
	return v
end

function violation(::Type{Inequality}, vals::SVector)
	return max.(vals, 0)
end

function violation(::Type{Equality}, vals::SVector)
	return abs.(vals)
end

function max_violation!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractConstraint{Inequality}}
	for i in eachindex(con.c_max)
		con.c_max[i] = viol_ineq(0.0, con.vals[i])
	end
	return nothing
end

function max_violation!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractConstraint{Equality}}
	for i in eachindex(con.c_max)
		# con.c_max[i] = norm(con.vals[i],Inf)
		con.c_max[i] = viol_eq(0.0, con.vals[i])
	end
	return nothing
end


function max_penalty!(con::ConstraintVals)
	for i in eachindex(con.c_max)
		con.c_max[i] = maximum(con.μ[i])
	end
	return nothing
end

get_c_max(con::ConstraintVals) = maximum(con.c_max)


function reset!(con::ConstraintVals{T,W,C,P}) where {T,W,C,P}
	reset_duals!(con)
	reset_penalties!(con)
end

function reset_duals!(con::ConstraintVals)
	λ = con.λ
	for i in eachindex(con.inds)
		λ[i] *= 0.0
	end
end

function reset_penalties!(con::ConstraintVals{T,W,C,P}) where {T,W,C,P}
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = con.params.μ0 * @SVector ones(T,P)
		c[i] *= 0.0
	end
end

function shift_fill!(con::ConstraintVals)
	shift_fill!(con.μ)
	shift_fill!(con.λ)
end

function cost!(J, con::ConstraintVals, Z)
	for (i,k) in enumerate(con.inds)
		c = con.vals[i]
		λ = con.λ[i]
		Iμ = penalty_matrix(con, i)
		J[k] += λ'c + 0.5*c'Iμ*c
	end
end

@generated function cost_expansion!(E::Vector{<:AbstractExpansion}, con::ConstraintVals{T,W},
		Z::Vector{<:KnotPoint{T,N,M}}) where {T,W<:Stage,N,M}
	if W <: State
		expansion = quote
			cx = SMatrix(con.∇c[i])
			E[k].xx .+= cx'Iμ*cx
			E[k].x .+= cx'g
		end
	elseif W <: Control
		expansion = quote
			cu = SMatrix(con.∇c[i])
			E[k].uu .+= cu'Iμ*cu
			E[k].u .+= cu'g
		end
	else
		expansion = quote
			cx = SMatrix(con.∇c[i])[:,ix]
			cu = SMatrix(con.∇c[i])[:,iu]

			E[k].xx .+= cx'Iμ*cx
			E[k].uu .+= cu'Iμ*cu
			E[k].ux .+= cu'Iμ*cx

			E[k].x .+= cx'g
			E[k].u .+= cu'g
		end
	end
	quote
		ix,iu = Z[1]._x, Z[1]._u
		@inbounds for i in eachindex(con.inds)
			k = con.inds[i]
			c = con.vals[i]
			λ = con.λ[i]
			μ = con.μ[i]
			a = con.active[i]
			Iμ = Diagonal( a .* μ )
			g = Iμ*c + λ

			$expansion
		end
	end
end
