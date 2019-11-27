
struct ConstraintVals{T,W,C,P,NM,PNM}
	con::C
	inds::UnitRange{Int}
	vals::Vector{SVector{P,T}}
	vals_prev::Vector{SVector{P,T}}
	∇c::Vector{SMatrix{P,NM,T,PNM}}
	λ::Vector{SVector{P,T}}
	μ::Vector{SVector{P,T}}
	active::Vector{SVector{P,Bool}}
	c_max::Vector{T}
	function ConstraintVals(con::AbstractStaticConstraint{S, W},
			inds::UnitRange{Int}, vals::V, vals_prev,
			∇c::Vector{SMatrix{P,NM,T,PNM}}, λ::V, μ::V, active::Vector{SVector{P,Bool}},
			c_max::Vector{T}) where {S,W,T,P,NM,PNM, V}
		new{T,W,typeof(con),P,NM,PNM}(con,inds,vals,vals_prev,∇c,λ,μ,active,c_max)
	end
end

function ConstraintVals(con::C, inds::UnitRange) where C
	n,m,p = size(con)
	w = width(con)
	P = length(inds)
	λ    = [@SVector zeros(p) for k = 1:P]
	μ    = [@SVector ones(p)  for k = 1:P]
	atv  = [@SVector ones(Bool,p) for k = 1:P]
	vals = [@SVector zeros(p) for k = 1:P]
	∇c   = [@SMatrix zeros(p,w) for k = 1:P]
	ConstraintVals(con, inds, vals, deepcopy(vals), ∇c, λ, μ, atv, zeros(P))
end

function _index(con::ConstraintVals, k::Int)
	if k ∈ con.inds
		return k - con.inds[1] + 1
	else
		return 0
	end
end


Base.length(::ConstraintVals{T,W,C,P}) where {T,W,C,P} = P
Base.length(con::ConstraintVals, k::Int) = k ∈ con.inds ? length(con) : 0
Base.size(con::ConstraintVals) = size(con.con)
constraint_type(con::ConstraintVals{T,W,C}) where {T,W,C} = C
is_bound(con::ConstraintVals) = is_bound(con.con)
duals(con::ConstraintVals) = con.λ
duals(con::ConstraintVals, k::Int) = con.λ[_index(con,k)]
penalty(con::ConstraintVals) = con.μ
penalty(con::ConstraintVals, k::Int) = con.μ[_index(con,k)]
penalty_matrix(con::ConstraintVals, i::Int) = Diagonal(con.active[i] .* con.μ[i])
lower_bound(con::ConstraintVals) = lower_bound(con.con)
upper_bound(con::ConstraintVals) = upper_bound(con.con)

evaluate!(con::ConstraintVals, Z::Traj) = evaluate!(con.vals, con.con, Z)
jacobian!(con::ConstraintVals, Z::Traj) = jacobian!(con.∇c, con.con, Z)


function update_active_set!(con::ConstraintVals{T,W,C}, tol=0.0) where
		{T,W,C<:AbstractConstraint{Inequality}}
	for i in eachindex(con.vals)
		con.active[i] = @. (con.vals[i] >= tol) | (con.λ[i] > 0)
	end
	return nothing
end

update_active_set!(con::ConstraintVals{T,W,C}, tol=0.0) where
	{T,W,C<:AbstractConstraint{Equality}} = nothing

function update_active_set!(con::ConstraintVals{T,W,C}, ::Val{tol}) where
		{T,W,C<:AbstractStaticConstraint{Inequality},tol}
	for i in eachindex(con.vals)
		con.active[i] = @. (con.vals[i] >= tol) | (con.λ[i] > 0)
	end
	return nothing
end

update_active_set!(con::ConstraintVals{T,W,C}, ::Val{tol}) where
	{T,W,C<:AbstractStaticConstraint{Equality},tol} = nothing

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

function max_violation!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractStaticConstraint{Inequality}}
	for i in eachindex(con.c_max)
		con.c_max[i] = viol_ineq(0.0, con.vals[i])
	end
	return nothing
end

function max_violation!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractStaticConstraint{Equality}}
	for i in eachindex(con.c_max)
		# con.c_max[i] = norm(con.vals[i],Inf)
		con.c_max[i] = viol_eq(0.0, con.vals[i])
	end
	return nothing
end

get_c_max(con::ConstraintVals) = maximum(con.c_max)

function cost!(J, con::ConstraintVals, Z)
	for (i,k) in enumerate(con.inds)
		c = con.vals[i]
		λ = con.λ[i]
		Iμ = penalty_matrix(con, i)
		J[k] += λ'c + 0.5*c'Iμ*c
	end
end


"""
Assumes constraints, active set, and constrint jacobian have all been calculated
"""
function cost_expansion(E, con::ConstraintVals{T,Stage}, Z) where T
	ix,iu = Z[1]._x, Z[1]._u
	@inbounds for i in eachindex(con.inds)
		k = con.inds[i]
		c = con.vals[i]
		λ = con.λ[i]
		μ = con.μ[i]
		a = con.active[i]
		Iμ = Diagonal( a .* μ )
		cx = con.∇c[i][:,ix]
		cu = con.∇c[i][:,iu]

		E.xx[k] += cx'Iμ*cx
		E.uu[k] += cu'Iμ*cu
		E.ux[k] += cu'Iμ*cx

		g = Iμ*c + λ
		E.x[k] += cx'g
		E.u[k] += cu'g
	end
end

function dual_update!(con::ConstraintVals{T,W,C},
		opts::AugmentedLagrangianSolverOptions{T}) where
		{T,W,C<:AbstractStaticConstraint{Equality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], -opts.dual_max, opts.dual_max)
	end
end

function dual_update!(con::ConstraintVals{T,W,C},
		opts::AugmentedLagrangianSolverOptions{T}) where
		{T,W,C<:AbstractStaticConstraint{Inequality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], 0.0, opts.dual_max)
	end
end

function penalty_update!(con::ConstraintVals{T}, opts::AugmentedLagrangianSolverOptions{T}) where T
	ϕ = opts.penalty_scaling
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = clamp.(ϕ * μ[i], 0.0, opts.penalty_max)
	end
end

function reset!(con::ConstraintVals{T,W,C,P}, opts::AugmentedLagrangianSolverOptions{T}) where {T,W,C,P}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = opts.penalty_initial * @SVector ones(T,P)
		c[i] *= 0.0
		λ[i] *= 0.0
	end
end


############################################################################################
#  								CONSTRAINT SETS 										   #
############################################################################################

struct ConstraintSets{T}
	constraints::Vector{<:ConstraintVals}
	p::Vector{Int}
	c_max::Vector{T}
end

function ConstraintSets(N)
	constraints = Vector{ConstraintVals}()
	p = zeros(Int,N)
	c_max = zeros(N)
	ConstraintSets(constraints,p,c_max)
end


Base.length(conSet::ConstraintSets, k) = constraints.p[k]

function ConstraintSets(constraints, N)
	p = zeros(Int,N)
	c_max = zeros(length(constraints))
	for con in constraints
		for k = 1:N
			p[k] += length(con, k)
		end
	end
	ConstraintSets(constraints, p, c_max)
end

function num_constraints!(conSet::ConstraintSets)
	p = conSet.p
	p .*= 0
	for con in conSet.constraints
		for k = 1:length(p)
			p[k] += length(con, k)
		end
	end
end

function max_violation!(conSet::ConstraintSets{T}) where T
	for i in eachindex(conSet.constraints)
		con = conSet.constraints[i]
		max_violation!(con)
		conSet.c_max[i] = maximum(con.c_max::Vector{T})
	end
end

function evaluate(conSet::ConstraintSets, Z::Traj)
	for con in conSet.constraints
		evaluate(con, Z)
	end
end

function jacobian(conSet::ConstraintSets, Z::Traj)
	for con in conSet.constraints
		jacobian(con, Z)
	end
end

function update_active_set!(conSet::ConstraintSets, Z::Traj, tol=0.0)
	for con in conSet.constraints
		update_active_set!(con, tol)
	end
end

Base.iterate(conSet::ConstraintSets) = @inbounds (conSet.constraints[1], 1)
Base.iterate(conSet::ConstraintSets, i) = @inbounds i >= length(conSet.constraints) ? nothing : (conSet.constraints[i+1], i+1)

function reset!(conSet::ConstraintSets, opts)
	for con in conSet.constraints
		reset!(con, opts)
	end
end
