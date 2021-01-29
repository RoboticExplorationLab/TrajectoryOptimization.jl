#--- NLPData
"""
Holds all the required data structures for evaluating a trajectory optimization problem as
	an NLP. It represents the cost gradient, Hessian, constraints, and constraint Jacobians
	as large, sparse arrays, as applicable.

# Constructors
	NLPData(G, g, zL, zU, D, d, λ)
	NLPData(G, g, zL, zU, D, d, λ, v, r, c)
	NLPData(NN, P, [nD])  # suggested constructor

where `G` and `g` are the cost function gradient and hessian of size `(NN,NN)` and `(NN,)`,
`zL` and `zU` are the lower and upper bounds on the `NN` primal variables,
`D` and `d` are the constraint jacobian and violation of size `(P,NN)` and `(P,)`, and
`v`, `r`, `c` are the values, rows, and columns of the non-zero elements of the costraint
Jacobian, all of length `nD`.
"""
mutable struct NLPData{T}
	G::SparseMatrixCSC{T,Int}
	g::Vector{T}
	zL::Vector{T}  # primal lower bounds
	zU::Vector{T}  # primal upper bounds
	D::SparseMatrixCSC{T,Int}
	d::Vector{T}
	λ::Vector{T}
	v::Vector{T}  # entries of D
	r::Vector{Int}  # rows of D
	c::Vector{Int}  # columns of D
	function NLPData(G::SparseMatrixCSC, g, zL, zU, D::SparseMatrixCSC, d, λ)
		@assert size(G) == (length(g), length(g))
		@assert size(D) == (length(d), length(g))
		@assert length(d) == length(λ)
		new{eltype(G)}(G, g, zL, zU, D, d, λ)
	end
	function NLPData(G::SparseMatrixCSC, g, zL, zU, D::SparseMatrixCSC, d, λ,
			v::AbstractVector, r::Vector{Int}, c::Vector{Int})
		@assert size(G) == (length(g), length(g))
		@assert size(D) == (length(d), length(g))
		@assert length(d) == length(λ)
		@assert length(v) == length(r) == length(c)
		new{eltype(G)}(G, g, zL, zU, D, d, λ, v, r, c)
	end
end

function NLPData(NN::Int, P::Int, nD=nothing)
	G = spzeros(NN,NN)
	g = zeros(NN)
	zL = fill(-Inf,NN)
	zU = fill(+Inf,NN)
	D = spzeros(P,NN)
	d = zeros(P)
	λ = zeros(P)
	if isnothing(nD)
		NLPData(G, g, zL, zU, D, d, λ)
	else
		v = zeros(nD)
		r = zeros(Int,nD)
		c = zeros(Int,nD)
		NLPData(G, g, zL, zU, D, d, λ, v, r, c)
	end
end

#--- NLP Constraint Set
"""
	NLPConstraintSet{T}

Constraint set that updates views to the NLP constraint vector and Jacobian.

The views can be reset to new arrays using `reset_views!(::NLPConstraintSet, ::NLPData)`
"""
struct NLPConstraintSet{T} <: AbstractConstraintSet
    convals::Vector{ConVal}
    errvals::Vector{ConVal}
	jac::JacobianStructure
	λ::Vector{Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}}
	hess::Vector{Matrix{SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}}}
	c_max::Vector{T}
end

function NLPConstraintSet(model::AbstractModel, cons::ConstraintList, data;
		jac_structure=:by_knotpoint, jac_type=:sparse)
	if !has_dynamics_constraint(cons)
		throw(ArgumentError("must contain a dynamics constraint"))
	end
	isequal = integration(cons[end]) <: Implicit

	n,m = size(model)
	n̄ = RobotDynamics.state_diff_size(model)
	ncon = length(cons)
	N = length(cons.p)

	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N-1]
	push!(zinds, (N-1)*(n+m) .+ (1:n))

	# Block sizes
	NN = N*n̄ + (N-1)*m
	P = sum(num_constraints(cons))

	# Initialize arrays
	d = data.d
	if jac_type == :sparse
		D = data.D
	elseif jac_type == :vector
		D = data.v
	end

	# Create ConVals as views into D and d
	jac = JacobianStructure(cons)
	C,c = gen_convals(D, d, cons, jac)
	useG = model isa LieGroupModel
	errvals = map(1:ncon) do i
		ConVal(n̄, m, cons[i], cons.inds[i], C[i], c[i])
	end
	convals = map(errvals) do errval
		ConVal(n, m, errval)
	end
	errvals = convert(Vector{ConVal}, errvals)
	convals = convert(Vector{ConVal}, convals)

	# Create views into the multipliers
	λ = map(1:ncon) do i
		map(jac.cinds[i]) do ind
			view(data.λ, ind)
		end
	end

	# Create views into the Hessian matrix
	G = data.G
	zinds = gen_zinds(n,m,N,isequal)
	hess1 = map(zip(cons)) do (inds,con)
		zind = get_inds(con, n̄, m)[1]
		map(enumerate(inds)) do (i,k)
			zind_ = zind .+ ((k-1)*(n+m))
			view(G, zind_, zind_)
		end
	end
	hess2 = map(zip(cons)) do (inds,con)
		zind = get_inds(con, n̄, m)
		map(enumerate(inds)) do (i,k)
			if length(zind) > 1
				zind_ = zind[2] .+ ((k-1)*(n+m))
			else
				zind_ = (1:0) .+ ((k-1)*(n+m))
			end
			view(G, zind_, zind_)
		end
	end
	hess = map(zip(hess1, hess2)) do (h1,h2)
		[h1 h2]
	end

	NLPConstraintSet(convals, errvals, jac, λ, hess, zeros(ncon))
end

@inline get_convals(conSet::NLPConstraintSet) = conSet.convals
@inline get_errvals(conSet::NLPConstraintSet) = conSet.errvals

function norm_violation(conSet::NLPConstraintSet, p=2)
	norm(conSet.d, p)
end

@inline ∇jacobian!(conSet::NLPConstraintSet, Z) = ∇jacobian!(conSet.hess, conSet, Z, conSet.λ)

function reset_views!(conSet::NLPConstraintSet, data::NLPData)
	D,d = data.D, data.d
	v = data.v
	λ = data.λ
	for i = 1:length(conSet)
		cval = conSet.convals[i]
		for (j,k) in enumerate(cval.inds)
			cval.vals[j] = change_parent(cval.vals[j], d)
			conSet.λ[i][j] = change_parent(conSet.λ[i][j], λ)
			for l = 1:size(cval.jac,2)
				jac = cval.jac[j,l]
				if jac isa Base.ReshapedArray{<:Any,2,<:SubArray}
					parent = v
				else
					parent = D
				end
				cval.jac[j,l] = change_parent(cval.jac[j,l], parent)
			end
		end
	end
end

function change_parent(x::SubArray, P::AbstractArray)
	return view(P, x.indices...)
end

function change_parent(x::Base.ReshapedArray{<:Any,2,<:SubArray}, P::AbstractArray)
	return reshape(view(P, x.parent.indices...), x.dims)
end

#--- NLP Cost Functions
"""
	QuadraticViewCost{n,m,T}

A quadratic cost that is a view into a large sparse matrix
"""
struct QuadraticViewCost{n,m,T} <: QuadraticCostFunction{n,m,T}
	Q::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	R::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	H::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	q::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
	r::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
	c::T
	zeroH::Bool
	terminal::Bool
	function QuadraticViewCost(Q::SubArray, R::SubArray, H::SubArray,
		q::SubArray, r::SubArray, c::Real; checks::Bool=true, terminal::Bool=false)
		if checks
			TrajOptCore.run_posdef_checks(Q,R)
		end
		n,m = length(q), length(r)
        T = promote_type(eltype(Q), eltype(R), eltype(H), eltype(q), eltype(r), typeof(c))
        zeroH = norm(H,Inf) ≈ 0
		new{n,m,T}(Q, R, H, q, r, c, zeroH, terminal)
	end
end

function QuadraticViewCost(G::SparseMatrixCSC, g::Vector,
		cost::QuadraticCostFunction, k::Int)
	n,m = state_dim(cost), control_dim(cost)
	ix = (k-1)*(n+m) .+ (1:n)
	iu = ((k-1)*(n+m) + n) .+ (1:m)
	NN = length(g)

	Q = view(G,ix,ix)
	q = view(g,ix)

	if cost.Q isa Diagonal
		for i = 1:n; Q[i,i] = cost.Q[i,i] end
	else
		Q .= cost.Q
	end
	q .= cost.q

	# Point the control-dependent values to null matrices at the terminal time step
	if cost.terminal &&  NN == k*n + (k-1)*m
		R = view(spzeros(m,m), 1:m, 1:m)
		H = view(spzeros(m,n), 1:m, 1:n)
		r = view(zeros(m), 1:m)
	else
		R = view(G,iu,iu)
		H = view(G,iu,ix)
		r = view(g,iu)
		if cost.R isa Diagonal
			for i = 1:m; R[i,i] = cost.R[i,i] end
		else
			R .= cost.R
		end
		r .= cost.r
		if !is_blockdiag(cost)
			H .= cost.H
		end
	end

	QuadraticViewCost(Q, R, H, q, r, cost.c, checks=false, terminal=cost.terminal)
end

is_blockdiag(cost::QuadraticViewCost) = cost.zeroH

function reset_views!(obj::Objective{<:QuadraticViewCost}, data::NLPData)
	N = length(obj)
	G,g = data.G, data.g
	for k = 1:N
		obj.cost[k] = change_parent(obj[k], G, g)
	end
end

function change_parent(costfun::QuadraticViewCost, G, g)
	Q = change_parent(costfun.Q, G)
	q = change_parent(costfun.q, g)
	if !costfun.terminal
		R = change_parent(costfun.R, G)
		H = change_parent(costfun.H, G)
		r = change_parent(costfun.r, g)
	else
		R = costfun.R
		H = costfun.H
		r = costfun.r
	end
	QuadraticViewCost(Q, R, H, q, r, costfun.c, checks=false, terminal=costfun.terminal)
end

"""
	ViewKnotPoint{T,n,m}

An `AbstractKnotPoint` whose data is a view into the vector containing all primal variables
in the trajectory optimization problem.
"""
struct ViewKnotPoint{T,N,M} <: AbstractKnotPoint{T,N,M}
    z::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T
    t::T
    function ViewKnotPoint(z::SubArray, _x::SVector{N,Int}, _u::SVector{M,Int},
            dt::T1, t::T2) where {N,M,T1,T2}
        T = promote_type(T1,T2)
        new{T,N,M}(z, _x, _u, dt, t)
    end
end

function ViewKnotPoint(z::SubArray, n, m, dt, t=0.0)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    ViewKnotPoint(z, ix, iu, dt, t)
end

#--- NLP Trajectories
"""
	TrajData{n,m,T}

Describes the partitioning of the vector of primal variables, where `xinds[k]` and `uinds[k]`
give the states and controls at time step `k`, respectively. `t` is the vector of times
and `dt` are the time step lengths for each time step.
"""
struct TrajData{n,m,T}
	xinds::Vector{SVector{n,Int}}
	uinds::Vector{SVector{m,Int}}
	t::Vector{T}
	dt::Vector{T}
end

function TrajData(Z::Traj{n,m}) where {n,m}
	N = length(Z)
	Nu = RobotDynamics.is_terminal(Z[end]) ? N-1 : N
	xinds = [Z[k]._x .+ (k-1)*(n+m) for k = 1:N]
	uinds = [Z[k]._u .+ (k-1)*(n+m) for k  = 1:Nu]
	t = get_times(Z)
	dt = [z.dt for z in Z]
	TrajData(xinds, uinds, t, dt)
end

Base.length(Zdata::TrajData) = length(Zdata.xinds)

function RobotDynamics.StaticKnotPoint(Z::Vector, Zdata::TrajData{n,m}, k::Int) where {n,m}
	x = Z[Zdata.xinds[k]]
	if k <= length(Zdata.uinds)
		u = Z[Zdata.uinds[k]]
	else
		u = @SVector zeros(m)
	end
	dt = Zdata.dt[k]
	t = Zdata.t[k]
	StaticKnotPoint(x,u,dt,t)
end

"""
	NLPTraj{n,m,T} <: AbstractTrajectory{n,m,T}

A trajectory of states and controls, where the underlying data storage is a large vector.

Supports indexing and iteration, where the elements are `StaticKnotPoint`s.
"""
mutable struct NLPTraj{n,m,T} <: AbstractTrajectory{n,m,T}
	Z::Vector{T}
	Zdata::TrajData{n,m,Float64}
end

function NLPTraj(Z::AbstractTrajectory)
	NN = num_vars(Z)
	Zvec = zeros(NN)
	Zdata = TrajData(Z)
	Ztraj = NLPTraj(Zvec, Zdata)
	copyto!(Ztraj, Z)
	return Ztraj
end

@inline Base.getindex(Z::NLPTraj, k::Int) = StaticKnotPoint(Z.Z, Z.Zdata, k)
function Base.setindex!(Z::NLPTraj, z::AbstractKnotPoint, k::Int)
	Z.Z[Z.Zdata.xinds[k]] = state(z)
	if k < length(Z) || RobotDynamics.terminal_control(Z)
		Z.Z[Z.Zdata.uinds[k]] = control(z)
	end
	return z
end
@inline Base.iterate(Z::NLPTraj) = length(Z.Zdata) == 0 ? nothing : (Z[1],1)
@inline Base.iterate(Z::NLPTraj, k::Int) = k >= length(Z.Zdata) ? nothing : (Z[k+1],k+1)
@inline Base.length(Z::NLPTraj) = length(Z.Zdata)
@inline Base.size(Z::NLPTraj) = (length(Z.Zdata),)
@inline Base.eltype(Z::NLPTraj{n,m,T}) where {n,m,T} = StaticKnotPoint{n,m,T,n+m}
@inline Base.IteratorSize(Z::NLPTraj) = Base.HasLength()
@inline Base.IteratorEltype(Z::NLPTraj) = Base.HasEltype()
@inline Base.firstindex(Z::NLPTraj) = 1
@inline Base.lastindex(Z::NLPTraj) = length(Z)

function RobotDynamics.set_states!(Z::NLPTraj, X0)
	xinds = Z.Zdata.xinds
	for k in eachindex(X0)
		Z.Z[xinds[k]] = X0[k]
	end
end

function RobotDynamics.set_controls!(Z::NLPTraj, U0)
	uinds = Z.Zdata.uinds
	for k in eachindex(U0)
		Z.Z[uinds[k]] = U0[k]
	end
end

function RobotDynamics.rollout!(::Type{Q}, model::AbstractModel, Z::NLPTraj, x0=state(Z[1])) where Q <: RD.QuadratureRule
	xinds = Z.Zdata.xinds
	Z.Z[xinds[1]] = x0
	for k = 1:length(Z)-1
		Z.Z[xinds[k+1]] = RD.discrete_dynamics(Q, model, Z[k])
	end
end

#--- TrajOpt NLP Problem

mutable struct NLPOpts{T}
	reset_views::Bool
end

function NLPOpts(;
		reset_views::Bool = false
		)
	NLPOpts{Float64}(reset_views)
end


"""
	TrajOptNLP{n,m,T}

Represents a trajectory optimization problem as a generic nonlinear program (NLP). Convenient
for use with direct methods that manipulate the decision variables across all time steps as
as a single vector (i.e. a "batch" formulation).

# Constructor
	TrajOptNLP(prob::Problem; remove_bounds, jac_type)

If `remove_bounds = true`, any constraints that can be expressed as simple upper and lower
bounds on the primal variables (the states and controls) are removed from the `ConstraintList`
and treated separately.

Options for `jac_type`
- `:sparse`: Use a `SparseMatrixCSC` to represent the constraint Jacobian.
- `:vector`: Use `(v,r,c)` tuples to represent the constraint Jacobian, where
`D[r[i],c[i]] = v[i]` if `D` is the constraint Jacobian.
"""
struct TrajOptNLP{n,m,T} <: MOI.AbstractNLPEvaluator
	model::AbstractModel
	zinds::Vector{UnitRange{Int}}

	# Data
	data::NLPData{T}

	# Objective
	obj::AbstractObjective
	E::Objective{QuadraticViewCost{n,m,T}}

	# Constraints
	conSet::NLPConstraintSet{T}

	# Solution
	Z::NLPTraj{n,m,T}

	# Options
	opts::NLPOpts{T}
end

function TrajOptNLP(prob::Problem; remove_bounds::Bool=false, jac_type=:sparse, add_dynamics=false)
	if add_dynamics
		add_dynamics_constraints!(prob)
	end
	n,m,N = size(prob)
	NN = N*n + (N-1)*m  # number of primal variables

	cons = get_constraints(prob)

	# Remove goal and bound constraints and store them in data.zL and data.zU
	zL = fill(-Inf,NN)
	zU = fill(+Inf,NN)
	if remove_bounds
		cons = copy(cons)
		primal_bounds!(zL, zU, cons, true)
		num_constraints!(cons)
	end
	P = sum(num_constraints(cons))
	jac = JacobianStructure(cons)

	data = NLPData(NN, P, jac.nD)
	data.zL = zL
	data.zU = zU

	conSet = NLPConstraintSet(prob.model, cons, data, jac_type=jac_type)

	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N-1]
	push!(zinds, (N-1)*(n+m) .+ (1:n))

	E = Objective([QuadraticViewCost(
			data.G, data.g, QuadraticCost{Float64}(n, m, terminal=(k==N)),k)
			for k = 1:N])

	Z = NLPTraj(prob.Z)

	opts = NLPOpts()
	TrajOptNLP(prob.model, zinds, data, prob.obj, E, conSet, Z, opts)
end

@inline num_knotpoints(nlp::TrajOptNLP) = length(nlp.zinds)
@inline RobotDynamics.num_vars(nlp::TrajOptNLP) = length(nlp.data.g)
@inline num_constraints(nlp::TrajOptNLP) = length(nlp.data.d)

@inline get_primals(nlp::TrajOptNLP) = nlp.Z.Z
@inline get_duals(nlp::TrajOptNLP) = nlp.data.λ
@inline get_trajectory(nlp::TrajOptNLP) = nlp.Z
@inline get_constraints(nlp::TrajOptNLP) = nlp.conSet
@inline get_model(nlp::TrajOptNLP) = nlp.model
@inline max_violation(nlp::TrajOptNLP) = max_violation(get_constraints(nlp))
@inline initial_trajectory!(nlp::TrajOptNLP, Z0::AbstractTrajectory) = 
	copyto!(get_trajectory(nlp), Z0)

function integration(nlp::TrajOptNLP)
	conSet = get_constraints(nlp)
	for i = 1:length(conSet)
		if conSet.convals[i].con isa DynamicsConstraint
			return integration(conSet.convals[i].con)
		end
	end
end
rollout!(nlp::TrajOptNLP) = rollout!(integration(nlp), get_model(nlp), get_trajectory(nlp))

#---  Evaluation methods

"""
	eval_f(nlp::TrajOptNLP, Z)

Evalate the cost function at `Z`.
"""
function eval_f(nlp::TrajOptNLP, Z=get_primals(nlp))
	if eltype(Z) !== eltype(nlp.Z.Z)
		Z_ = NLPTraj(Z, nlp.Z.Zdata)
	else
		nlp.Z.Z = Z
		Z_ = nlp.Z
	end
	return cost(nlp.obj, Z_)
end
function cost(nlp::TrajOptNLP, Z=get_trajectory(nlp))
	if Z !== get_trajectory(nlp)
		nlp.Z.Z = Z
	end
	eval_f(nlp)
end

"""
	grad_f!(nlp::TrajOptNLP, Z, g)

Evaluate the gradient of the cost function for the vector of decision variables `Z`, storing
	the result in the vector `g`.
"""
function grad_f!(nlp::TrajOptNLP, Z=get_primals(nlp), g=nlp.data.g)
	N = num_knotpoints(nlp)
	nlp.Z.Z = Z
	cost_gradient!(nlp.E, nlp.obj, nlp.Z)
	if g !== nlp.data.g
		copyto!(g, nlp.data.g)
		if nlp.opts.reset_views
			println("reset gradient views")
			nlp.data.g = g
			reset_views!(nlp.E, nlp.data)
		end
	end
	return g
end

"""
	hess_f!(nlp::TrajOptNLP, Z, G)

Evaluate the hessian of the cost function for the vector of decision variables `Z`,
	storing the result in `G`, a sparse matrix.
"""
function hess_f!(nlp::TrajOptNLP, Z=get_primals(nlp), G=nlp.data.G)
	N = num_knotpoints(nlp)
	nlp.Z.Z = Z
	cost_hessian!(nlp.E, nlp.obj, nlp.Z, init=true)  # TODO: figure out how to not require the reset
	if G !== nlp.data.G
		copyto!(G, nlp.data.G)
		if nlp.opts.reset_views
			println("reset Hessian views")
			nlp.data.G = G
			reset_views!(nlp.E, nlp.data)
		end
	end
	return G
end

"""
	hess_f_structure(nlp::TrajOptNLP)

Returns a sparse matrix `D` of the same size as the constraint Jacobian, corresponding to
the sparsity pattern of the constraint Jacobian. Additionally, `D[i,j]` is either zero or
a unique index from 1 to `nnz(D)`.
"""
function hess_f_structure(nlp::TrajOptNLP)
	NN = num_vars(nlp)
	N = num_knotpoints(nlp)
	n,m = size(nlp.model)
	G = spzeros(Int, NN, NN)
	if nlp.obj isa Objective{<:DiagonalCost}
		for i = 1:NN
			G[i,i] = i
		end
	else
		zinds = nlp.zinds
		off = 0
		for k = 1:N
			nm = length(zinds[k])
			blk = reshape(1:nm^2, nm, nm)
			view(G, zinds[k], zinds[k]) .= blk .+ off
			off += nm^2
		end
	end
	return G
end

"""
	get_rc(A::SparseMatrixCSC)

Given a matrix `A` specifying the sparsity structure, where each non-zero element of `A`
is a unique integer ranging from 1 to `nnz(A)`, return the list of row-column pairs such that
`A[r[i],c[i]] = i`.
"""
function get_rc(A::SparseMatrixCSC)
    row,col,inds = findnz(A)
    v = sortperm(inds)
    row[v],col[v]
end

"""
	eval_c!(nlp::TrajOptNLP, Z, c)

Evaluate the constraints at `Z`, storing the result in `c`.
"""
function eval_c!(nlp::TrajOptNLP, Z=get_primals(nlp), c=nlp.data.d)
	if eltype(Z) !== eltype(nlp.Z.Z)
		# Back-up if trying to ForwardDiff
		Z_ = NLPTraj(Z, nlp.Z.Zdata)
	else
		nlp.Z.Z = Z
		Z_ = nlp.Z
	end
	evaluate!(nlp.conSet, Z_)
	if c !== nlp.data.d
		copyto!(c, nlp.data.d)
		if nlp.opts.reset_views
			println("reset constraint views")
			nlp.data.d = c
			reset_views!(nlp.conSet, nlp.data)
		end
	end
	return c
end

"""
	jac_c!(nlp::TrajOptNLP, Z, C)

Evaluate the constraint Jacobian at `Z`, storing the result in `C`.
"""
function jac_c!(nlp::TrajOptNLP, Z=get_primals(nlp), C::AbstractArray=nlp.data.D)
	nlp.Z.Z = Z
	jacobian!(nlp.conSet, nlp.Z)
	if C isa AbstractMatrix && C !== nlp.data.D
		copyto!(C, nlp.data.C)
		if nlp.opts.reset_views
			nlp.data.D = C
			reset_views!(nlp.conSet, nlp.data)
		end
	elseif C isa AbstractVector && C != nlp.data.v
		copyto!(C, nlp.data.v)
		if nlp.opts.reset_views
			println("reset Jacobian views")
			nlp.data.v = C
			reset_views!(nlp.conSet, nlp.data)
		end
	end
	return C
end

"""
	jacobian_structure(nlp::TrajOptNLP)

Returns a sparse matrix `D` of the same size as the constraint Jacobian, corresponding to
the sparsity pattern of the constraint Jacobian. Additionally, `D[i,j]` is either zero or
a unique index from 1 to `nnz(D)`.
"""
@inline jacobian_structure(nlp::TrajOptNLP) = jacobian_structure(nlp.conSet.jac)


"""
	hess_L(nlp::TrajOptNLP, Z, λ, G)

Calculate the Hessian of the Lagrangian `G`, with the vector of current primal variables `Z`
and dual variables `λ`.
"""
function hess_L!(nlp::TrajOptNLP, Z, λ=nlp.data.λ, G=nlp.data.G)
	nlp.Z.Z = Z
	if λ !== nlp.data.λ
		copyto!(nlp.data.λ, λ)  # TODO: reset views instead of copying
	end

	# Cost hessian
	hess_f!(nlp, Z, G)

	# Add Second-order constraint expansion
	∇jacobian!(nlp.conSet, nlp.Z)

	if G !== nlp.data.G
		copyto!(G, nlp.data.G)  # TODO: reset views instead of copying
	end
	return G
end

function ∇jac_c!(nlp::TrajOptNLP, Z=get_primals(nlp), λ=nlp.data.λ, C=nlp.data.G)
	C .= 0  # zero out since ∇jacobian adds to the current result

	nlp.Z.Z = Z
	if λ !== nlp.data.λ
		copyto!(nlp.data.λ, λ)  # TODO: reset views instead of copying
	end

	# Add Second-order constraint expansion
	∇jacobian!(nlp.conSet, nlp.Z)

	if C !== nlp.data.G
		copyto!(C, nlp.data.G)  # leave as copy since nlp.data.G is the hessian of the Lagrangian (or cost)
	end
	return C
end

"""
	primal_bounds!(nlp::TrajOptNLP, zL, zU)

Get the lower and upper bounds on the primal variables.
"""
function primal_bounds!(nlp::TrajOptNLP, zL=nlp.data.zL, zU=nlp.data.zU)
	if zL !== nlp.data.zL
		zL .= nlp.data.zL
		zU .= nlp.data.zU
		nlp.data.zL = zL
		nlp.data.zU = zU
	end
	return zL, zU
end


"""
	constraint_type(nlp::TrajOptNLP)

Build a vector of length `IE = num_constraints(nlp)` where `IE[i]` is the type of constraint
for constraint `i`.

Legend:
 - 0 -> Inequality
 - 1 -> Equality
"""
function constraint_type(nlp::TrajOptNLP)
	# IE = zeros(Int, num_constraints(nlp))
	IE = Vector{Symbol}(undef, num_constraints(nlp))
	constraint_type!(nlp, IE)
end
function constraint_type!(nlp::TrajOptNLP, IE)
	conSet = nlp.conSet

	for i = 1:length(conSet)
		conval = conSet.convals[i]
		cinds = conSet.jac.cinds[i]
		for j = 1:length(cinds)
			v = sense(conval.con) == Equality() ? :Equality : :Inequality
			IE[cinds[j]] .= v
		end
	end
	return IE
end

function constraint_bounds(nlp::TrajOptNLP)
	IE = constraint_type(nlp)
	P = length(IE)
	cL = zeros(P)
	cU = zeros(P)
	for i = 1:P
		if IE[i] == :Inequality
			cL[i] = -Inf
		elseif i == :Equality
			cL[i] = 0
		end
		cU[i] = 0
	end
	return cL, cU
end

############################################################################################
#                               MATH OPT INTERFACE
############################################################################################

MOI.features_available(nlp::TrajOptNLP) = [:Grad, :Jac]
MOI.initialize(nlp::TrajOptNLP, features) = nothing

function MOI.jacobian_structure(nlp::TrajOptNLP)
	D = jacobian_structure(nlp)
	r,c = get_rc(D)
	collect(zip(r,c))
end

MOI.hessian_lagrangian_structure(nlp::TrajOptNLP) = []

@inline MOI.eval_objective(nlp::TrajOptNLP, Z) = eval_f(nlp, Z)
@inline MOI.eval_objective_gradient(nlp::TrajOptNLP, grad_f, Z) = grad_f!(nlp, Z, grad_f)
@inline MOI.eval_constraint(nlp::TrajOptNLP, g, Z) = eval_c!(nlp, Z, g)
@inline MOI.eval_constraint_jacobian(nlp::TrajOptNLP, jac, Z) = jac_c!(nlp, Z, jac)
@inline MOI.eval_hessian_lagrangian(::TrajOptNLP, H, x, σ, μ) = nothing

function build_MOI!(nlp::TrajOptNLP, optimizer::MOI.AbstractOptimizer)
	NN = num_vars(nlp)

	zL,zU = primal_bounds!(nlp)

	has_objective = true
	cL,cU = constraint_bounds(nlp)
	nlp_bounds = MOI.NLPBoundsPair.(cL, cU)
	block_data = MOI.NLPBlockData(nlp_bounds, nlp, has_objective)

	Z = MOI.add_variables(optimizer, NN)
	MOI.add_constraints(optimizer, Z, MOI.LessThan.(zU))
	MOI.add_constraints(optimizer, Z, MOI.GreaterThan.(zL))

	MOI.set(optimizer, MOI.VariablePrimalStart(), Z, nlp.Z.Z)

	MOI.set(optimizer, MOI.NLPBlock(), block_data)
	MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	return optimizer
	# return optimizer
	MOI.optimize!(optimizer)
	V = [MOI.VariableIndex(k) for k = 1:NN]
	res = MOI.get(optimizer, MOI.VariablePrimal(), V)
	copyto!(nlp.Z.Z, res)
	return nlp
end
