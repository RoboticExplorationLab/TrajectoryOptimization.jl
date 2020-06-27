############################################################################################
#                                  IMPLICIT METHODS 								       #
############################################################################################

function discrete_dynamics(::Type{RK3}, model::AbstractModel, x::SVector{N,T}, u::SVector{M,T},
		t, dt::T) where {N,M,T}
    k1 = dynamics(model, x,             u, t       )*dt;
    k2 = dynamics(model, x + k1/2,      u, t + dt/2)*dt;
    k3 = dynamics(model, x - k1 + 2*k2, u, t + dt  )*dt;
    x + (k1 + 4*k2 + k3)/6
end

function discrete_dynamics(::Type{RK2}, model::AbstractModel, x::SVector, u::SVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	x + k2
end

function discrete_dynamics(::Type{RK4}, model::AbstractModel, x::SVector, u::SVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	k3 = dynamics(model, x + k2/2, u, t + dt/2)*dt
	k4 = dynamics(model, x + k3,   u, t + dt  )*dt
	x + (k1 + 2k2 + 2k3 + k4)/6
end

function discrete_dynamics(::Type{RK8}, model::AbstractModel, y::SVector{N,T}, u::SVector{M,T},t, dt::T) where {N,M,T}
	α    = @SVector [ 2/27, 1/9, 1/6, 5/12, .5, 5/6, 1/6, 2/3, 1/3, 1, 0, 1 ];
	β    = @SMatrix [  2/27       0       0      0        0         0       0         0     0      0     0 0 0;
	1/36       1/12    0      0        0         0       0         0     0      0     0 0 0;
	1/24       0       1/8    0        0         0       0         0     0      0     0 0 0;
	5/12       0       -25/16 25/16    0         0       0         0     0      0     0 0 0;
	.05        0       0      .25      .2        0       0         0     0      0     0 0 0;
	-25/108    0       0      125/108  -65/27    125/54  0         0     0      0     0 0 0;
	31/300     0       0      0        61/225    -2/9    13/900    0     0      0     0 0 0;
	2          0       0      -53/6    704/45    -107/9  67/90     3     0      0     0 0 0;
	-91/108    0       0      23/108   -976/135  311/54  -19/60    17/6  -1/12  0     0 0 0;
	2383/4100  0       0      -341/164 4496/1025 -301/82 2133/4100 45/82 45/164 18/41 0 0 0;
	3/205      0       0      0        0         -6/41   -3/205    -3/41 3/41   6/41  0 0 0;
	-1777/4100 0       0      -341/164 4496/1025 -289/82 2193/4100 51/82 33/164 12/41 0 1 0]
	χ     = @SVector[0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/840, 41/840 ]; # Difference between two bottom layers of butcher tableau

	f = y*zeros(eltype(y),1,13)

	f[:,1] = dynamics(model,y,u,t)
	for j = 1:12
	f[:,j+1] = dynamics(model, y + dt*f*β[j,:],u,t + α[j]*dt)
	end
	return y + dt*f*χ
end


############################################################################################
#                                  EXPLICIT METHODS 								       #
############################################################################################

# Hermite Simpson
function evaluate!(vals::Vector{<:AbstractVector}, con::DynamicsConstraint{HermiteSimpson},
		Z::Traj, inds=1:length(Z)-1)
	N = length(Z)
	model = con.model
	fVal = con.fVal
	xMid = con.xMid

	for k = inds.start:inds.stop+1
		fVal[k] = dynamics(model, Z[k])
	end
	for k in inds
		xMid[k] = (state(Z[k]) + state(Z[k+1]))/2 + Z[k].dt/8 * (fVal[k] - fVal[k+1])
	end
	for k in inds
		Um = (control(Z[k]) + control(Z[k+1]))*0.5
		fValm = dynamics(model, xMid[k], Um)
		vals[k] = state(Z[k]) - state(Z[k+1]) + Z[k].dt*(fVal[k] + 4*fValm + fVal[k+1])/6
	end
end

function jacobian!(∇c::Vector{<:SizedMatrix}, con::DynamicsConstraint{HermiteSimpson,L,T,n,m},
		Z::Traj, inds=1:length(Z)-1) where {L,T,n,m}
	N = length(Z)
	model = con.model
	∇f = con.∇f
	A = con.A
	B = con.B
	xMid = con.xMid
	In = Diagonal(@SVector ones(n))
	Fmid = con.∇fMid[1]
	Amid = con.Am[1]
	Bmid = con.Bm[1]

	xi = Z[1]._x
	ui = Z[1]._u

	# Compute dynamics Jacobian at each knot point
	for k = inds.start:inds.stop+1
		jacobian!(∇f[k], model, Z[k])
	end

	# TODO: write an in-place version for large arrays where this will probably die at compile time
	for k in inds
		Um = (control(Z[k]) + control(Z[k+1]))*0.5
		zMid = StaticKnotPoint([xMid[k]; Um], xi, ui, Z[k].dt, Z[k].t)
		jacobian!(Fmid, model, zMid)
		A1 = SMatrix{n,n}(A[k])
		B1 = SMatrix{n,m}(B[k])
		Am = SMatrix{n,n}(Amid)
		Bm = SMatrix{n,m}(Bmid)
		A2 = SMatrix{n,n}(A[k+1])
		B2 = SMatrix{n,m}(B[k+1])
		dt = Z[k].dt
		A_ = dt/6*(A1 + 4Am*( dt/8*A1 + In/2)) + In
		B_ = dt/6*(B1 + 4Am*( dt/8*B1) + 2Bm)
		C_ = dt/6*(A2 + 4Am*(-dt/8*A2 + In/2)) - In
		D_ = dt/6*(B2 + 4Am*(-dt/8*B2) + 2Bm)
		∇c[k] .= [A_ B_ C_ D_]
	end
end

function cost(obj, dyn_con::DynamicsConstraint{HermiteSimpson}, Z)
	N = length(Z)
	model = dyn_con.model
    xMid = dyn_con.xMid
	fVal = dyn_con.fVal
	for k = 1:N
		fVal[k] = dynamics(model, Z[k])
	end
	for k = 1:N-1
		xMid[k] = (state(Z[k]) + state(Z[k+1]))/2 + Z[k].dt/8 * (fVal[k] - fVal[k+1])
	end
	J = 0.0
	for k = 1:N-1
		Um = (control(Z[k]) + control(Z[k+1]))*0.5
		J += Z[k].dt/6 * (stage_cost(obj[k], state(Z[k]), control(Z[k])) +
					    4*stage_cost(obj[k], xMid[k], Um) +
					      stage_cost(obj[k], state(Z[k+1]), control(Z[k+1])))
	end
	J += stage_cost(obj[N], state(Z[N]))
	return J
end

function cost_gradient!(E, obj, dyn_con::DynamicsConstraint{HermiteSimpson},
		Z::Vector{<:KnotPoint{<:Any,n,m}}) where {n,m}
	N = length(Z)
	xi = Z[1]._x
	ui = Z[1]._u

	model = dyn_con.model
	fVal = dyn_con.fVal
	xMid = dyn_con.xMid
	∇f = dyn_con.∇f
	A = dyn_con.A
	B = dyn_con.B
	grad = dyn_con.grad
	Fm = dyn_con.∇fMid[1]
	Amid = dyn_con.Am[1]
	Bmid = dyn_con.Bm[1]

	for k = 1:N
		fVal[k] = dynamics(model, Z[k])
	end
	for k = 1:N-1
		xMid[k] = (state(Z[k]) + state(Z[k+1]))/2 + Z[k].dt/8 * (fVal[k] - fVal[k+1])
	end
	for k = 1:N
		jacobian!(∇f[k], model, Z[k])
		E[k].x .*= 0
		E[k].u .*= 0
	end

	for k in 1:N-1
		x1, u1 = state(Z[k]),   control(Z[k])
		x2, u2 = state(Z[k+1]), control(Z[k+1])
		xm, um = xMid[k], 0.5*(u1 + u2)

		zMid = StaticKnotPoint([xm; um], xi, ui, Z[k].dt, Z[k].t)
		jacobian!(Fm, model, zMid)
		A1 = SMatrix{n,n}(A[k])
		B1 = SMatrix{n,m}(B[k])
		Am = SMatrix{n,n}(Amid)
		Bm = SMatrix{n,m}(Bmid)
		A2 = SMatrix{n,n}(A[k+1])
		B2 = SMatrix{n,m}(B[k+1])
		dt = Z[k].dt

		gradient!(grad[1], obj[k], x1, u1)
		gradient!(grad[2], obj[k], x2, u2)
		gradient!(grad[3], obj[k], xm, um)

		∇x1,∇u1 = grad[1].x, grad[1].u #gradient(obj[k], x1, u1)
		∇x2,∇u2 = grad[2].x, grad[2].u #gradient(obj[k], x2, u2)
		∇xm,∇um = grad[3].x, grad[3].u #gradient(obj[k], xm, um)

		E[k].x   .+= dt/6 * (∇x1 + 4*( dt/8 * A1 + I/2)'∇xm)
		E[k].u   .+= dt/6 * (∇u1 + 4*( ( dt/8 * B1)'∇xm + 0.5I'*∇um))
		E[k+1].x .+= dt/6 * (∇x2 + 4*(-dt/8 * A2 + I/2)'∇xm)
		E[k+1].u .+= dt/6 * (∇u2 + 4*( (-dt/8 * B2)'∇xm + 0.5I'*∇um))
	end

	gradient!(grad[1], obj[N], state(Z[N]), control(Z[N]))
	E[N].x .+= grad[1].x #gradient(obj[N], state(Z[N]), control(Z[N]))[1]
	return nothing
end
