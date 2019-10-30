using StaticArrays, ForwardDiff, BenchmarkTools, LinearAlgebra

n,m,N = 3,2,11
dt = 0.1
x = @SVector [1.,2,3,]
u = @SVector [-1.,1]
z = [x;u]
Z = [KnotPoint(x,u,dt) for k = 1:N]


xc = @SVector [0,1,2,3.]
yc = @SVector [0,2,2,0.]
vals = [@SVector zeros(length(xc)) for k = 1:N]
circlecon = CircleConstraint(n,m,xc,yc, 0.5)
generate_jacobian(circlecon)

vals2 = [@SVector zeros(1) for k = 1:N]
normcon = NormConstraint(n,m,1, 1.0)
generate_jacobian(normcon)

@btime evaluate($circlecon, $x, $u)
@btime evaluate($normcon, $x, $u)

@btime jacobian($circlecon, $z)
@btime jacobian($normcon, $z)

con1 = KnotConstraint(circlecon, 1:N)
con2 = KnotConstraint(normcon, 1:4)

function eval_constraint(con, Z)
	for k in con.inds
		con.vals[k] = evaluate(con.con, state(Z[k]), control(Z[k]))
	end
end

function eval_constraints(constraints, Z)
	for con in constraints
		eval_constraint(con, Z)
	end
end


@btime eval_constraint($con1, $Z)
constraints = [con1, con2]
@btime eval_constraints($constraints, $Z)

@btime evaluate($con1, $Z)
@btime jacobian($con1, $Z)

@btime update_active_set!($con1)
@btime duals($con1, 3)

@btime max_violation!($con1)
@btime max_violation!($con2)


conSet = ConstraintSets(constraints, N)
max_violation!(conSet)
@btime max_violation!($conSet)

@btime jacobian($conSet, $Z)


# AL Objective
xf = @SVector zeros(n)
Q = Diagonal(@SVector ones(n))
R = Diagonal(@SVector ones(m))
Qf = Diagonal(@SVector ones(n))
obj = LQRObjective(Q,R,Qf,xf,N)

alobj = StaticALObjective3(obj, conSet)
E = CostExpansion(n,m,N)
@btime cost_expansion($E, $obj, $Z)
cost_expansion(E,alobj,Z)
@btime cost_expansion($Q,$alobj,$Z)

cost_expansion(E, con1, Z)
@btime cost_expansion($E, $con1, $Z)
J = obj.J
@btime cost!($J, $con1, $Z)
@btime cost!($alobj, $Z)






# Custom constraints

struct CustomConstraint{C} <: AbstractConstraint{C}
	n::Int
	m::Int
	p::Int
	c::Function
	∇c::Function
end

function CustomConstraint{C}(n::Int, m::Int, p::Int, c::Function) where C
	generate_jacobian(c, n, m)
	∇c = eval(Symbol("∇" * string(c)))
	CustomConstraint{C}(n,m,p,c,∇c)
end

evaluate(con::CustomConstraint, x, u) = con.c(x,u)
evaluate(con::CustomConstraint, z) = con.c(z)
jacobian(con::CustomConstraint, x, u) = con.∇c(x, u)
jacobian(con::CustomConstraint, z) = con.∇c(z)



# Test constraints
n,m = 3,2
p1 = 3

x = @SVector [1.,2,3]
u = @SVector [-5.,5]
z = [x;u]

x_max = @SVector [3,2,Inf]
x_min = @SVector [-3,-2,-Inf]
u_max = @SVector [Inf, 0.5]
u_min = @SVector [0.0, -0.5]

bnd = BoundConstraint(n,m, x_max=x_max, x_min=x_min, u_max=u_max, u_min=u_min)

@btime evaluate($bnd, $xs, $us)
@btime evaluate($bnd, $xs)

function mycon1(x, u)
	@SVector [x[1]^2 + x[2]^2 - 5, u[1] - 1, u[2] - 1]
end
jacob_eq1(x,u) = [2x[1] 2x[2] 0 0 0;
                0     0     0 1 0;
                0     0     0 0 1];
eq2 = CustomConstraint{Equality}(n,m,3,mycon1)

evaluate(eq2, x, u) == [0,-6,4]

@btime evaluate($eq2,$x,$u)
@btime jacobian($eq2, $z)


mycon2(x,u) = @SVector [sin(x[1]), sin(x[3])]
ineq2 = CustomConstraint{Inequality}(n,m,2,mycon2)
@btime evaluate($ineq2, $x, $u)
@btime jacobian($ineq2, $z)



constraints = [bnd, eq2, ineq2]
constraints = [eq2, eq2]

out = [@SVector zeros(3) for k = 1:2]
function eval_constraints!(out, con, x, u)
	out[2] = evaluate(con, x, u)
end
constraints = (eq2, eq2)
@btime eval_constraints!($out, $eq2, $x, $u)
@btime evaluate($eq2, $x, $u)
@btime evaluate($constraints[2], $x, $u)

constraints = @SVector [eq2, eq2]
@btime evaluate($bnd, $x, $u)




# Quadrotor constraint
n,m = 13,4
dt = 0.1
xs,us = (@SVector rand(n)), (@SVector rand(m))
x,u = Array(xs), Array(us)
zs = [xs;us]
z = [x;u]
Z = KnotPoint(x,u,dt)


struct CircleConstraint{T} <: AbstractConstraint{Inequality}
	n::Int
	m::Int
	p::Int
	r0::SVector{3,T}
	radius::T
end

Base.size(con::CircleConstraint) = (con.n, con.m, con.p)

function evaluate(con::CircleConstraint, x, u)
	p = con.r0
	r = con.radius
	return @SVector [-((x[1] - p[1])^2 + (x[2]-p[2])^2  - r^2)]
end

function evaluate(con::CircleConstraint, z)
	p = con.r0
	r = con.radius
	return @SVector [-((z[1] - p[1])^2 + (z[2]-p[2])^2  - r^2)]
end

r0 = @SVector [0.5, 0.5, 0]
radius = 0.1
con = CircleConstraint(n, m, 1, r0, radius)
@btime evaluate($con, $xs, $us)

generate_jacobian(con)
jacobian(con, zs)
jacobian(con, xs, us)
@btime jacobian($con, $zs)


x_max = @SVector ones(n)
x_min = -@SVector ones(n)
u_max = 2@SVector ones(m)
u_min = -2@SVector ones(m)

bnd = BoundConstraint(n,m, x_max=x_max, x_min=x_min, u_max=u_max, u_min=u_min)

constraints = (con, con, bnd)

Z = [KnotPoint(x,u,dt) for k = 1:10]

function eval_constraints!(cval, constraint, Z::Traj)
	for k in eachindex(cval)
		cval[k] = evaluate(constraint, state(Z[k]), control(Z[k]))
	end
end


function eval_all_constraints!(cvals, constraints, Z::Traj)
	eval_constraints!(cvals[1], constraints[1], Z)
	# eval_constraints!(cvals[2], constraints[2], Z)
	# eval_constraints!(cvals[3], constraints[3], Z)
	# for i in eachindex(cvals)
	# 	eval_constraints!(cvals[i], constraints[i], Z)
	# end
end

@btime eval_constraints!($(cvals[1]), $(constraints[1]), $Z)
constraints = (bnd, bnd, con)
cvals = [[@SVector zeros(size(con)[3]) for k = 1:10] for con in constraints]
@btime eval_all_constraints!($cvals, $constraints, $Z)
