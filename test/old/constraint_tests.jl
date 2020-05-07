using Test

function alloc_con(con,z)
    p,w = length(con), TO.width(con)
    ∇c = zeros(p,w)
    allocs  = @allocated evaluate(con,z)
    allocs += @allocated jacobian!(∇c, con,z)
end

model = Dynamics.DubinsCar()
x,u = rand(model)
z = KnotPoint(x,u,0.1)
n,m = size(model)

# All static
A = @SMatrix rand(4,3)
b = @SVector rand(4)
∇c = zeros(4,3)
con = LinearConstraint{Inequality,State}(n,m,A,b)
evaluate(con, z)
@test length(con) == 4
@test TO.width(con) == 3
@test evaluate(con, z) == A*x - b
jacobian!(∇c, con, x)
@test ∇c == A
@test alloc_con(con,z) == 0
@test state_dim(con) == n
@test_throws MethodError control_dim(con)

# both dynamic
con = LinearConstraint{Inequality,State}(n,m,Matrix(A),Vector(b))
@test evaluate(con, z) == A*x - b
jacobian!(∇c, con, x)
@test ∇c == A
@test alloc_con(con,z) == 0

# mixed
con = LinearConstraint{Inequality,State}(n,m,A,Vector(b))
@test evaluate(con, z) == A*x - b
jacobian!(∇c, con, x)
@test ∇c == A
@test alloc_con(con,z) == 0

# wrong input size
@test_throws AssertionError LinearConstraint{Inequality,State}(m,m,A,b)
@test_throws AssertionError LinearConstraint{Inequality,Control}(n,m,A,b)
con = LinearConstraint{Inequality,Control}(n,n,A,b)
@test evaluate(con, x) == A*x - b
jacobian!(∇c, con, x)
@test ∇c == A


model = Dynamics.Quadrotor2{UnitQuaternion{Float64,VectorPart}}()
x,u = rand(model)
n,m = size(model)

# Goal Constraint
xf = @SVector rand(n)
con = GoalConstraint(xf)
size(con) == (13,13)
∇c = zeros(size(con)...)
@test evaluate(con, x) == x-xf
@test jacobian!(∇c, con, x) == I(n)

con = GoalConstraint(xf, 1:3)
∇c = zeros(size(con))
@test evaluate(con, x) == (x-xf)[1:3]
@test jacobian!(∇c, con, x) == Matrix(I,3,n)

noquat = collect(1:n)
noquat = deleteat!(noquat, 4:7)
noquat = SVector{9}(noquat)
con = GoalConstraint(xf, noquat)
∇c = zeros(size(con))
@test evaluate(con, x) == (x-xf)[noquat]
@test jacobian!(∇c, con, x) == I(n)[noquat,:]

# Sphere Constraint
xc = @SVector rand(4)
yc = @SVector rand(4)
zc = @SVector rand(4)
r = @SVector fill(0.1,4)
con = SphereConstraint(n, xc, yc, zc, r)
∇c = zeros(size(con))
@test state_dim(con) == n
@test evaluate(con, x) isa SVector{4}
@test norm(jacobian!(∇c, con, x)[:,4:end],Inf) == 0


# Norm Constraint
con = TO.NormConstraint{Equality,Control}(m,4.0)
@test TO.sense(con) == Equality
@test TO.contype(con) == Control
@test evaluate(con, u)[1] == norm(u)^2 - 4
@test_throws MethodError state_dim(con)
@test control_dim(con) == m
@test TO.check_dims(con,n,m)

con = TO.NormConstraint{Inequality,State}(n, 2.0)
@test TO.sense(con) == Inequality
@test TO.contype(con) == State
@test evaluate(con, x)[1] == norm(x)^2 - 2
@test_throws MethodError control_dim(con)
@test state_dim(con) == n
@test TO.check_dims(con,n,m)


# Bounds Constraint
x_max = @SVector fill(+1,n)
x_min = @SVector fill(-1,n)
u_max = @SVector fill(+2,m)
u_min = @SVector fill(-2,m)

bnd = BoundConstraint(n,m, x_max=x_max, x_min=x_min)
@test_throws ArgumentError BoundConstraint(n,m, x_max=x_min, x_min=x_max)
@test_nowarn BoundConstraint(n,m, x_max=1, x_min=x_min)
@test_nowarn BoundConstraint(n,m, x_max=x_max, x_min=-1.)
@test TO.is_bound(bnd)
@test TO.upper_bound(bnd) == [x_max; u_max*Inf]
@test TO.lower_bound(bnd) == [x_min; u_min*Inf]


# Variable Bound Constraint
N = 101
X_max = [copy(x_max) for k = 1:N]
bnd = VariableBoundConstraint(n,m,N, x_max=X_max)
@test state_dim(bnd) == n
@test control_dim(bnd) == m
vals = [@SVector zeros(n) for k = 1:N]
Z = Traj(x,u,0.1,N)
@test_nowarn evaluate!(vals, bnd, Z)
@test jacobian(bnd, Z[1]) == Matrix(I,n,n+m)


# Indexed Constraint
bnd = BoundConstraint(n,m, x_max=x_max, x_min=x_min)
con = TO.IndexedConstraint(2n,2m, bnd)
∇c = zeros(size(con))
x2 = repeat(x,2)
u2 = repeat(u,2)
z = KnotPoint(x,u,0.1)
z2 = KnotPoint(x2,u2,0.1)
@test evaluate(bnd, z) ≈ evaluate(con, z2)
∇c0 = zeros(size(bnd))
jacobian!(∇c0, bnd, z)
jacobian!(∇c, con, z2)
@test ∇c ≈ [∇c0 zeros(26, 17)]

@test TO.width(con) == 2(n+m)
@test TO.width(bnd) == n+m

con = TO.NormConstraint{Equality,Control}(m,4.0)
∇c0 = zeros(size(con))
ix = n .+ @SVector [i for i in 1:n]
iu = m .+ @SVector [i for i in 1:m]
idx = TO.IndexedConstraint(2n,2m, con, ix, iu)
∇c = zeros(size(idx))
@test TO.contype(idx) == Control
@test TO.sense(idx) == Equality
@test evaluate(idx, z2) == evaluate(con, z)
jacobian!(∇c0, con, z)
jacobian!(∇c, idx, z2)
@test ∇c == [zeros(1,m) ∇c0]

con = TO.NormConstraint{Equality,State}(n,4.0)
idx = TO.IndexedConstraint(2n,2m, con)
@test evaluate(idx, z2) == evaluate(con, z)
∇c0 = zeros(size(con))
∇c = zeros(size(idx))
jacobian!(∇c0, con, z)
jacobian!(∇c, idx, z2)
@test ∇c == [∇c0 zeros(1,n)]
