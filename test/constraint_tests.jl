function alloc_con(con,z)
    ∇c = TO.gen_jacobian(con)
    allocs  = @allocated TO.evaluate(con, z)
    allocs += @allocated TO.jacobian!(∇c, con,z)
end

model = Cartpole()
n,m = size(model)
x,u = rand(model)
z = KnotPoint(x,u,0.1)

# Goal Constraint
xf = @SVector rand(n)
goal = GoalConstraint(xf)
@test TO.evaluate(goal, x) ≈ x - xf
C = zeros(n,n)
@test TO.jacobian!(C, goal, z) == true
@test C ≈ I(n)

@test length(goal) == n
@test TO.upper_bound(goal) ≈ zero(xf)
@test TO.lower_bound(goal) ≈ zero(xf)
@test TO.is_bound(goal)
@test TO.check_dims(goal, n, m)
@test TO.check_dims(goal, n+1, m) == false
@test TO.check_dims(goal, n, m+1) == true
@test TO.widths(goal) == (n,)
@test TO.widths(goal, n, m) == (n,)
@test TO.widths(goal, n+1, m) == (n+1,)
@test state_dim(goal) == n

@test GoalConstraint(Vector(xf)).xf isa SVector{n}


# Linear Constraint
p = 5
A = @SMatrix rand(p,n+m)
b = @SVector rand(p)
∇c = zeros(p,n+m)
lin = LinearConstraint(n,m,A,b, Inequality())
@test TO.evaluate(lin, z) ≈ A*z.z - b
@test jacobian!(∇c, lin, z) == true
@test ∇c ≈ A

lin2 = LinearConstraint(n,m, Matrix(A), Vector(b), Inequality())
lin2.A isa SizedMatrix{p,n+m}
lin2.b isa SVector{p}

@test length(lin) == p
@test TO.widths(lin) == (n+m,)
@test state_dim(lin) == n
@test control_dim(lin) == m
@test TO.upper_bound(lin) ≈ zeros(p)
@test TO.lower_bound(lin) ≈ fill(-Inf,p)
@test TO.is_bound(lin) == false

A = @SMatrix rand(p,n)
lin2 = LinearConstraint(n,m, A,b, Equality(), 1:n)
@test TO.evaluate(lin2, z) ≈ A*x - b
∇c = TO.gen_jacobian(lin2)
@test TO.jacobian!(∇c, lin2, z) == true
@test ∇c ≈ [A zeros(p)]

@test TO.widths(lin2) == (n+m,)
@test alloc_con(lin2,z) == 0
@test state_dim(lin2) == n
@test control_dim(lin2) == m
@test TO.upper_bound(lin2) ≈ zeros(p)
@test TO.lower_bound(lin2) ≈ zeros(p)
@test TO.is_bound(lin2) == false

@test TO.sense(lin) == Inequality()
@test TO.sense(lin2) == Equality()

A = @SMatrix rand(p,m)
lin3 = LinearConstraint(n,m, A,b, Inequality(), n .+ (1:m))
@test TO.evaluate(lin3, z) ≈ A*u - b
∇c = TO.gen_jacobian(lin3)
@test TO.jacobian!(∇c, lin3, z) == true
@test ∇c ≈ [zeros(p,n) A]


# Circle/Sphere Constraints
xc = SA[1,1,1]
yc = SA[1,2,3]
r  = SA[1,1,1]
cir = CircleConstraint(n, xc, yc, r)
@test TO.evaluate(cir, z) ≈ -((x[1] .- xc).^2 + (x[2] .- yc).^2 .- r.^2)
∇c = TO.gen_jacobian(cir)
@test TO.jacobian!(∇c, cir, z) == false
@test ∇c ≈ hcat(-2*(x[1] .- xc), -2*(x[2] .- yc), zeros(3,n-2))
@test cir isa CircleConstraint{3,Int}
@test cir isa TO.StateConstraint
@test length(cir) == 3
@test state_dim(cir) == n
@test_throws MethodError control_dim(cir)
@test TO.widths(cir) == (n,)

cir_ = CircleConstraint(n, Float64.(xc), yc, r)
@test cir_ isa CircleConstraint{3,Float64}
cir_ = CircleConstraint{3,Float64}(n, xc, yc, r)
@test cir_ isa CircleConstraint{3,Float64}

cir2 = CircleConstraint(n, Float64.(xc), yc, r, 2, 3)
@test TO.evaluate(cir2, z) ≈ -((x[2] .- xc).^2 + (x[3] .- yc).^2 .- r.^2)
∇c = TO.gen_jacobian(cir2)
@test TO.jacobian!(∇c, cir2, z) == false
@test ∇c ≈ hcat(zeros(3), -2*(x[2] .- xc), -2*(x[3] .- yc), zeros(3,n-3))

@test_throws AssertionError CircleConstraint(n, push(xc,2), yc, r)


zc = SA[3,3,3]
sph = SphereConstraint{3,Int}(n, xc, yc, zc, r)
@test TO.evaluate(sph, z) ≈ -((x[1] .- xc).^2 + (x[2] .- yc).^2  .+ (x[3] .- zc).^2 .- r.^2)
∇c = TO.gen_jacobian(sph)
@test TO.jacobian!(∇c, sph, z) == false
@test ∇c ≈ hcat(-2*(x[1] .- xc), -2*(x[2] .- yc), -2*(x[3] .- zc), zeros(3,n-3))
@test sph isa SphereConstraint{3,Int}
@test sph isa TO.StateConstraint
@test length(sph) == 3

sph_ = SphereConstraint(n, Float64.(xc), yc, zc, r)
@test sph_ isa SphereConstraint{3,Float64}
sph_ = SphereConstraint{3,Float64}(n, xc, yc, zc, r)
@test sph_ isa SphereConstraint{3,Float64}

sph2 = SphereConstraint(n, Float64.(xc), yc, zc, r, 2, 3, 1)
@test sph2.zi == 1
@test TO.evaluate(sph2, z) ≈ -((x[2] .- xc).^2 + (x[3] .- yc).^2 + (x[1] .- zc).^2 .- r.^2)
∇c = TO.gen_jacobian(sph2)
@test TO.jacobian!(∇c, sph2, z) == false
@test ∇c ≈ hcat(-2*(x[1] .- zc), -2*(x[2] .- xc), -2*(x[3] .- yc), zeros(3,n-3))


# Collision Constraint
x1 = SA[1,2]
x2 = SA[3,4]
col = CollisionConstraint(n, x1, x2, 2.)
d = x[x1] - x[x2]
@test TO.evaluate(col, z) ≈ SA[4 - d'd]
∇c = TO.gen_jacobian(col)
TO.jacobian!(∇c, col, z) == false
@test ∇c ≈ [-2d' 2d']
@test length(col) == 1

col_ = CollisionConstraint(n, x1, x2, 1)
@test col_.radius isa Float64
col_ = CollisionConstraint(n, 1:2, x2, 1)
@test col_.x1 isa SVector{2,Int}
@test_throws AssertionError CollisionConstraint(n, 1:2, 1:3, 1.0)


# Norm Constraint
ncon = NormConstraint(n,m, 2.0, Inequality(), 1:n)
@test TO.evaluate(ncon, z) ≈ [x'x - 2]
∇c = TO.gen_jacobian(ncon)
@test TO.jacobian!(∇c, ncon, z) == false
@test ∇c ≈ [2x; 0]'

@test length(ncon) == 1
@test TO.widths(ncon) == (n+m,)
@test TO.sense(ncon) == Inequality()

ncon2 = NormConstraint(n,m, 2.0, Inequality(), :state)
@test TO.evaluate(ncon, z) ≈ TO.evaluate(ncon2, z)

ncon2 = NormConstraint(n, m, 3.0, Equality(), :control)
@test TO.evaluate(ncon2, z) ≈ [u'u - 3]
∇c = TO.gen_jacobian(ncon2)
@test TO.jacobian!(∇c, ncon2, z) == false
@test ∇c ≈ [zeros(n); 2u]'

ncon3 = NormConstraint(n, m, 4.0, Inequality(), SA[1,3,5])
@test TO.evaluate(ncon3, z) ≈ [x[1]^2 + x[3]^2 + u'u - 4]
∇c = TO.gen_jacobian(ncon3)
@test TO.jacobian!(∇c, ncon3, z) == false
@test ∇c ≈ [2x[1] 0 2x[3] 0 2u[1]]



# Bound Constraint
xmin = -@SVector rand(n)
xmax = +@SVector rand(n)
umin = -@SVector rand(m)
umax = +@SVector rand(m)

bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)
@test TO.evaluate(bnd, z) ≈ [x - xmax; u - umax; xmin - x; umin - u]
∇c = TO.gen_jacobian(bnd)
@test TO.jacobian!(∇c, bnd, z) == true
@test ∇c ≈ [I(n+m); -I(n+m)]
@test length(bnd) == 2(n+m)
@test TO.widths(bnd) == (n+m,)
@test TO.upper_bound(bnd) == [xmax; umax]
@test TO.lower_bound(bnd) == [xmin; umin]
@test TO.is_bound(bnd) == true

xmin = pop(pushfirst(xmin, -Inf))
umax = popfirst(push(umax, Inf))
bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)
@test TO.evaluate(bnd, z) ≈ [x - xmax; u[1:end-1] - umax[1:end-1];
    xmin[2:end] - x[2:end]; umin - u]
∇c = TO.gen_jacobian(bnd)
@test TO.jacobian!(∇c, bnd, z) == true
iz = ones(Bool,2(n+m))
iz[n+1] = 0
iz[n+m+1] = 0
@test ∇c ≈ [I(n+m); -I(n+m)][iz, :]
@test length(bnd) == 2(n+m) - 2
@test TO.widths(bnd) == (n+m,)
@test TO.upper_bound(bnd) == [xmax; umax]
@test TO.lower_bound(bnd) == [xmin; umin]
@test TO.is_bound(bnd) == true

bnd_ = BoundConstraint(n,m, x_min=-10, x_max=10, u_min=umin, u_max=umax)
@test TO.evaluate(bnd_, z) ≈ [x .- 10; u[1:end-1] - umax[1:end-1];
    -10 .- x; umin - u]
bnd_ = BoundConstraint(n,m, x_min=-10, x_max=10, u_min=Vector(umin), u_max=umax)
@test TO.evaluate(bnd_, z) ≈ [x .- 10; u[1:end-1] - umax[1:end-1];
    -10 .- x; umin - u]
bnd_ = BoundConstraint(n,m, x_min=-10, x_max=10, u_min=Vector(umin), u_max=MVector(umax))
@test TO.evaluate(bnd_, z) ≈ [x .- 10; u[1:end-1] - umax[1:end-1];
    -10 .- x; umin - u]

xmin = -rand(1:10,n)
xmax = rand(1:10,n)
bnd_ = BoundConstraint(n,m, x_min=xmin, x_max=xmax)
@test TO.evaluate(bnd_, z) ≈ [x .- xmax; xmin .- x]

@test_throws ArgumentError BoundConstraint(n,m, x_min=10, x_max=-10, u_min=umin, u_max=umax)


# Indexed Constraint
n2,m2 = 2n, 2m
idx = TO.IndexedConstraint(n2,m2, bnd)
