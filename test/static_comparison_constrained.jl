using LinearAlgebra, StaticArrays

############################################################################################
#                             ORIGINAL PROBLEM                                             #
############################################################################################
quad = Dynamics.quadrotor
quad_d = rk3(quad)

n = quad.n; m = quad.m
N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1)

x0 = zeros(n)
q0 = [1.;0.;0.;0.] # unit quaternion
x0[1:3] = [0.; 0.; 10.]
x0[4:7] = q0

xf = zero(x0)
xf[1:3] = [0.;60.; 10.]
xf[4:7] = q0;

# cost
Q = (1.0e-3)*Diagonal(I,n)
Q[4:7,4:7] = (1.0e-3)*Diagonal(I,4)
R = (1.0e-2)*Diagonal(I,m)
Qf = 1.0*Diagonal(I,n)
obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs

# Constraints
u_min = 0.
u_max = 50.
x_max = Inf*ones(n)
x_min = -Inf*ones(n)

x_max[1:3] = [25.0; Inf; 20]
x_min[1:3] = [-25.0; -Inf; 0.]
bnd_u = BoundConstraint(n,m,u_min=u_min, u_max=u_max)
bnd = BoundConstraint(n,m,u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

goal_con = goal_constraint(xf)

x_col = copy(x0)
x_col[1:3] = [0,9,3]  # a point in collision

cyl_obs = [(0., 10., 3.),
          (10., 30., 3.),
          (-13., 25., 2.),
          (5.,50.,4.)]

sph_obs = [( 0., 40., 5., 2.),
           (-5., 15., 3., 1.),
           (10., 20., 7., 2.)]
r_quad = 2.0

cylinder_con= let cylinders=cyl_obs, spheres=sph_obs, r_quad=r_quad
    function cylinder_con(c,x,u)
        for (i,cyl) in enumerate(cylinders)
            c[i] = circle_constraint(x,cyl[1],cyl[2],cyl[3]+r_quad)
        end
    end
end

sphere_con= let cylinders=cyl_obs, spheres=sph_obs, r_quad=r_quad
    function sphere_con(c,x,u)
        for (i,sphere) in enumerate(spheres)
            c[i] = TrajectoryOptimization.sphere_constraint(x,sphere[1],sphere[2],sphere[3],sphere[4]+r_quad)
        end
    end
end

cyl_con = Constraint{Inequality}(cylinder_con,n,m,length(cyl_obs),:cylinders)
sph_con = Constraint{Inequality}(  sphere_con,n,m,length(sph_obs),:spheres)


constraints = Constraints(N)
# constraints[1] += bnd_u
for k = 1:N-1
    constraints[k] += bnd + cyl_con + sph_con
end
constraints[N] += goal_con


# Problem
quad_obs = Problem(quad_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt)
u_hover = 0.5*9.81/4*ones(m)
U_hover = [u_hover for k = 1:N-1] # initial hovering control trajectory
initial_controls!(quad_obs,U_hover); # initialize problem with control
prob = copy(quad_obs)


############################################################################################
#                               STATIC PROBLEM                                             #
############################################################################################

# Generate model
quad_ = Dynamics.Quadrotor()
generate_jacobian(quad_)
rk3_gen(quad_)
generate_discrete_jacobian(quad_)


# Static Objective
x0 = SVector{n}(prob.x0)
xf = SVector{n}(prob.xf)
Q = (1.0e-3)*Diagonal(@SVector ones(n))  # use static diagonal to avoid allocations
R = (1.0e-2)*Diagonal(@SVector ones(m))
Qf = 1.0*Diagonal(@SVector ones(n))
obj = LQRObjective(Q,R,Qf,xf,N)

# Constraints
cyl_x = @SVector [cyl[1] for cyl in cyl_obs]
cyl_y = @SVector [cyl[2] for cyl in cyl_obs]
cyl_r = @SVector [cyl[3] + r_quad for cyl in cyl_obs]

sph_x = @SVector [sph[1] for sph in sph_obs]
sph_y = @SVector [sph[2] for sph in sph_obs]
sph_z = @SVector [sph[3] for sph in sph_obs]
sph_r = @SVector [sph[4] + r_quad for sph in sph_obs]

scyl_con = CircleConstraint(n,m,cyl_x,cyl_y,cyl_r)
ssph_con = SphereConstraint(n,m,sph_x,sph_y,sph_z,sph_r)
generate_jacobian(scyl_con)
generate_jacobian(ssph_con)
sbnd = StaticBoundConstraint(n,m, u_min=SVector{m}(u_min*ones(m)), u_max=SVector{m}(u_max*ones(m)),
    x_min=SVector{n}(x_min), x_max=SVector{n}(x_max))
sgoal = GoalConstraint(SVector{n}(xf))

con_cyl = KnotConstraint(scyl_con, 1:N-1)
con_sph = KnotConstraint(ssph_con, 1:N-1)
con_bnd = KnotConstraint(sbnd, 1:N-1)
con_goal = KnotConstraint(sgoal, N:N)
conSet = ConstraintSets([con_cyl, con_sph, con_bnd, con_goal], N)

# Test arrays
xs,us = SVector{n}(x_col), SVector{m}(u_hover)
x,u = Array(xs), Array(us)
zs = [xs;us]
z = [x;u]

# Static problem
z = KnotPoint(xs,us,dt)
Z = [z for k = 1:N]
Z[end] = KnotPoint(xs,m)
sprob = StaticProblem(quad_, obj, conSet, x0, xf, Z, deepcopy(Z), N, dt, prob.tf)

############################################################################################
#                               PROBLEM TESTS                                              #
############################################################################################

# Init solvers
silqr = StaticiLQRSolver(sprob)
ilqr = iLQRSolver(prob)

initial_controls!(prob, [u for k = 1:N-1])
initial_states!(prob, [x for k = 1:N])

X = prob.X
U = prob.U
dt_traj = get_dt_traj(prob)

# Test dynamics
@btime evaluate!($x, $prob.model, $x, $u, $dt)
@btime discrete_dynamics($quad_,$xs,$us,$dt) # 11x faster

# Dynamics jacobians
∇f = silqr.∇F
jacobian!(prob, ilqr)
discrete_jacobian!(∇f, quad_, Z)
silqr.∇F ≈ ilqr.∇F
@btime jacobian!($prob, $ilqr)
@btime discrete_jacobian!($∇f, $quad_, $Z) # ≈3x faster

# Objective
cost(prob) ≈ cost(obj, Z)
@btime cost($prob)
@btime cost($(sprob.obj), $Z) # 7.5x faster


# Cost expansion
E = silqr.Q
cost_expansion!(prob, ilqr)
cost_expansion(E, obj, Z)
all([E.xx[k] ≈ ilqr.Q[k].xx for k in eachindex(E.xx)])
all([E.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([E.x[k] ≈ ilqr.Q[k].x for k in eachindex(E.x)])
all([E.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])

@btime cost_expansion!($prob, $ilqr)
@btime cost_expansion($E, $obj, $Z)  # 7x faster


# Constraints
c_cyl = zeros(length(cyl_con))
c_sph = zeros(length(sph_con))
evaluate!(c_cyl, cyl_con, x, u)
evaluate!(c_sph, sph_con, x, u)
c_cyl ≈ evaluate(scyl_con, x, u)
c_sph ≈ evaluate(ssph_con, x, u)

@btime evaluate!($c_cyl, $cyl_con, $x, $u)
@btime evaluate($scyl_con, $xs, $us) # way faster (1000x)
@btime evaluate!($c_sph, $sph_con, $x, $u)
@btime evaluate($ssph_con, $xs, $us) # way faster (1000x)

∇c_cyl = zeros(length(cyl_con),n+m)
∇c_sph = zeros(length(sph_con),n+m)
jacobian!(∇c_cyl, cyl_con, x, u)
jacobian!(∇c_sph, sph_con, x, u)
jacobian(scyl_con, xs, us) ≈ ∇c_cyl
jacobian(ssph_con, xs, us) ≈ ∇c_sph

@btime jacobian!($∇c_cyl, $cyl_con, $x, $u)
@btime jacobian($scyl_con, $z)  # 10x faster
@btime jacobian!($∇c_sph, $sph_con, $x, $u)
@btime jacobian($ssph_con, $z) # 10x faster

b = zeros(length(bnd))
evaluate!(b, bnd, x, u)
evaluate(sbnd, xs, us) ≈ b
@btime evaluate!($b, $bnd, $x, $u)
@btime evaluate($sbnd, $xs, $us) # way faster

@btime evaluate($sgoal, $xs, $us)


# Augmented Lagrangian Solver
sopts = AugmentedLagrangianSolverOptions{Float64}()
sopts.opts_uncon = StaticiLQRSolverOptions{Float64}()
alsolver = AugmentedLagrangianSolver(prob)
salsolver = StaticALSolver(sprob, sopts)

# Augmented Lagrangian Objective
num_constraints(prob) ==  num_constraints(sprob)

alobj = AugmentedLagrangianObjective(prob, alsolver)
salobj = StaticALObjective(sprob.obj, sprob.constraints)

update_constraints!(alobj.C, alobj.constraints, X, U)
evaluate(salobj.constraints, Z)

cost!(salobj, Z)
sum(salobj.obj.J)
sum(salobj.obj.J) ≈ cost(alobj,X,U,dt_traj)

@btime cost($alobj, $X, $U, $dt_traj)
@btime cost!($salobj, $Z) # 24x faster

Q = ilqr.Q
E = silqr.Q
cost_expansion!(Q, alobj, X, U, dt_traj)
cost_expansion(E, salobj, Z)
all([E.xx[k] ≈ ilqr.Q[k].xx for k in eachindex(E.xx)])
all([E.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([E.x[k] ≈ ilqr.Q[k].x for k in eachindex(E.x)])
all([E.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])
@btime cost_expansion!($Q, $alobj, $X, $U, $dt_traj)
@btime cost_expansion($E, $salobj, $Z) # 11x faster

max_violation(alsolver)
max_violation!(conSet)
maximum(conSet.c_max) == max_violation(alsolver)
@btime max_violation($alsolver)
@btime max_violation!($conSet)  # 6.5x faster
