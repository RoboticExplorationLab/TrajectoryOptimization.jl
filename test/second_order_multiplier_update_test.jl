# Confirm that foh alt gives same results as foh

### Solver options ###
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = false
# opts.c1 = 1e-4
opts.c2 = 5.0
opts.cost_intermediate_tolerance = 1e-5
opts.constraint_tolerance = 1e-5
opts.cost_tolerance = 1e-5
opts.iterations_outerloop = 50
opts.iterations = 500
# opts.iterations_linesearch = 50
opts.τ = 0.25
opts.outer_loop_update = :individual
######################

### Set up model, objective, solver ###
# Model, objective (unconstrained)
model, obj_uncon = TrajectoryOptimization.Dynamics.pendulum!

# -Constraints
u_min = -2
u_max = 2
x_min = [-20;-20]
x_max = [20; 20]

# # -Constrained objective
# function cE(cdot,x,u)
#     cdot[1] = x[1] - pi
# end

obj_con = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)#, cE=cE)

# Solver
opts.use_static = false
solver = Solver(model,obj_con,integration=:rk3_foh,dt=dt,opts=opts)

# -Initial state and control trajectories
U = 10.0*rand(solver.model.m,solver.N)

results = ConstrainedVectorResults(solver.model.n,solver.model.m,solver.obj.p,solver.N)

copyto!(results.U,U)
rollout!(results,solver)
update_constraints!(results,solver)
calculate_jacobians!(results,solver)

results


# zoh version
# # Build the Hessian of the Lagrangian, stacked: constraints, Jacobians, multipliers
# n = solver.model.n
# m = solver.model.m
# p = solver.obj.p
# pI = solver.obj.pI
# N = solver.N
# Q = solver.obj.Q
# R = solver.obj.R
# Qf = solver.obj.Qf
# dt = solver.dt
#
# # initialize
# L = zeros((n+m)*(N-1)+n,(n+m)*(N-1)+n)
# c = zeros(p*(N-1)+n)
# cz = zeros(p*(N-1)+n,(n+m)*(N-1)+n)
# λ = zeros(p*(N-1)+n)
# μ = zeros(p*(N-1)+n,p*(N-1)+n)
#
# # get inequality indices
# idx_inequality = []
# tmp = [i for i = 1:pI]
#
# for i = 1:N-1
#     idx_inequality = cat(idx_inequality,tmp .+ (i-1)*p,dims=(1,1))
# end
# idx_inequality
#
# # assemble constraints
# for i = 1:N-1
#     c[(i-1)*p+1:(i-1)*p+p] = results.C[i]
#     cz[(i-1)*p+1:(i-1)*p+p,(i-1)*(n+m)+1:(i-1)*(n+m)+(n+m)] = [results.Cx[i] results.Cu[i]]
# end
# c[(N-1)*p+1:(N-1)*p+n] = results.CN
# cz[(N-1)*p+1:(N-1)*p+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = results.Cx_N
#
# # assemble lagrange multipliers
# for i = 1:N-1
#     λ[(i-1)*p+1:(i-1)*p+p] = results.LAMBDA[i]
# end
# λ[(N-1)*p+1:(N-1)*p+n] = results.λN
#
# # assemble penalty matrix
# for i = 1:N-1
#     μ[(i-1)*p+1:(i-1)*p+p,(i-1)*p+1:(i-1)*p+p] = results.Iμ[i]
# end
# μ[(N-1)*p+1:(N-1)*p+n,(N-1)*p+1:(N-1)*p+n] = results.IμN
#
# if !solver.opts.λ_second_order_update
#     # first order multiplier update
#     λ .= λ + μ*c
#     λ[idx_inequality] = max.(0.0,λ[idx_inequality])
# end
# # second order multiplier update
# constraint_status = ones(Bool,p*(N-1)+n)
#
# for i in idx_inequality
#     if c[i] <= 0.0 #&& λ[i] == 0.0
#         constraint_status[i] = false
#     end
# end
#
# idx_active = findall(x->x==true,constraint_status)
#
# # build Hessian
# for i = 1:N-1
#     L[(i-1)*(n+m)+1:(i-1)*(n+m)+(n+m),(i-1)*(n+m)+1:(i-1)*(n+m)+(n+m)] = [dt*Q zeros(n,m); zeros(m,n) dt*R]
# end
# L[(N-1)*(n+m)+1:(N-1)*(n+m)+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = Qf
#
# L .+= cz[idx_active,:]'*μ[idx_active,idx_active]*cz[idx_active,:]
#
# B = cz[idx_active,:]*inv(L)*cz[idx_active,:]'
#
# if solver.opts.λ_second_order_update
#     λ[idx_active] = λ[idx_active] + inv(B)*c[idx_active]
#     λ[idx_inequality] = max.(0.0,λ[idx_inequality])
# end
#
# # store results
# for i = 1:N-1
#     results.LAMBDA[i] = λ[(i-1)*p+1:(i-1)*p+p]
# end
# results.λN .= λ[(N-1)*p+1:(N-1)*p+n]

# foh case
# Build the Hessian of the Lagrangian, stacked: constraints, Jacobians, multipliers
n = solver.model.n
m = solver.model.m
q = n+m
p = solver.obj.p
pI = solver.obj.pI
N = solver.N
Q = solver.obj.Q
R = solver.obj.R
Qf = solver.obj.Qf
dt = solver.dt
if solver.model.m != length(results.U[1])
    m += n
    p += n
end

# initialize
L = zeros((n+m)*N,(n+m)*N)
c = zeros(p*N+n)
cz = zeros(p*N+n,(n+m)*N)
λ = zeros(p*N+n)
μ = zeros(p*N+n,p*N+n)

# get inequality indices
idx_inequality = []
tmp = [i for i = 1:pI]

for k = 1:N
    idx_inequality = cat(idx_inequality,tmp .+ (k-1)*p,dims=(1,1))
end
idx_inequality

# assemble constraints
for k = 1:N
    c[(k-1)*p+1:(k-1)*p+p] = results.C[k]
    cz[(k-1)*p+1:(k-1)*p+p,(k-1)*(n+m)+1:(k-1)*(n+m)+(n+m)] = [results.Cx[k] results.Cu[k]]
end
c[N*p+1:N*p+n] = results.CN
cz[N*p+1:N*p+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = results.Cx_N

# assemble lagrange multipliers
for k = 1:N
    λ[(k-1)*p+1:(k-1)*p+p] = results.LAMBDA[k]
end
λ[N*p+1:N*p+n] = results.λN

# assemble penalty matrix
for k = 1:N
    μ[(k-1)*p+1:(k-1)*p+p,(k-1)*p+1:(k-1)*p+p] = results.Iμ[k]
end
μ[N*p+1:N*p+n,N*p+1:N*p+n] = results.IμN

if !solver.opts.λ_second_order_update
    # first order multiplier update
    λ .= λ + μ*c
    λ[idx_inequality] = max.(0.0,λ[idx_inequality])
end
# second order multiplier update
constraint_status = ones(Bool,N*p+n)

for i in idx_inequality
    if c[i] <= 0.0 #&& λ[i] == 0.0
        constraint_status[i] = false
    end
end

idx_active = findall(x->x==true,constraint_status)

# build Hessian
for k = 1:N-1
    # Unpack Jacobians, ̇x
    Ac1, Bc1 = results.Ac[k], results.Bc[k]
    Ac2, Bc2 = results.Ac[k+1], results.Bc[k+1]
    Ad, Bd, Cd = results.fx[k], results.fu[k], results.fv[k]

    xm = results.xmid[k]
    um = (U[k] + U[k+1])/2.0

    lxx = dt/6*Q + 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 + dt/8*Ac1)
    luu = dt/6*R + 4*dt/6*((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)
    lyy = dt/6*Q + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(I/2 - dt/8*Ac2)
    lvv = dt/6*R + 4*dt/6*((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)

    lxu = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(dt/8*Bc1)
    lxy = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 - dt/8*Ac2)
    lxv = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(-dt/8*Bc2)
    luy = 4*dt/6*(dt/8*Bc1)'*Q*(I/2 - dt/8*Ac2)
    luv = 4*dt/6*((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)
    lyv = 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(-dt/8*Bc2)

    L[(k-1)*(n+m)+1:(k-1)*(n+m)+2*(n+m),(k-1)*(n+m)+1:(k-1)*(n+m)+2*(n+m)] = [lxx lxu lxy lxv; lxu' luu luy luv; lxy' luy' lyy lyv; lxv' luv' lyv' lvv]
end
L[(N-1)*(n+m)+1:(N-1)*(n+m)+n,(N-1)*(n+m)+1:(N-1)*(n+m)+n] = Qf

L .+= cz[idx_active,:]'*μ[idx_active,idx_active]*cz[idx_active,:]

B = cz[idx_active,:]*inv(L)*cz[idx_active,:]'

if solver.opts.λ_second_order_update
    λ[idx_active] = λ[idx_active] + inv(B)*c[idx_active]
    λ[idx_inequality] = max.(0.0,λ[idx_inequality])
end

# store results
for k = 1:N
    results.LAMBDA[k] = λ[(k-1)*p+1:(k-1)*p+p]
end
results.λN .= λ[N*p+1:N*p+n]
