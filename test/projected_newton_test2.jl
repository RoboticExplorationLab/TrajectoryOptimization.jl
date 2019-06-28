using BlockArrays

# Set up Problem
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
N = 51
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=1.5)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min)
bnd_x = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf])
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4))
obs1 = planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
con = ProblemConstraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd  + obs1 + obs2
end
con[N] += goal
prob = Problem(rk4(model), Objective(costfun, N), constraints=con, tf=3)

# Solve with ALTRO
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions(opts_uncon=ilqr)
al.constraint_tolerance = 1e-2
al.constraint_tolerance_intermediate = 1e-1
al.verbose = true
solve!(prob, al)
plot()
plot_circle!(obs[1]...)
plot_circle!(obs[2]...)
plot_trajectory!(prob.X,markershape=:circle)
plot(prob.U)
max_violation(prob)

# Create PN Solver
solver = ProjectedNewtonSolver(prob)
NN = N*n + (N-1)*m
p = num_constraints(prob)
P = N*n + sum(p)

# Test functions
dynamics_constraints!(prob, solver)
update_constraints!(prob, solver)
solver.C[1]
active_set!(prob, solver)
@test all(solver.a.primals)
# @test all(solver.a.ν)
# @test all(solver.a.λ[end-n+1:end])
# @test !all(solver.a.λ)
dynamics_jacobian!(prob, solver)
@test solver.∇F[1].xx == solver.Y[1:n,1:n]
@test solver.∇F[2].xx == solver.Y[n .+ (1:n),1:n]

constraint_jacobian!(prob, solver)
@test solver.∇C[1] == solver.Y[2n .+ (1:p[1]), 1:n+m]

cost_expansion!(prob, solver)
# Y,y = Array(Y), Array(y)

# Test Constraint Violation
solver = ProjectedNewtonSolver(prob)
solver.opts.active_set_tolerance = 0.0
dynamics_constraints!(prob, solver)
update_constraints!(prob, solver)
dynamics_jacobian!(prob, solver)
constraint_jacobian!(prob, solver)
active_set!(prob, solver)
Y,y = active_constraints(prob, solver)
viol = calc_violations(solver)
@test maximum(maximum.(viol)) == norm(y,Inf)
@test norm(y,Inf) == max_violation(prob)
@test max_violation(solver) == max_violation(prob)

# Test Projection
solver = ProjectedNewtonSolver(prob)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.active_set_tolerance = 1e-3
update!(prob, solver)
Y,y = active_constraints(prob, solver)
projection!(prob, solver)
update!(prob, solver, solver.V)
max_violation(solver)
multiplier_projection!(prob, solver)

# Build KKT
V = solver.V
V0 = copy(V)
cost_expansion!(prob, solver)
J0 = cost(prob, V)
res0 = norm(residual(prob, solver))
viol0 = max_violation(solver)
δV = solveKKT(prob, solver)
V_ = line_search(prob, solver, δV)

# Test Newton Step
solver = ProjectedNewtonSolver(prob)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.verbose = false
V_ = newton_step!(prob, solver)
@btime newton_step!(prob, solver)
copyto!(solver.V.V, V_.V)
V_ = newton_step!(prob, solver)


update!(prob, solver)
Y,y = active_constraints(prob, solver)
@test length(y) == num_active_constraints(solver)
sum.(solver.active_set)
solver.active_set
inv(prob.obj[1].Q)*(N-1)
solver.∇C[1]

solver = ProjectedNewtonSolver(prob)
V = solver.V
update!(prob, solver)
Pa = num_active_constraints(solver)
Y,y = active_constraints(prob, solver)

Hinv = inv(Diagonal(solver.H))
Qinv = [begin
            off = (k-1)*(n+m) .+ (1:n);
            Diagonal(Hinv[off,off]);
        end for k = 1:N]
Rinv = [begin
            off = (k-1)*(n+m) .+ (n+1:n+m);
            Diagonal(Hinv[off,off]);
        end for k = 1:N-1]
A = [F.xx for F in solver.∇F[2:end]]
B = [F.xu for F in solver.∇F[2:end]]
C = [Array(F.x[a,:]) for (F,a) in zip(solver.∇C, solver.active_set)]
D = [Array(F.u[a,:]) for (F,a) in zip(solver.∇C, solver.active_set)]
g = solver.g

S0 = Y*Hinv*Y'
δλ = S0\(y-Y*Hinv*g)
δz = -Hinv*(g+Y'δλ)

δV0 = solveKKT(prob, solver)
δV0 = PrimalDual(copy(δV0), n, m, N, length(solver.y))
@test primals(δV0) == δV[1:NN]
@test primals(δV0) ≈ δz
@test duals(δV0)[solver.a.duals] ≈ δλ

δV1 = solveKKT_Shur(prob, solver, Hinv)
δV1 ≈ δV0

δV2 = solveKKT_chol(prob, solver, Qinv, Rinv, A, B, C, D)
δV2 ≈ δV0

@btime solveKKT($prob, $solver)
@btime solveKKT_Shur($prob, $solver, $Hinv);
@btime solveKKT_chol($prob, $solver, $Qinv, $Rinv, $A, $B, $C, $D);

S,L = buildShurCompliment(prob, solver)
@test S ≈ S0
@test L*L' ≈ S0


solver.parts.primals

L0 = cholesky(Array(S)).L
Array{Int}(L .≈ L0)
@test L ≈ L0
@test L*L' ≈ S


y_part = ones(Int,2,N-1)*n
p = sum.(solver.active_set)
p = num_constraints(prob)
y_part[2,:] = p[1:end-1]
y_part = vec(y_part)
insert!(y_part,1,3)
push!(y_part, p[N])
c_blocks = push!(collect(3:2:length(y_part)),length(y_part))

z_part = repeat([n,m],N-1)
push!(z_part, n)
NN = sum(z_part)
P = sum(y_part)
part_a = (primals=1:NN, duals=NN+1:NN+P, ν=NN .+ (1:N*n), λ=NN + N*n .+ (1:sum(p)))

Y = PseudoBlockArray(solver.Y, y_part, z_part)
y = PseudoBlockArray(zeros(sum(y_part)),y_part)
a = PseudoBlockArray(ones(Bool,NN+P), [NN; y_part])

∇C = [view(Y,Block(i,j)) for (i,j) in zip(c_blocks, 1:2:2N)]
C = [view(y,Block(i)) for i in c_blocks]
act_set = [view(a,Block(i+1)) for i in c_blocks]
a_ = PartedVector(a,part_a)

println(typeof(a_))
∇C isa Vector{SubArray{T,2,PseudoBlockArray{T,2,SparseArrays.SparseMatrixCSC{T,Int64},BlockArrays.BlockSizes{2,NTuple{2,Vector{Int}}}},NTuple{2,BlockArrays.BlockSlice{Block{1,Int64}}},false}} where T

a = ones(Bool,)

view(Y, Block(1,1))
Y[Block(1,1)]

lcum = cumsum(len)
y_part = [lcum[k]:lcum[k+1]-1 for k = 1:length(lcum)-1]



T = Float64
n,m,N = size(prob)
X_ = [zeros(T,n) for k = 1:N-1] # midpoints

NN = N*n + (N-1)*m
p = num_constraints(prob)
pcum = insert!(cumsum(p),1,0)
P = sum(p) + N*n

V = PrimalDual(prob)

part_f = create_partition2(prob.model)
constraints = prob.constraints

# Block Array partitions
y_part = ones(Int,2,N-1)*n
y_part[2,:] = p[1:end-1]
y_part = vec(y_part)
insert!(y_part,1,3)
push!(y_part, p[N])
c_blocks = push!(collect(3:2:length(y_part)),length(y_part))

z_part = repeat([n+m],N-1)
push!(z_part, n)
d_blocks = insert!(collect(2:2:length(y_part)-1),1,1)


# Build Blocks
H = spzeros(NN,NN)
g = zeros(NN)
Y = PseudoBlockArray(spzeros(P,NN), y_part, z_part)
y = PseudoBlockArray(zeros(sum(y_part)),y_part)
a = PseudoBlockArray(ones(Bool,NN+P), [NN; y_part])


# Build views
fVal = [view(y,Block(i)) for i in d_blocks]
∇F = [PartedMatrix(zeros(n,n+m+1), part_f) for k = 1:N]

C = [PartedArray(view(y, Block(c_blocks[k])),
     create_partition(constraints[k], k==N ? :terminal : :stage)) for k = 1:N]
∇C = [PartedArray(view(Y, Block(c_blocks[k],k)),
     create_partition2(constraints[k], n,m, k==N ? :terminal : :stage)) for k = 1:N]
println(typeof(∇C))

c_inds = [C[k].A.indices[1].indices for k = 1:N]
d_inds = [fVal[k].indices[1].indices for k = 1:N]
part_a = (primals=1:NN, duals=NN+1:NN+P)
active_set = [view(a,Block(i+1)) for i in c_blocks]
a = PartedVector(a, part_a)

len
issymmetric()

num_constraints(prob)
sum(solver.a.duals)
length(solver.a)
length(solver.a.ν)

using Juno
using Profile
using InteractiveUtils
Profile.init(delay=1e-4)
@profiler newton_step!(prob, solver)


plot()
plot_circle!(obs[1]...)
plot_circle!(obs[2]...)
plot_trajectory!(prob.X,markershape=:circle)
plot_trajectory!(V_.X,markershape=:circle)
plot(prob.U, color=:blue)
plot!(V_.U, color=:red)
