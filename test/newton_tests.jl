
solver = Solver(model,obj,N=21)
n,m,N = get_sizes(solver)
p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
obj.cost.Q .= Diagonal(I,n)
obj.cost.R .= Diagonal(I,m)*2
obj.cost.H .= ones(m,n)

Nx = N*n
Nu = (N-1)*m
Nz = Nx+Nu
X = rand(n,N)
U = rand(m,N-1)
Z = packZ(X,U)

V = [Z; rand(21)]
P = PrimalVars(V,n,m,N)
@test P.Z == Z
@test P.X == X
@test P.U == U
@test get_sizes(P) == (n,m,N)
@test P[3] == [X[:,3]; U[:,3]]

Nx = N*n
Nu = (N-1)*m
Nz = Nx + Nu
Nh = (N-1)*pE + pE_N
Ng = (N-1)*pI + pI_N
NN = 2Nx + Nu + Nh + Ng

names = (:z,:ν,:λ,:μ)
ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng],names)
ind2 = TrajectoryOptimization.create_partition2([Nz,Nx,Nh,Ng],names)

ind_x, ind_u = get_primal_inds(n,m,N)
ind_z = TrajectoryOptimization.create_partition([Nx,Nu],(:x,:u))
ind_h = create_partition([(N-1)*pE,pE_N],(:k,:N))
ind_g = create_partition([(N-1)*pI,pI_N],(:k,:N))
ind_h = (k=ind_h.k, N=ind_h.N, k_mat=reshape(ind_h.k,pE,N-1))
ind_g = (k=ind_g.k, N=ind_g.N, k_mat=reshape(ind_g.k,pI,N-1))

CE = zeros(Nh)
CE[ind_h.k_mat]
CI = zeros(Ng)
CI[ind_g.k_mat]

Z = packZ(X,U)
mycost, c, grad_J, hess_J, jacob_c = gen_usrfun_newton(solver)
mycost(Z)
c.d(Z)
c.I(Z)
c.E(Z)
@test grad_J(Z) == ForwardDiff.gradient(mycost,Z)
@test jacob_c.E(Z) == ForwardDiff.jacobian(c.E,Z)
@testArray(jacob_c.I(Z)) == ForwardDiff.jacobian(c.I,Z)
@test Array(jacob_c.d(Z)) == ForwardDiff.jacobian(c.d,Z)




model,obj = Dynamics.dubinscar_parallelpark
solver = Solver(model,obj,N=21)
n,m,N = get_sizes(solver)
p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
dt = solver.dt
U0 = ones(m,N-1)

X = rand(n,N)
U = rand(m,N-1)
Z = PrimalVars(X,U)
Nx,Nu,Nh,Ng,NN = get_batch_sizes(solver)
Nz = Nx + Nu
part = create_partition([Nz,Nx,Nh,Ng],(:z,:ν,:λ,:μ))
ν = rand(Nx)
λ = rand(Nh)
μ = rand(Ng)
V = NewtonVars(Z,ν,λ,μ)
V1 = copy(V)
V1.V[1] = 100
V1.Z.Z[1] == 100
V1.V[Nz+1] = 200
V1.ν[1] == 200

res,stats = solve(solver,U0)
plot(res.X)

# Convert to PrimalVar
Z = PrimalVars(res)

# Get Constants
n,m,N = get_sizes(solver)
Nz = length(Z.Z)   # Number of decision variables
Nx = N*n           # Number of collocation constraints
nH = 0             # Number of entries in cost hessian
