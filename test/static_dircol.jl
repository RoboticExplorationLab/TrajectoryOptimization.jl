
prob = copy(Problems.cartpole)
prob = update_problem(prob, model=Dynamics.cartpole)
bnds = remove_bounds!(prob)
z_U, z_L, g_U, g_L = get_bounds(prob, bnds)


sprob = copy(Problems.cartpole_static)
n,m,N = size(sprob)
xlim = @SVector [10,Inf,Inf,Inf]
ulim = @SVector [2]
bnd1 = StaticBoundConstraint(n,m, u_min=-ulim/2, u_max=ulim/2)
bnd2 = StaticBoundConstraint(n,m, x_min=-xlim, x_max=xlim, u_min=-ulim, u_max=ulim)
bnd3 = StaticBoundConstraint(n,m, x_min=-xlim, x_max=xlim)
con1 = ConstraintVals(bnd1,1:50)
con2 = ConstraintVals(bnd2,70:101)
con3 = ConstraintVals(bnd3,45:70)
con4 = ConstraintVals(GoalConstraint(sprob.xf), N:N)

# Test bounds
conSet = ConstraintSets([con1,con2,con3,con4],N)
@test_throws AssertionError get_bounds(conSet)

conSet = ConstraintSets([con1,con2,con4],N)
zU,zL,gU,gL = get_bounds(conSet)
@test (gU,gL) == get_bounds(conSet)[3:4]
@test gU == zero(gU)


# Test Copying
Z = sprob.Z
P = StaticPrimals(n,m,N)
NN = (n+m)*N
V = rand(NN)

x = state(Z[1])
u = control(Z[1])

copyto!(Z, V, P.xinds, P.uinds)
@test Z[1].z[1:n] == V[1:n]
@test Z[N].z == V[end-n-m+1:end]

cost(sprob)
