import TrajectoryOptimization.TrajectoryVariable
n,m,N = 3,2,501
X = TrajectoryVariable(N,n)
X[1] == zeros(n)
size(X) == (n,N)
to_array(X.x)

K = TrajectoryVariable(N,n,m)
K[1] == zeros(n,m)
size(K) == (n,m,N)

X = TrajectoryVariable(ones(n,N))
X *2


K = TVar1(N,(n,m),size_N=(1,1))
TVar1([Diagonal(I,4) for k = 1:N])


X = rand(n,N)
U = ones(m,N)
Xv = to_dvecs(X)
Uv = to_dvecs(U)
Xt = TVar1(Xv)
Ut = TVar1(Uv)
TX = TVar(X)
TU = TVar(U)


model,obj = Dynamics.dubinscar
solver = Solver(model,obj,N=N)
U = ones(m,N)

solver.opts.restype = Matrix
b1 = @benchmark res,stat = solve(solver,U)
norm(res.X[N] - obj.xf)

solver.opts.restype = TVar1
b_tv1 = @benchmark res_tv1,stat_tv1 = solve(solver,U)
norm(res_tv1.X[N] - obj.xf)

solver.opts.restype = TVar
b_tv = @benchmark res_tv,stat_tv = solve(solver,U)
norm(res_tv.X[N] - obj.xf)

judge(median(b_tv1),median(b1))
judge(median(b_tv),median(b1))


model,obj = Dynamics.dubinscar
n,m,N = model.n,model.m,51
x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]
obj_con = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)
solver = Solver(model, obj_con, N=N)
U = ones(m,N-1)

solver.opts.restype = Matrix
res,stat = solve(solver,U)
solver.opts.restype = TrajectoryVariable
res_t,stat_t = solve(solver,U)
res.X[N] == res_t.X[N]

solver.opts.restype = Matrix
b1 = @benchmark solve(solver,U)
solver.opts.restype = TrajectoryVariable
b_tv1 = @benchmark solve(solver,U)
judge(median(b_tv1),median(b1))

cost(solver,X,U)
cost(solver,Xv,Uv)
cost(solver,Xt,Ut)
cost(solver,TX,TU)

b  = @benchmark(cost($solver,$X,$U))
bv = @benchmark(cost($solver,$Xv,$Ut))
bt = @benchmark(cost($solver,$Xt,$Uv))
btv = @benchmark(cost($solver,$Xt,$Uv))

judge(median(bv),median(b))
judge(median(bt),median(bv))
judge(median(btv),median(bv))


X2 = [view(X,1:n,k) for k = 1:N]
X2[1] .= zeros(n)

X[:,1]
X[:,1] = ones(n)
X2[1]

X = rand(n,N)
@btime copyto!($TX,$X)
@btime copy2!($TX,$X)
@btime copy3!($TX,$X)
@btime copyto!($Xt,$X)

@btime copyto!($TX,$TX)
@btime copyto!($Xt,$Xt)

b = rand(n)
@btime $TX[5] = $b
@btime $Xt[5] = $b

@btime $TX[5]
@btime $Xt[5]

n,m,N = 3,2,501
UnconstrainedVectorResults(n,m,N)
UnconstrainedVectorResults(n,m,N,TrajectoryVariable) isa UnconstrainedVectorResults{TV,TM} where {TV <: TrajectoryVariable, TM <: TrajectoryVariable}
ConstrainedVectorResults(n,m,N,10,4,Matrix)
ConstrainedVectorResults(n,m,N,10,4)


T = TVar1
T(N,n)

TVar(N,n)
