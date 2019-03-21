f = Dynamics.dubins_dynamics!
n,m = 3,2
model = Model(f,n,m)
x0 = [0; 0.; 0.];
xf = [1; 0.; 0.]; # (ie, swing up)
u0 = [1.;1.]
Q = 1e-3*Diagonal(I,n)
Qf = 100. *Diagonal(I,n)
R = 1e-3*Diagonal(I,m)
tf = 5.
costfun = LQRCost(Q,R,Qf,xf)
cc = copy(costfun)
# check_problem(p1)
model
P1 = Problem(model,costfun)
P2 = Problem(model,costfun,rand(n),[rand(m) for k = 1:10])
P3 = Problem(model,costfun,rand(n),[rand(n) for k = 1:11],[rand(m) for k = 1:10])

# Constraints
c(v,x,u) = begin v[1]  = x[1]^2 + x[2]^2 - 5; v[2:3] =  u - ones(2,1) end
p1 = 3
con = Constraint{Equality}(c,n,m,p1,:custom)
c2(v,x,u) = begin v[1] = sin(x[1]); v[2] = sin(x[3]) end
p2 = 2
con2 = Constraint{Inequality}(c2,∇c2,p2,:ineq)
# Bound constraint
x_max = [5,5,Inf]
x_min = [-10,-5,0]
u_max = 0
u_min = -10
p3 = 2(n+m)
bnd = bound_constraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max)
# Create Constraint Set
C = [con,con2,bnd]

P4 = update_problem(P3,dt=0.1)
add_constraints!(P3,con)
P4

bp = BackwardPassNew(P4)
bp2 = copy(bp)
reset!(bp)
bp
res = iLQRResults(P4)
res2 = copy(res)

ALres = ALResults(P4)
con
add_constraints!(P4,C_term)
T = Float64
N = 10
p = 5
n = 5
m = 2
p_N = 3
∇cval = [BlockArray(zeros(p,n+m),c_part2) for k = 1:N-1]
push!(∇cval,BlockMatrix(C_term,n,m))

∇cval

prob = P4
prob
n = prob.model.n; m = prob.model.m; N = prob.N
p = num_stage_constraints(prob.constraints)
p_N = num_terminal_constraints(prob.constraints)

c_stage = stage(prob.constraints)
c_term = terminal(prob.constraints)
c_part = create_partition(c_stage)
c_part2 = create_partition2(c_stage,n,m)
c_term_part2 = create_partition2(c_term,n,0)

C = [BlockArray(zeros(T,p),c_part) for k = 1:N-1]
C_prev = [BlockArray(zeros(T,p),c_part) for k = 1:N-1]
∇C = [BlockArray(zeros(T,p,n+m),c_part2) for k = 1:N-1]
λ = [BlockArray(zeros(T,p),c_part) for k = 1:N-1]
Iμ = [i != N ? Diagonal(ones(T,p)) : Diagonal(ones(T,p_N)) for i = 1:N]
active_set_ = [BlockArray(ones(Bool,p),c_part) for k = 1:N-1]
push!(C,BlockVector(T,c_term))
push!(C_prev,BlockVector(T,c_term))
push!(∇C,BlockMatrix(T,c_term,n,m))
push!(λ,BlockVector(T,c_term))
push!(active_set_,BlockVector(Bool,c_term))
