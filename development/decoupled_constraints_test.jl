# Tests
n,m = 3,2

model = Model(Dynamics.dubins_dynamics!,n,m)

# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = (1e-2)*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.01

function c(x,u)
    x[1:n] = 10*ones(n)
end
u_max = 3
u_min = -3
x_max = 10
x_min = -10

x0 = rand(n)
u0 = rand(m)
obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_con = ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,x_max=x_max,x_min=x_min)
@test obj_con.pIs == 2*n
@test obj_con.pIc == 2*m
@test obj_con.pEs == 0
@test obj_con.pEsN == n
@test obj_con.pEc == 0

# default state and control constraints
z = zeros(obj_con.pIs)
Z = zeros(obj_con.pIs,n)
gs = generate_state_inequality_constraints(obj_con)
gsx = generate_state_inequality_constraint_jacobian(obj_con)
gs(z,x0)
gsx(Z,x0)
@test isapprox(z,[x0 .- x_max; x_min .- x0])
@test isapprox(Z,[Matrix(I,n,n);-Matrix(I,n,n)])

z = zeros(obj_con.pEs)
Z = zeros(obj_con.pEs,n)
z0 = copy(z)
Z0 = copy(Z)
hs = generate_state_equality_constraints(obj_con)
hsx = generate_state_equality_constraint_jacobian(obj_con)
hs(z,x0)
hsx(Z,x0)
@test isapprox(z,z0)
@test isapprox(Z,Z0)

z = zeros(obj_con.pIc)
Z = zeros(obj_con.pIc,m)
gc = generate_control_inequality_constraints(obj_con)
gcu = generate_control_inequality_constraint_jacobian(obj_con)
gc(z,u0)
gcu(Z,u0)
@test isapprox(z,[u0 .- u_max; u_min .- u0])
@test isapprox(Z,[Matrix(I,m,m);-Matrix(I,m,m)])

z = zeros(obj_con.pEc)
Z = zeros(obj_con.pEc,m)
z0 = copy(z)
Z0 = copy(Z)

hc = generate_control_equality_constraints(obj_con)
hcu = generate_control_equality_constraint_jacobian(obj_con)
hc(z,u0)
hcu(Z,u0)
@test isapprox(z,z0)
@test isapprox(Z,Z0)

z = zeros(obj_con.pEsN)
Z = zeros(obj_con.pEsN,n)
hsN = generate_state_terminal_constraints(obj_con)
hsNx = generate_state_terminal_constraint_jacobian(obj_con)
hsN(z,x0)
hsNx(Z,x0)
@test isapprox(z,x0-xf)
@test isapprox(Z,Matrix(I,n,n))

# custom constraints
obj_con = ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,x_max=x_max,x_min=x_min,gs_custom=c,gc_custom=c,hs_custom=c,hc_custom=c)
@test obj_con.pIs == 2*n+n
@test obj_con.pIc == 2*m+n
@test obj_con.pEs == 0+n
@test obj_con.pEsN == n+n
@test obj_con.pEc == 0+n

z = zeros(obj_con.pIs)
Z = zeros(obj_con.pIs,n)
gs = generate_state_inequality_constraints(obj_con)
gsx = generate_state_inequality_constraint_jacobian(obj_con)

gs(z,x0)
gsx(Z,x0)
@test isapprox(z,[x0 .- x_max; x_min .- x0; 10*ones(n)])
@test isapprox(Z,[Matrix(I,n,n); -Matrix(I,n,n); zeros(n,n)])

z = zeros(obj_con.pEs)
z0 = copy(z)
hs = generate_state_equality_constraints(obj_con)
hs(z,x0)
@test isapprox(z,10*ones(n))

Z = zeros(obj_con.pEs,n)
Z0 = copy(Z)
hsx = generate_state_equality_constraint_jacobian(obj_con)
hsx(Z,x0)
@test isapprox(Z,zeros(n,n))

z = zeros(obj_con.pIc)
gc = generate_control_inequality_constraints(obj_con)
gc(z,u0)
@test isapprox(z,[u0 .- u_max; u_min .- u0; 10*ones(n)])

Z = zeros(obj_con.pIc,m)
gcu = generate_control_inequality_constraint_jacobian(obj_con)
gcu(Z,u0)
@test isapprox(Z,[Matrix(I,m,m);-Matrix(I,m,m);zeros(n,m)])

z = zeros(obj_con.pEc)
z0 = copy(z)
hc = generate_control_equality_constraints(obj_con)
hc(z,u0)
@test isapprox(z,10*ones(n))

Z = zeros(obj_con.pEc,m)
Z0 = copy(Z)
hcu = generate_control_equality_constraint_jacobian(obj_con)
hcu(Z,u0)
@test isapprox(Z,zeros(n,m))

z = zeros(obj_con.pEsN)
hsN = generate_state_terminal_constraints(obj_con)
hsN(z,x0)
@test isapprox(z,[x0-xf;10*ones(n)])

Z = zeros(obj_con.pEsN,n)
hsNx = generate_state_terminal_constraint_jacobian(obj_con)
hsNx(Z,x0)
@test isapprox(Z,[Matrix(I,n,n); zeros(n,n)])

# minimum time
obj_con_min_time = update_objective(obj_con,tf=0.0)
@test obj_con.pIs == 2*n+n
@test obj_con.pIc == 2*m+n
@test obj_con.pEs == 0+n
@test obj_con.pEsN == n+n
@test obj_con.pEc == 0+n

z = zeros(obj_con_min_time.pIc+2)
gc = generate_control_inequality_constraints(obj_con_min_time; max_dt=1.0,min_dt=1.0e-3)
gc(z,[u0; 0.1])
@test isapprox(z,[u0 .- u_max; 0.1-sqrt(1.0); u_min .- u0;sqrt(1e-3) - 0.1; 10*ones(n)])

Z = zeros(obj_con_min_time.pIc+2,m+1)
gcu = generate_control_inequality_constraint_jacobian(obj_con_min_time)
gcu(Z,[u0; 0.1])
@test isapprox(Z,[Matrix(I,m+1,m+1);-Matrix(I,m+1,m+1);zeros(n,m+1)])

z = zeros(obj_con_min_time.pEc+1)
z0 = copy(z)
hc = generate_control_equality_constraints(obj_con_min_time)
hc(z,[u0;0.1])
@test isapprox(z,[0;10*ones(n)])

Z = zeros(obj_con_min_time.pEc+1,m+1)
Z0 = copy(Z)
hcu = generate_control_equality_constraint_jacobian(obj_con_min_time)
hcu(Z,[u0;0.1])
Z0[1,m+1] = 1.
@test isapprox(Z,Z0)

# minimum_time + infeasible
ui = [1;2;3]
U = [u0;0.1;ui]
z = zeros(obj_con_min_time.pIc+2)
gc = generate_control_inequality_constraints(obj_con_min_time)
gc(z,U)
@test isapprox(z,[u0 .- u_max; 0.1-sqrt(1.0); u_min .- u0;sqrt(1e-3) - 0.1; 10*ones(n)])

Z = zeros(obj_con_min_time.pIc+2,m+1+n)
gcu = generate_control_inequality_constraint_jacobian(obj_con_min_time)
gcu(Z,U)
@test isapprox(Z,[Matrix(I,m+1,m+1+n);-Matrix(I,m+1,m+1+n);zeros(n,m+1+n)])

z = zeros(obj_con_min_time.pEc+1+n)
z0 = copy(z)
hc = generate_control_equality_constraints(obj_con_min_time)
hc(z,U)
@test isapprox(z,[1;2;3;0;10*ones(n)])

Z = zeros(obj_con_min_time.pEc+1+n,m+1+n)
Z0 = copy(Z)
hcu = generate_control_equality_constraint_jacobian(obj_con_min_time)
hcu(Z,U)
@test isapprox(Z,[Matrix(I,n,m+1+n); 0 0 1 0 0 0; zeros(n,m+1+n)])

##
N = 25
solver = Solver(model,obj_con,N=N)

pIs, pIc, pEs, pEsN, pEc = get_num_constraints(solver)
get_initial_dt(solver)
get_num_controls(solver)

results = ConstrainedVectorResults(n,m,N,pIs,pIc,pEs,pEsN,pEc)
for k = 1:N
    solver.gs(results.gs[k],k*ones(n))
end
to_array(results.gs)
cost(solver,results)
update_constraints!(results,solver)

max_state_inequality = maximum(abs.(to_array(results.gs)[convert.(Bool,to_array(results.gs_active_set))]))

convert.(Bool,to_array(results.gs_active_set))
to_array(results.gs)

max_violation(results)
U = ones(m,N)

results, stats1 = solve(solver,U)
max_violation(results1)
results1.X[end]

state_inequality = to_array(results.gs)[convert.(Bool,to_array(results.gs_active_set))]
control_inequality = to_array(results.gc)[convert.(Bool,to_array(results.gc_active_set))]
state_equality1 = to_array(results.hs)
control_equality = to_array(results.hc)
