using ForwardDiff
using LinearAlgebra
using Plots
using Formatting
import TrajectoryOptimization: get_num_terminal_constraints, generate_constraint_functions, parse_snopt_summary
import TrajectoryOptimization: newton_ipopt, newton_snopt, PrimalVars

model,obj = Dynamics.pendulum
obj.cost.Q .= Diagonal(I,2)*1
obj.cost.R .= Diagonal(I,1)*1
obj.cost.Qf .= Diagonal(I,2)*100
obj = ConstrainedObjective(obj,u_max=3)
obj = update_objective(obj,tf=5)

model,obj = Dynamics.dubinscar
obj.cost.Q .= Diagonal(I,3)*1e-1
obj.cost.R .= Diagonal(I,2)*1e-1
obj.cost.Qf .= Diagonal(I,3)*100
obj_c = ConstrainedObjective(obj,u_max=0.75,u_min=-0.75,x_min=[-0.5;-0.1;-Inf],x_max=[0.5,1.1,Inf])

model,obj = Dynamics.quadrotor
obj_c = ConstrainedObjective(obj,u_max=4,u_min=0)

model,obj = Dynamics.double_integrator
obj_c = ConstrainedObjective(obj)

# obj_c = ConstrainedObjective(obj,u_min=-0.3,u_max=0.4)
solver = Solver(model,obj_c,N=25)
n,m,N = get_sizes(solver)
p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
c_function!, c_jacobian!, c_labels, cI!, cE! = generate_constraint_functions(solver.obj)
dt = solver.dt
U0 = ones(m,N-1)
solver.opts.verbose = true
solver.opts.penalty_initial = 0.01
solver.opts.cost_tolerance_intermediate = 1e-2
solver.opts.cost_tolerance = 1e-2
solver.opts.constraint_tolerance = 1e-4
res,stats = solve(solver,U0)

Zopt,optval,stats = newton_snopt(solver,res,1e-8)
stats["major iterations"]
Zopt = PrimalVars(Zopt,n,m,N)
res_snopt = ConstrainedVectorResults(solver,Zopt.X,Zopt.U)
backwardpass!(res_snopt,solver)
rollout!(res_snopt,solver,0.0)
max_violation(res_snopt)

Zopt = newton_ipopt(solver,res)
Zopt = PrimalVars(Zopt,n,m,N)
res_ipopt = ConstrainedVectorResults(solver,Zopt.X,Zopt.U)
backwardpass!(res_ipopt,solver)
rollout!(res_ipopt,solver,0.0)
max_violation(res_ipopt)

plot()
plot_trajectory!(res)
plot(res.X)
plot(res.U)

x_L = isfinite.(solver.obj.x_min)
x_U = isfinite.(solver.obj.x_max)
u_L = isfinite.(solver.obj.u_min)
u_U = isfinite.(solver.obj.u_max)
active = [u_U;u_L;x_U;x_L]
lambda_ = [-1,-2,1,2,-3,-4,3,4]
labels = TrajectoryOptimization.get_constraint_labels(solver)
inds = (x_U=labels .== "state (upper bound)",
        x_L=labels .== "state (lower bound)",
        u_U=labels .== "control (upper bound)",
        u_L=labels .== "control (lower bound)",
        x=1:n, u=n.+(1:m))

lambda = zeros(n+m,2)




function mycost(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx .+ (1:Nu)],m,N-1)
    cost(solver,X,U)

end

function lagrangian(V)
    nu = V[Nz .+ (1:Nx)]
    λ = V[(Nz + Nx) .+ (1:Nh)]
    μ = V[(Nz + Nx + Nh) .+ (1:Ng)]
    Z = V[1:Nz]
    J = mycost(Z) + nu'dynamics(Z) + λ'cE(Z) + μ'cI(Z)
end

function al_lagrangian(V,ρ)
    eps = 1e-4
    Z = V[1:Nz]
    d = dynamics(Z)
    h = cE(Z)
    g = cI(Z)
    a = g .> eps
    lagrangian(V) + ρ/2*sqrt(d'd + h'h + g[a]'g[a])
end

function cE(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

    C = zeros(eltype(Z),pE,N-1)
    for k = 1:N-1
        cE!(view(C,1:pE,k),X[:,k],U[:,k])
    end
    CN = zeros(eltype(Z),pE_N)
    cE!(CN,X[:,N])
    CE = [vec(C); CN]
    return CE
end

function cI(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

    C = zeros(eltype(Z),pI,N-1)
    for k = 1:N-1
        cI!(view(C,1:pI,k),X[:,k],U[:,k])
    end
    CN = zeros(eltype(Z),pI_N)
    cI!(CN,X[:,N])
    CI = [vec(C); CN]
    return CI
end

function dynamics(Z)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

    D = zeros(eltype(Z),n,N)
    D[:,1] = X[:,1] - solver.obj.x0
    for k = 2:N
        solver.fd(view(D,1:n,k),X[:,k-1],U[:,k-1])
        D[:,k] -= X[:,k]
    end
    return vec(D)
end

function active_set(Z,eps=0)
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx.+(1:Nu)],m,N-1)

    a = ones(Bool,NN)
    ci = cI(Z)
    c_inds = ones(Bool,length(ci))
    c_inds = ci .>= -eps

    a[(Nz + Nx + Nh) .+ (1:Ng)] = c_inds
    # a[ind1.s] = c_inds
    return a
end

function armijo_line_search(merit::Function,V,d,grad; max_iter=10, ϕ=0.01)
    α = 1
    J_prev = merit(V)
    J = merit(V+α*d)
    iter = 1
    while J > J_prev + α*ϕ*grad'd
        α *= 0.75
        J = merit(V + α*d)
        if iter > max_iter
            println("Max iterations")
            α = 0
            break
        end
        iter += 1
    end
    return α
end

function solve_newton(V,ρ,type; iters=10, verbose=false, iters_linesearch=10)
    # Define merit function
    meritfun(V) = al_lagrangian(V,ρ)
    max_c(V) = norm(dynamics(V),Inf)

    # Initial cost
    J0 = meritfun(V)
    V_ = copy(V)
    println("Initial Cost: $J0")

    # Stats
    cost = zeros(iters)
    grad = zeros(iters)
    c_max = zeros(iters)

    # Take Newton Steps
    for i = 1:iters
        A,b = buildKKT(V_,ρ,type)
        δV = -A\b
        α = armijo_line_search(meritfun,V_,δV,b, max_iter=iters_linesearch)
        V_ = V_ + α*δV
        J = meritfun(V_)
        cost[i] = J
        grad[i] = norm(b)
        c_max[i] = max_c(V_)
        if verbose
            println("Iter $i:")
            println("  cost: $J")
            println("  grad: $(grad[i])")
            println("  c_max: $(c_max[i])")
            println("  α: $α")
            println("  rank: $(rank(A))")
            println("  cond: $(cond(A))")
        end
    end
    stats = Dict("cost"=>cost,"grad"=>grad,"c_max"=>c_max)
    return V, stats
end

function buildKKT(V,ρ,type=:penalty)
    Z = V[1:Nz]
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx+1:end],m,N-1)
    nu = V[Nz.+(1:Nx)]
    λ = V[(Nz + Nx) .+ (1:Nh)]
    μ = V[(Nz + Nx + Nh) .+ (1:Ng)]

    ∇²J = ForwardDiff.hessian(mycost,Z)
    ∇J = ForwardDiff.gradient(mycost,Z)
    D = ForwardDiff.jacobian(dynamics,Z)
    H = ForwardDiff.jacobian(cE,Z)
    G = ForwardDiff.jacobian(cI,Z)
    d = dynamics(Z)
    h = cE(Z)
    g = cI(Z)

    if type == :penalty
        A = [∇²J   D'         H'      G';
             D   -1/ρ*I       zeros(Nx,Nh+Ng) ;
             H   zeros(Nh,Nx) -1/ρ*I zeros(Nh,Ng);
             G   zeros(Ng,Nx) zeros(Ng,Nh) -1/ρ*I]
        b = [∇J + D'nu + H'λ + G'μ; d; h; g]
    elseif type == :kkt
        A = [∇²J   D'    H'                G';
             D   zeros(Nx,Nx) zeros(Nx,Nh) zeros(Nx,Ng);
             H   zeros(Nh,Nx) zeros(Nh,Nh) zeros(Nh,Ng);
             G   zeros(Ng,Nx) zeros(Ng,Nh) zeros(Ng,Ng)]
        b = [∇J + D'nu + H'λ + G'μ; d; h; g]
    elseif type == :ad_lagrangian
        A = ForwardDiff.hessian(lagrangian,V)
        b = ForwardDiff.gradient(lagrangian,V)
    elseif type == :ad_aulag
        meritfun(V) = al_lagrangian(V,ρ)
        A = ForwardDiff.hessian(meritfun,V)
        b = ForwardDiff.gradient(meritfun,V)
    elseif type == :gradient_descent
        A = Diagonal(I,NN)
        b = ForwardDiff.gradient(lagrangian,V)
    end
    return A,b
end

function viewmatrix(A::Matrix,filename="logs/viewer.txt")
    io = open(filename,"w")
    show(IOContext(io, :limit=>false),"text/plain",A)
    close(io)
end

function create_results_from_newton(V)
    results = ConstrainedVectorResults(n,m,p,N,p_N)
    Z = V[1:Nz]
    X = reshape(Z[1:Nx],n,N)
    U = reshape(Z[Nx+1:end],m,N-1)
    nu = V[Nz.+(1:Nx)]
    λ = V[(Nz + Nx) .+ (1:Nh)]
    μ = V[(Nz + Nx + Nh) .+ (1:Ng)]
    copyto!(results.X,X)
    copyto!(results.U,U)

    lambda = reshape(view(λ,1:(N-1)*pE),pE,N-1)
    mu = reshape(view(μ,1:(N-1)*pI),pI,N-1)
    copyto!(results.λ,[mu; lambda])
    results.λ[N] = [μ[end-pI_N+1:end]; λ[end-pE_N+1:end]]
    return results
end

function newton_step(V,ρ,type;
    eps=1e-2, verbose=false, iters_linesearch=10, projection=:none, reg=Diagonal(zero(V)), meritfun=al_lagrangian(V,ρ))

    # Define constraint violation functions
    max_c2(V) = max(Ng > 0 ? maximum(cI(V)) : 0, norm(cE(V),Inf))
    max_c(V) = max(norm(dynamics(V),Inf),max_c2(V))

    # Initial cost
    J0 = meritfun(V)
    V_ = copy(V)
    verbose ? println("Initial Cost: $J0") : nothing

    # Build and solve KKT
    A,b = buildKKT(V_,ρ,type)
    a = active_set(V_,1e-3)
    amu = a[ind1.μ]
    δV = zero(V)
    Ā = A[a,a] + reg[a,a]
    δV[a] = -Ā\b[a]

    if projection == :jacobian
        V1 = V + δV
        Z1 = V1[ind1.z]
        Z = V[ind1.z]
        D = ForwardDiff.jacobian(dynamics,Z)
        H = ForwardDiff.jacobian(cE,Z)
        G = ForwardDiff.jacobian(cI,Z)
        Y = [D;H;G[amu,:]]
        d1 = dynamics(Z1)
        h1 = cE(Z1)
        g1 = cI(Z1)
        y = [d1;h1;g1[amu]]
        # dv̂ = -D'*((D*D')\d1)
        δV̂ = -Y'*((Y*Y')\y)
        δV[ind1.z] += δV̂
    elseif projection == :constraints
        b̂ = copy(b)
        for j = 1:5
            V1 = V_ + δV
            d1 = dynamics(V1)
            h1 = cE(V1)
            g1 = cI(V1)
            b̂[ind1.ν] = d1
            b̂[ind1.λ] = h1
            b̂[ind1.μ] = g1
            δV[a] = -Ā\b̂[a]
            V_ = V_ + δV
        end
    end

    # Line Search
    α = armijo_line_search(meritfun,V,δV,b, max_iter=iters_linesearch)
    V_ = V_ + α*δV
    J = meritfun(V_)
    cost = J
    grad = norm(b)
    c_max = max_c(V_)

    change(x,x0) = format((x0-x)/x0*100,precision=2) * "%"
    println()
    println("  cost: $J $(change(J,J0))")
    println("  grad: $(grad)")
    println("  c_max: $(c_max)")
    println("  c_max2: $(max_c2(V_))")
    println("  α: $α")
    println("  rank: $(rank(Ā))")
    println("  cond: $(cond(Ā))")
    stats = Dict("cost"=>cost,"grad"=>grad,"c_max"=>c_max)
    return V_
end


function createV(res)
    x = vec(res.X)
    u = vec(res.U)
    nu = zeros(n,N)

    for k = 1:N
        nu[:,k] = res.s[k]*0
    end
    nu = vec(nu)

    if res isa ConstrainedVectorResults
        s = zeros(pI,N-1)
        μ = zeros(pI,N-1)
        λ = zeros(pE,N-1)
        ineq = 1:pI
        eq = pI .+ (1:pE)
        for k = 1:N-1
            μ[:,k] = res.λ[k][ineq]
            λ[:,k] = res.λ[k][eq]
        end
        s = vec(s)
        μ = vec(μ)
        λ = vec(λ)
        append!(s, sqrt.(2.0*max.(0,-res.C[N][1:pI_N])))
        append!(μ, res.λ[N][1:pI_N])
        append!(λ, res.λ[N][pI_N .+ (1:pE_N)])
    else
        λ = Float64[]
        μ = Float64[]
    end
    return [x;u; nu; λ; μ]
end

function KKT_reg(;z=0,ν=0,λ=0,μ=0)
    r = ones(NN)
    r[ind1.z] .= z
    r[ind1.ν] .= -ν
    r[ind1.λ] .= -λ
    r[ind1.μ] .= -μ
    Diagonal(r)
end

isfullrank(A) = rank(A) == minimum(size(A))


Nx = N*n
Nu = (N-1)*m
Nz = Nx + Nu
Nh = (N-1)*pE + pE_N
Ng = (N-1)*pI + pI_N
NN = 2Nx + Nu + Nh + Ng
names = (:z,:ν,:λ,:μ)
ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng],names)
ind2 = TrajectoryOptimization.create_partition2([Nz,Nx,Nh,Ng],names)

inds = (z=1:Nz, ν=Nz.+(1:Nx), λ=(Nz+Nx) .+ (1:Nh), μ=(Nz+Nx+Nh) .+ (1:Ng))

# Build KKT System
V = createV(res)
Z = V[ind1.z]

ρ = 1000
meritfun(V) = al_lagrangian(V,1)

V = createV(res)
reg = KKT_reg(z=1,ν=1,λ=1,μ=10)
V = newton_step(V,1,:kkt,projection=:none,eps=1e-2,meritfun=meritfun)
d = dynamics(V)
nu = V[ind1.ν]
sign.(d) == sign.(nu)
d .* nu



import TrajectoryOptimization: NewtonSolver, gen_newton_functions, create_V
nsolver = NewtonSolver(solver)
nsolver.opts[:ϵ_as] = 1e-2
Vn = create_V(nsolver,res)
solve(nsolver,Vn,ρ,meritfun=meritfun)
solve(nsolver,res)
newton_step2,buildAb,act_set = gen_newton_functions(nsolver)


# Comparison
max_c2(V) = max(Ng > 0 ? maximum(cI(V)) : 0, norm(cE(V),Inf))
max_c(V) = max(norm(dynamics(V),Inf),max_c2(V))

ρ = 1
meritfun(V) = al_lagrangian(V,ρ)
V = createV(res)
Vn = create_V(nsolver,res)
V == Vn

A,b = buildKKT(V,ρ,:kkt)
An,bn = buildAb(Vn,ρ,:kkt)
A == An
b == bn

a = active_set(V,1e-2)
an = act_set(V,1e-2)
a == an

dv = dvn = zero(V)
dv[a] = -A[a,a]\b[a]
dvn[a] = -An[a,a]\bn[a]
dv == dvn

armijo_line_search(meritfun,V,dv,b)
armijo_line_search(meritfun,Vn,dvn,bn)

V1 = V + dv
V1n = Vn + dvn
V1 == V1n

max_c(V1)
max_c2(V1)

# Build KKT
# A,b = buildKKT(V,ρ,:penalty)
A_,b_ = buildKKT(V,ρ,:kkt)
a = active_set(Z,1e-2)
n_active = Ng - (length(a) - count(a))
amu = a[inds.μ]
reg = 0
A = A_[a,a] + KKT_reg(z=0,ν=reg,λ=reg,μ=reg)[a,a]
b = b_[a]

# Take Step
dv = zero(V)
dv[a] = -A\b
# dv[a] = -A[a,a]\b[a]
rank(A)
isfullrank(A)
cond(A)
r = A*dv[a] + b
norm(r)
# dv[a] = -A[a,a]\b[a]

α = armijo_line_search(meritfun,V,dv,b_)

# Projection corection
V1 = V + dv
Z1 = V1[ind1.z]
∇J = ForwardDiff.gradient(mycost,Z)
D = ForwardDiff.jacobian(dynamics,Z)
H = ForwardDiff.jacobian(cE,Z)
G = ForwardDiff.jacobian(cI,Z)
Y = [D;H;G[amu,:]]
d1 = dynamics(Z1)
h1 = cE(Z1)
g1 = cI(Z1)
y = [d1;h1;g1[amu]]
# dv̂ = -D'*((D*D')\d1)
dv̂ = -Y'*((Y*Y')\y)
dv[ind1.z] += dv̂
b̂ = [∇J + D'nu + H'λ + G'μ; d1; h1; g1]

# Resolve for constraints
_,b1 = buildKKT(V1,ρ,:kkt)
meritfun(V)
dv2[a] = A_[a,a]\b1[a]
α = armijo_line_search(meritfun,V,dv2,b1)

V1 = V + dv*α

# Evaluatex
dJ = J0 - lagrangian(V1)
d = dynamics(V1)
norm(d)
norm(d,Inf)
argmax(abs.(dynamics(V1)))
norm(cE(V1))
norm(b_)

g = cI(V1)
maximum(g[amu])
maximum(g)
amu[argmax(g)] == false

# Cap multipier
mu1 .= max.(0,mu1)
V1[inds.μ] = mu1
V = copy(V1)

results = create_results_from_newton(V)
copyto!(results.μ,res.μ)
TrajectoryOptimization.update_constraints!(results,solver)
TrajectoryOptimization.update_jacobians!(results,solver)
max_violation(results)
plot()
plot_trajectory!(results)

J0 = cost(solver,results)
copyto!(results.U_,results.U)
rollout!(results.X_,results.U_,solver)
TrajectoryOptimization.update_constraints!(results,solver,results.X_,results.U_)
max_violation(results)
plot_trajectory!(to_array(results.X_))
results.X_

∇v = backwardpass!(results,solver)
rollout!(results,solver,0.0)
max_violation(results)
J = cost(solver,results,results.X_,results.U_)
J0 - J
max_violation(res)

copyto!(results.X,results.X_)
copyto!(results.U,results.U_)
plot_trajectory!(to_array(results.X_))
Vnew = createV(results)
norm(dynamics(Vnew),Inf)

ForwardDiff.gradient(lagrangian,V1)
Z = V1[1:Nz]
X = reshape(Z[1:Nx],n,N)
U = reshape(Z[Nx.+(1:Nu)],m,N-1)
plot(X')
lagrangian(V1)

Afd,bfd = buildKKT(V,ρ,:ad_lagrangian)

meritfun(V) = al_lagrangian(V,ρ)
armijo_line_search(meritfun,V,-b,b; max_iter=15)
J0 = meritfun(V)
meritfun(V + 0.001*b)

# Check difference in Hessians
norm(Afd-A_)
norm(bfd-b_)



ρ = 1e10
V_pen, stats_pen = solve_newton(V,ρ,:penalty,verbose=true)
V_kkt, stats_kkt = solve_newton(V,ρ,:kkt,verbose=true)
V_fd, stats_fd = solve_newton(V,ρ,:ad_lagrangian,verbose=true)
V_fda, stats_fda = solve_newton(V,ρ,:ad_aulag)


mycost(V_pen[1:Nz])
val = "c_max"
plot(stats_pen[val],label="penalty",title=val,xlabel="iteration",yscale=:log10)
plot!(stats_kkt[val],label="kkt")
plot!(stats_fd[val],label="fd-lagrangian")
plot!(stats_fda[val],label="fd-aug lagrangian")
