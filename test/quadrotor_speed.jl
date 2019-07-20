
quad = Dynamics.quadrotor
quad2 = Dynamics.quadrotor2
n,m = quad.n, quad.m

# Discrete Models
dt = 0.05
quad_d = rk4!(quad, dt)
quad2_d = rk4!(quad2, dt)

# Create not-in-place model
params = Dynamics.quad_params
f(x,u) = Dynamics.quadrotor_dynamics2(x,u,params)
fd = rk4(f, dt)
fd!(xdot, x, u, dt) = copyto!(xdot, fd(x, u, dt))
∇fd = TO.generate_jacobian_nip(fd, n, m)
quad3_d = TO.AnalyticalModel{Discrete}(fd!, ∇fd, n, m)

# Test Continuous dynamics
x,u = rand(n), rand(m)
xs,us = SVector{n}(x), SVector{m}(u)
x[1:4] = [1,0,0,0]
xdot = zeros(n)
xdot2 = zeros(n)
xdot3 = zeros(n)
evaluate!(xdot, quad, x, u)
evaluate!(xdot2, quad2, x, u)
xdot3 = f(x,u)
tau,omega = quad.f(xdot,x,u)
xdot ≈ xdot2
xdot ≈ xdot3

@btime evaluate!(xdot, quad, x, u)
@btime evaluate!($xdot2, $quad2, $x, $u)
f = Dynamics.quadrotor_dynamics2
xdot3 = f(x,u, params)
xdot ≈ xdot3
@btime f($x,$u,$params)

# Discrete
evaluate!(xdot, quad_d, x, u, dt)
evaluate!(xdot2, quad2_d, x, u, dt)
evaluate!(xdot3, quad3_d, x, u, dt)
xdot3 ≈ fd(x,u)
xdot ≈ xdot2
xdot ≈ xdot3

@btime quad_d.f($xdot,$x,$u,$dt)
@btime quad2_d.f($xdot2,$x,$u,$dt)
@btime quad3_d.f($xdot3,$x,$u,$dt)
@btime fd(x,u)

# Discrete Jacobian
Z = PartedMatrix(quad_d)
Z2 = PartedMatrix(quad2_d)
Z3 = PartedMatrix(quad2_d)
quad_d.∇f(Z,x,u,dt)
quad2_d.∇f(Z2,x,u,dt)
∇fd(Z3,x,u,dt)
Z ≈ Z2
Z ≈ Z3
@btime quad_d.∇f($Z,$x,$u,$dt)
@btime quad2_d.∇f($Z2,$x,$u,$dt)
@btime ∇fd($Z3,$x,$u,$dt)

# Trajectories of Discrete jacobians
N = 1001
X = [rand(n) for k = 1:N]
U = [rand(m) for k = 1:N-1]
∇F = [PartedMatrix(quad_d) for k = 1:N-1]
∇F2 = [PartedMatrix(quad_d) for k = 1:N-1]
∇F3 = [PartedMatrix(quad_d) for k = 1:N-1]
fVal = [rand(n) for k = 1:N]
jacobian!(∇F, fVal, quad_d, X, U, dt)
jacobian!(∇F2, fVal, quad2_d, X, U, dt)
TO.calculate_jacobian!(∇F3, ∇fd, X, U, dt)
∇F ≈ ∇F2
∇F ≈ ∇F3
@btime jacobian!($∇F, $fVal, $quad_d, $X, $U, $dt)
@btime TO.jacobian_parallel!($∇F2, $fVal, $quad2_d, $X, $U, $dt)
@btime TO.calculate_jacobian!($∇F3, $∇fd, $X, $U, $dt)


# cost
Q = (1.0e-2)*Diagonal(I,n)
R = (1.0e-2)*Diagonal(I,m)
Qf = 1000.0*Diagonal(I,n)

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;50.;0.] # xyz position
xf[4:7] = q0

N = 101
U0 = [0.5*9.81/4.0*ones(m) for k = 1:N-1]

costfun = LQRCost(Q, R, Qf, xf)
prob = Problem(quad_d, Objective(costfun, N), U0, x0=x0, tf=5.0)
# initial_controls!(prob, rand(m,N-1))
rollout!(prob)
prob.X
prob2 = update_problem(prob, model=quad2_d)
prob3 = update_problem(prob, model=quad3_d)

solver = iLQRSolver(prob)
solver2 = iLQRSolver(prob2)
solver3 = iLQRSolver(prob3)
jacobian!(solver.∇F, solver.fVal, prob.model, prob.X, prob.U, prob.dt)
jacobian!(solver2.∇F, solver.fVal, prob2.model, prob.X, prob.U, prob.dt)
jacobian!(solver3.∇F, solver.fVal, prob3.model, prob.X, prob.U, prob.dt)
@btime jacobian!($solver.∇F, $solver.fVal, $prob.model, $prob.X, $prob.U, $prob.dt)
@btime jacobian!($solver2.∇F, $solver.fVal, $prob2.model, $prob.X, $prob.U, $prob.dt)
@btime jacobian!($solver3.∇F, $solver.fVal, $prob3.model, $prob.X, $prob.U, $prob.dt)

TO.jacobian_parallel!(solver.∇F, solver.fVal, prob.model, prob.X, prob.U, prob.dt)
TO.jacobian_parallel!(solver2.∇F, solver.fVal, prob2.model, prob.X, prob.U, prob.dt)
TO.jacobian_parallel2!(solver3.∇F, solver.fVal, prob3.model, prob.X, prob.U, prob.dt)

@btime TO.jacobian_parallel!($solver.∇F, $solver.fVal, $prob.model, $prob.X, $prob.U, $prob.dt)
@btime TO.jacobian_parallel!($solver2.∇F, $solver.fVal, $prob2.model, $prob.X, $prob.U, $prob.dt)
@btime TO.jacobian_parallel2!($solver3.∇F, $solver.fVal, $prob3.model, $prob.X, $prob.U, $prob.dt)

# Test solve
ilqr = iLQRSolverOptions()
@btime solve($prob , $ilqr)
@btime solve($prob2, $ilqr)
@btime solve($prob3, $ilqr)
ilqr = iLQRSolverOptions(parallel=true)
@btime solve($prob , $ilqr)
@btime solve($prob2, $ilqr)
@btime solve($prob3, $ilqr)

prob1_ = copy(prob)
prob2_ = copy(prob2)
prob3_ = copy(prob3)
use_parallel = true
solver1 = iLQRSolver(prob1_, iLQRSolverOptions(parallel=use_parallel))
solver2 = iLQRSolver(prob2_, iLQRSolverOptions(parallel=use_parallel))
solver3 = iLQRSolver(prob3_, iLQRSolverOptions(parallel=use_parallel))
J = cost(prob)
@time TO.step!(prob1_, solver1, J)
@time TO.step!(prob2_, solver2, J)
@time TO.step!(prob3_, solver3, J)

prob1_ = copy(prob3)
prob2_ = copy(prob3)
solver1 = iLQRSolver(prob1_, iLQRSolverOptions(parallel=false))
solver2 = iLQRSolver(prob2_, iLQRSolverOptions(parallel=true))
J = cost(prob1_)
@btime TO.step!(prob1_, solver1, J)
@btime TO.step!(prob2_, solver2, J)

# Try taking a step
@btime jacobian!($prob1_, $solver1)
@btime jacobian!($prob2_, $solver2)
@btime TO.cost_expansion!($prob1_, $solver1)
@btime TO.cost_expansion!($prob2_, $solver2)

function mystep(prob, solver)
    t_start = time_ns()
    for i = 1:1000
        jacobian!(prob, solver)
        TO.cost_expansion!(prob, solver)
    end
    # ΔV = backwardpass!(prob, solver)
    return float(time_ns()-t_start)
end
mystep(prob1_, solver1)
mystep(prob2_, solver2)
solver1.∇F == solver2.∇F
solver1.Q .≈ solver2.Q
maximum(norm.([q2.xx - q1.xx for (q1,q2) in zip(solver1.Q, solver2.Q)],Inf))
maximum(norm.([q2.uu - q1.uu for (q1,q2) in zip(solver1.Q, solver2.Q)],Inf))
maximum(norm.([q2.ux - q1.ux for (q1,q2) in zip(solver1.Q, solver2.Q)],Inf))
maximum(norm.([q2.x - q1.x for (q1,q2) in zip(solver1.Q, solver2.Q)],Inf))
maximum(norm.([q2.u - q1.u for (q1,q2) in zip(solver1.Q, solver2.Q)],Inf))
@btime backwardpass!($prob1_, $solver1)
@btime backwardpass!($prob2_, $solver2)

mystep(prob1_, solver1)
mystep(prob2_, solver2)
@code_warntype mystep(prob1_, solver1)
@code_warntype mystep(prob2_, solver2)
@profiler mystep(prob1_, solver1)
@profiler mystep(prob2_, solver2)
@btime mystep($prob1_, $solver1)
@btime mystep($prob2_, $solver2)

ilqr = iLQRSolverOptions()
function solve_all(probs, ilqr)
    for i = 1:4
        # i = Threads.threadid()
        solve!(probs[i], ilqr)
    end
end
probs = [copy(prob3) for i = 1:4]
@btime solve_all(probs, ilqr)
