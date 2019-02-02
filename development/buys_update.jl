model, obj_uncon = TrajectoryOptimization.Dynamics.pendulum

# -Constrained objective
obj = ConstrainedObjective(obj_uncon, u_min=-2, u_max=2, x_min=[-20;-20], x_max=[20;20])

# Solver
solver = Solver(model,obj,integration=:rk4,N=51)
U0 = ones(solver.model.m,solver1.N)
X0 = line_trajectory(solver)

results = init_results(solver,X0,U0)
rollout!(results,solver)
update_jacobians!(results, solver)
update_constraints!(results,solver)
_backwardpass!(results,solver)

results.bp

bp = results.bp
n,m,N = get_sizes(solver)
n̄,nn = get_num_states(solver)
m̄,mm = get_num_controls(solver)
p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

nm = nn + mm
Nz = nn*N + mm*(N-1)
Np = p*(N-1) + p_N
∇²L = zeros(Nz,Nz)
∇c = zeros(Np,Nz)

for k = 1:N
    if k < N
        idx = ((k-1)*nm + 1):k*nm
        ∇²L[idx,idx] = [bp.Qxx[k] bp.Qux[k]'; bp.Qux[k] bp.Quu[k]]

        idx2 = ((k-1)*p + 1):k*p
        ∇c[idx2,idx] = [results.Cx[k] results.Cu[k]]
    else
        idx = ((k-1)*nm + 1):Nz
        ∇²L[idx,idx] = results.S[N]

        idx2 = ((k-1)*p + 1):Np
        ∇c[idx2,idx] = results.Cx[N]
    end
end

∇²L
∇c

C = vcat(results.C...)
λ = vcat(results.λ...)

results.λ
λ[1] = 10.
active_set = vcat(results.active_set...)

∇c̄ = ∇c[active_set,:]

λ[active_set] += (∇c̄*(∇²L\∇c̄'))\C[active_set]

# update the results
for k = 1:N
    if k != N
        idx_pI = pI
        idx = (k-1)*p+1:k*p
    else
        idx_pI = pI_N
        idx = (k-1)*p+1:Np
    end
    results.λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, λ[idx]))
    results.λ[k][1:idx_pI] = max.(0.0,results.λ[k][1:idx_pI])
end
