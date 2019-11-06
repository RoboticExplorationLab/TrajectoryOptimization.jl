prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

solve!(prob, ilqr)
plot(prob.X)


solve!(sprob, silqr)
Xs = state(sprob)
Us = control(sprob)
plot(Xs)
norm(Xs - prob.X)

U0 = [u0 for k = 1:prob.N-1]
@btime begin
    initial_controls!($prob, $U0)
    solve($prob,$ilqr)
end

x0 = sprob.x0
u0 = control(sprob)[1]
@btime begin
    for k = 1:$sprob.N
        $sprob.Z[k].z = [$x0*NaN; $u0]
    end
    solve!($sprob, $silqr)  # 55x speedup
end


# Now make augmented lagragian problem
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)

al = AugmentedLagrangianSolver(prob)
prob_al = AugmentedLagrangianProblem(prob, al)

sopts = AugmentedLagrangianSolverOptions{Float64}()
sopts.opts_uncon = StaticiLQRSolverOptions()
reset!(sprob.constraints, sopts)
sal = StaticALSolver(sprob, sopts)
sprob_al = convertProblem(sprob, sal)

ilqr = iLQRSolver(prob_al)
silqr = StaticiLQRSolver(sprob_al)

solve!(prob_al, ilqr)
solve!(sprob_al, silqr)
norm(state(sprob_al) - prob_al.X)

for i = 1:5
    dual_update!(prob_al, al)
    penalty_update!(prob_al, al)

    dual_update!(sprob_al, sal)
    penalty_update!(sprob_al, sal)

    solve!(prob_al, ilqr)
    solve!(sprob_al, silqr)
end
norm(state(sprob_al) - prob_al.X)

ilqr.ρ[1]
silqr.ρ[1]

max_violation(al)
max_violation(sprob_al)
