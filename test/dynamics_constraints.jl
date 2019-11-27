model = Dynamics.Cartpole()
prob = Problems.cartpole_static
n,m = size(prob)

rollout!(prob)
Z = prob.Z
vals = [@SVector zeros(model.n) for k = 1:prob.N-1]
∇c = [@SMatrix zeros(n,2n+m) for k = 1:prob.N-1]

rk3_dyn = ImplicitDynamics(model, prob.N)
@btime evaluate($rk3_dyn, $Z[2], $Z[1])
@btime jacobian($rk3_dyn, $Z[2], $Z[1])
@btime evaluate!($vals, $rk3_dyn, $Z)
@btime jacobian!($∇c, $rk3_dyn, $Z)

hs_dyn = ExplicitDynamics{HermiteSimpson}(model, prob.N)
∇c = [@SMatrix zeros(n,2(n+m)) for k = 1:prob.N-1]
@btime evaluate!($vals, $hs_dyn, $Z)
@btime jacobian!($∇c, $hs_dyn, $Z)

con_hs = ConstraintVals(hs_dyn, 1:prob.N)
@btime evaluate!($con_hs, $Z)
@btime jacobian!($con_hs, $Z)
