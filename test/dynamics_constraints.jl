model = Dynamics.Cartpole()
prob = Problems.cartpole_static
bnd = prob.constraints.constraints[1]
n,m = size(prob)

rollout!(prob)
Z = prob.Z
vals = [@SVector zeros(model.n) for k = 1:prob.N-1]
∇c = [@SMatrix zeros(n,2n+m) for k = 1:prob.N-1]

rk3_dyn = ImplicitDynamics(model, prob.N)
@btime evaluate($rk3_dyn, $Z[2], $Z[1])
@btime jacobian($rk3_dyn, $Z[2], $Z[1])
@btime evaluate!($vals, $rk3_dyn, $Z, 1:4)
@btime jacobian!($∇c, $rk3_dyn, $Z)

hs_dyn = ExplicitDynamics{HermiteSimpson}(model, prob.N)
∇c = [@SMatrix zeros(n,2(n+m)) for k = 1:prob.N-1]
@btime evaluate!($vals, $hs_dyn, $Z)
@btime jacobian!($∇c, $hs_dyn, $Z)

con_hs = ConstraintVals(hs_dyn, 1:prob.N)
@btime evaluate!($con_hs, $Z)
@btime jacobian!($con_hs, $Z)

con_rk3 = ConstraintVals(rk3_dyn, 1:prob.N)

conSet = ConstraintSets([con_hs, con_rk3], prob.N)
@btime evaluate!($conSet, $Z)
@btime jacobian!($conSet, $Z)

bnd = StaticBoundConstraint(n,m, u_min=-u_bnd*(@SVector ones(m)), u_max=u_bnd*(@SVector ones(m)))
goal = GoalConstraint(SVector{n}(xf))
con_bnd = ConstraintVals(bnd, 1:N-1)
con_goal = ConstraintVals(goal, N:N)
conSet = ConstraintSets([con_bnd, con_goal], N)
