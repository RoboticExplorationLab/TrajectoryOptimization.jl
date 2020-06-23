@testset "Dynamics constraints" begin
model = Cartpole()
n,m = size(model)
N = 11
dt = 0.1
Z = Traj([rand(n) for k = 1:N], [rand(m) for k = 1:N-1], fill(dt,N))

dyn = TO.DynamicsConstraint(model, N)
@test TO.evaluate(dyn, Z[1], Z[2]) ≈ discrete_dynamics(RK3, model, Z[1]) - state(Z[2])
F = TO.gen_jacobian(dyn)
F0 = zero(F)
@test TO.jacobian!(F, dyn, Z[1], Z[2]) == false
discrete_jacobian!(RK3, F0, model, Z[1])
@test F0 ≈ F
F .= 0
@test TO.jacobian!(F, dyn, Z[1], Z[2], 2) == true
@test F ≈ -Matrix(I,n,n+m)

dyn = TO.DynamicsConstraint{RK4}(model, N)
@test TO.evaluate(dyn, Z[1], Z[2]) ≈ discrete_dynamics(RK4, model, Z[1]) - state(Z[2])
F = TO.gen_jacobian(dyn)
F0 = zero(F)
@test TO.jacobian!(F, dyn, Z[1], Z[2]) == false
discrete_jacobian!(RK4, F0, model, Z[1])
@test F0 ≈ F
F .= 0
@test TO.jacobian!(F, dyn, Z[1], Z[2], 2) == true
@test F ≈ -Matrix(I,n,n+m)

@test length(dyn) == n
@test TO.widths(dyn) == (n+m,n)

dyn = TO.DynamicsConstraint{HermiteSimpson}(model, N)
conval = TO.ConVal(n, m, dyn, 1:N-1)
TO.evaluate!(conval, Z)
X = states(Z)
U = controls(Z)
Xm = dyn.xMid
fVal = dyn.fVal
fValm = dynamics(model, Xm[1], 0.5*(U[1] + U[2]))
@test conval.vals[1] ≈ X[1] - X[2] + dt * (fVal[1] + 4*fValm + fVal[2])/6

@test length(dyn) == n
@test TO.widths(dyn) == (n+m,n+m)
@test size(conval.jac[2]) == (n,n+m)
@test TO.upper_bound(dyn) == zeros(n)
@test TO.lower_bound(dyn) == zeros(n)
end
