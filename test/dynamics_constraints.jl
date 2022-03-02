function test_allocs(dyn::TO.DynamicsConstraint)
    n,m = RD.dims(dyn)
    N = 11
    dt = 0.1
    vals = [zeros(n) for k = 1:N]
    vals2 = SVector{n}.(vals) 
    jacs = [zeros(n,n+m) for k = 1:N, i = 1:2]
    Z = SampledTrajectory([@SVector rand(n) for k = 1:N], [@SVector rand(m) for k = 1:N-1], dt=dt) 

    allocs = 0
    allocs += @ballocated TO.constraint_jacobians!(RD.StaticReturn(), RD.ForwardAD(), $dyn, $jacs, $vals, $Z) samples=2 evals=1
    allocs += @ballocated TO.constraint_jacobians!(RD.StaticReturn(), RD.FiniteDifference(), $dyn, $jacs, $vals, $Z) samples=2 evals=1
    allocs += @ballocated TO.constraint_jacobians!(RD.InPlace(), RD.ForwardAD(), $dyn, $jacs, $vals, $Z) samples=2 evals=1
    allocs += @ballocated TO.constraint_jacobians!(RD.InPlace(), RD.FiniteDifference(), $dyn, $jacs, $vals, $Z) samples=2 evals=1

    allocs += @ballocated TO.evaluate_constraints!(RD.StaticReturn(), $dyn, $vals, $Z) samples=2 evals=1
    allocs += @ballocated TO.evaluate_constraints!(RD.InPlace(), $dyn, $vals, $Z) samples=2 evals=1

    Z = SampledTrajectory([rand(n) for k = 1:N], [rand(m) for k = 1:N-1], dt=dt) 
    allocs += @ballocated TO.evaluate_constraints!(RD.InPlace(), $dyn, $vals, $Z) samples=2 evals=1
    allocs += @ballocated TO.constraint_jacobians!(RD.InPlace(), RD.ForwardAD(), $dyn, $jacs, $vals, $Z) samples=2 evals=1
    allocs += @ballocated TO.constraint_jacobians!(RD.InPlace(), RD.FiniteDifference(), $dyn, $jacs, $vals, $Z) samples=2 evals=1
    return allocs
end

@testset "Dynamics constraints" begin
model = Cartpole()
n,m = RD.dims(model)
N = 11
dt = 0.1
Z = SampledTrajectory([@SVector rand(n) for k = 1:N], [@SVector rand(m) for k = 1:N-1], dt=dt)

vals = [zeros(n) for k = 1:N]
vals2 = SVector{n}.(vals) 
jacs = [zeros(n,n+m) for k = 1:N, i = 1:2]
J = zero(jacs[1])

local dyn_explicit, dyn_implicit

@testset "Explicit" begin
@test_throws ErrorException TO.DynamicsConstraint(model)

dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
dyn = TO.DynamicsConstraint(dmodel)
TO.evaluate_constraints!(RD.InPlace(), dyn, vals, Z)
for k = 1:N-1
    @test vals[k] ≈ RD.discrete_dynamics(dmodel, Z[k]) - RD.state(Z[k+1])
end

TO.evaluate_constraints!(RD.StaticReturn(), dyn, vals, Z)
TO.evaluate_constraints!(RD.StaticReturn(), dyn, vals2, Z)
for k = 1:N-1
    @test vals[k] ≈ RD.discrete_dynamics(dmodel, Z[k]) - RD.state(Z[k+1])
    @test vals2[k] ≈ RD.discrete_dynamics(dmodel, Z[k]) - RD.state(Z[k+1])
end

TO.constraint_jacobians!(RD.InPlace(), RD.ForwardAD(), dyn, jacs, vals, Z)
J = zero(jacs[1])
for k = 1:N-1
    RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dmodel, J, vals[k], Z[k])
    @test jacs[k] ≈ J
    @test jacs[k,2] ≈ [-I(n) zeros(n,m)]
end

TO.constraint_jacobians!(RD.StaticReturn(), RD.FiniteDifference(), dyn, jacs, vals, Z)
J = zero(jacs[1])
for k = 1:N-1
    RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dmodel, J, vals[k], Z[k])
    @test jacs[k] ≈ J atol=1e-6
    @test jacs[k,2] ≈ [-I(n) zeros(n,m)]
end

dyn_explicit = dyn
end

# Implicit integrator 
@testset "Implicit" begin
dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
dyn = TO.DynamicsConstraint(dmodel)
TO.evaluate_constraints!(RD.InPlace(), dyn, vals, Z)
for k = 1:N-1
    fmid = RD.dynamics(model, (RD.state(Z[k]) + RD.state(Z[k+1]))/2, RD.control(Z[k]))
    @test vals[k] ≈ RD.state(Z[k]) - RD.state(Z[k+1]) + dt*fmid
end

TO.evaluate_constraints!(RD.StaticReturn(), dyn, vals, Z)
TO.evaluate_constraints!(RD.StaticReturn(), dyn, vals2, Z)
for k = 1:N-1
    fmid = RD.dynamics(model, (RD.state(Z[k]) + RD.state(Z[k+1]))/2, RD.control(Z[k]))
    @test vals[k] ≈ RD.state(Z[k]) - RD.state(Z[k+1]) + dt*fmid
    @test vals2[k] ≈ RD.state(Z[k]) - RD.state(Z[k+1]) + dt*fmid
end

TO.constraint_jacobians!(RD.InPlace(), RD.ForwardAD(), dyn, jacs, vals, Z)
mid(x1,x2,u) = x1 + RD.dynamics(model, (x1 + x2)/2, u) * dt - x2
for k = 1:N-1
    J1 = copy(J)
    J2 = copy(J)
    x1,x2 = RD.state(Z[k]), RD.state(Z[k+1])
    u = RD.control(Z[k])
    A1 = ForwardDiff.jacobian(x->mid(x,x2,u), x1)
    A2 = ForwardDiff.jacobian(x->mid(x1,x,u), x2)
    B1 = ForwardDiff.jacobian(u->mid(x1,x2,u), u)
    @test jacs[k,1] ≈ [A1 B1]
    @test jacs[k,2] ≈ [A2 zeros(n,m)]
end

TO.constraint_jacobians!(RD.StaticReturn(), RD.FiniteDifference(), dyn, jacs, vals, Z)
for k = 1:N-1
    J1 = copy(J)
    J2 = copy(J)
    x1,x2 = RD.state(Z[k]), RD.state(Z[k+1])
    u = RD.control(Z[k])
    A1 = ForwardDiff.jacobian(x->mid(x,x2,u), x1)
    A2 = ForwardDiff.jacobian(x->mid(x1,x,u), x2)
    B1 = ForwardDiff.jacobian(u->mid(x1,x2,u), u)
    @test jacs[k,1] ≈ [A1 B1] atol=1e-6
    @test jacs[k,2] ≈ [A2 zeros(n,m)] atol=1e-6
end

@test RD.output_dim(dyn) == n
@test RD.dims(dyn) == (n,m,n)
@test TO.widths(dyn) == (n+m,n+m)
@test TO.upper_bound(dyn) == zeros(n)
@test TO.lower_bound(dyn) == zeros(n)

dyn_implicit = dyn
end

if run_alloc_tests
    @testset "Allocations" begin
        @test test_allocs(dyn_explicit) == 0
        @test test_allocs(dyn_implicit) == 0
    end
end
end