q = normalize(@SVector rand(4))
q0 = normalize(@SVector rand(4))
dq(q) = quat_diff(q,q0)
@test ForwardDiff.jacobian(dq,q) ≈ quat_diff_jacobian(q0)


quad = Dynamics.Quadrotor()
x0 = Problems.quadrotor_static.x0
xf = Problems.quadrotor_static.xf
state_diff(quad, xf, x0)
@test let quad=quad, xf=xf, x0=x0
    allocs = (@allocated state_diff(quad, xf, x0))
    allocs += (@allocated state_diff_jacobian(quad, x0))
end == 0
state_diff_jacobian(quad, x0)
@btime state_diff($quad, $xf, $x0)
@btime state_diff_jacobian($quad, $x0)
sd(x) = let model=quad, x0=x0
    state_diff(model, x, x0)
end
state_diff(quad, xf, x0) == sd(xf)
ForwardDiff.jacobian(sd, xf) ≈ state_diff_jacobian(quad, x0)
