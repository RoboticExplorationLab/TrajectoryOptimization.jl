using LinearAlgebra
using BlockArrays
using ForwardDiff
using BenchmarkTools

model,obj = Dynamics.dubinscar
solver = Solver(model,obj,N=21)
n,m,N = get_sizes(solver)
obj.cost.Q .= Diagonal(I,n)
obj.cost.R .= Diagonal(I,m)*2
obj.cost.H .= ones(m,n)

X = rand(n,N)
U = rand(m,N-1)
dt = solver.dt
dx = zeros(n,n)
du = zeros(n,m)
x = X[:,1]
u = U[:,1]
solver.Fd(dx,du,x,u)
dz = zeros(n,n+m)
z = [x;u]
solver.Fd(dz,z)
dz == [dx du]

ind_z = create_partition((n,m),(:x,:u))
ind2_z = create_partition2((n,m),(:x,:u))
f = model.f
fd! = TrajectoryOptimization.rk4(f,solver.dt)
fd_aug!(xdot,z) = fd!(xdot,view(z,ind_z.x),view(z,ind_z.u))
f_aug!(xdot,z) = f(xdot,view(z,ind_z.x),view(z,ind_z.u))
fd_aug2!(dS,S::Array) = begin
    k1 = k2 = k3 = k4 = zero(S)
    f_aug!(k1,S);         k1 *= dt;
    f_aug!(k2,S + k1/2); k2 *= dt;
    f_aug!(k3,S + k2/2); k3 *= dt;
    f_aug!(k4,S + k3);    k4 *= dt;
    copyto!(dS, (S + (k1 + 2*k2 + 2*k3 + k4)/6)[1:n])
end
xdot = zeros(n)
fd_aug!(xdot,z)
xdot2 = zeros(n)
fd_aug2!(xdot2,z)
xdot2 == xdot


Fd!(dz,z) = ForwardDiff.jacobian!(dz,fd_aug!,xdot,z)
Fd2!(dz,z) = ForwardDiff.jacobian!(dz,fd_aug2!,xdot,z)
dz2 = zero(dz)
Fd!(dz,z)
Fd2!(dz2,z)
dz == dz2 == [dx du]
@btime Fd!($dz,$z)
@btime Fd2!($dz,$z)
@btime solver.Fd($dz,$z)

inds = (x=1:n,u=n .+ (1:m), dt=n+m .+ (1:1), xx=(1:n,1:n),xu=(1:n,n .+ (1:m)), xdt=(1:n,n+m.+(1:1)))

function Fd_split!(dx,du,x,u)
    z[ind_z.x] = x
    z[ind_z.u] = u
    Fd!(dz,z)
    copyto!(dx,dz[ind2_z.xx])
    copyto!(du,dz[ind2_z.xu])
end
dx2 = zero(dx)
du2 = zero(du)
solver.Fd(dx,du,x,u)
Fd_split!(dx2,du2,x,u)
dx == dx2
du == du2
@btime solver.Fd($dx,$du,$x,$u)
@btime Fd_split!($dx,$du,$x,$u)
