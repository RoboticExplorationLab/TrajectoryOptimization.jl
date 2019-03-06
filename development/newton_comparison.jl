model,obj = Dynamics.quadrotor_3obs
opts = SolverOptions()
opts.verbose = true
opts.square_root = true
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-4

model,obj,circles = Dynamics.dubinscar_obstacles
obj = update_objective(obj,u_max=[10,2],u_min=[-10,-2])


solver = Solver(model,obj,N=101,opts=opts)
n,m,N = get_sizes(solver)
U_hover = 0.5*9.81/4.0*ones(m, N-1)
res,stats = solve(solver,U_hover)
Z = PrimalVars(res)
V = NewtonVars(solver,res)
plot()
plot_obstacles(circles)
plot_trajectory!(res)
plot(res.U)

p,pI,pE = get_num_constraints(solver,:custom)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver,:custom)
Nx,Nu,Nh,Ng,NN = get_batch_sizes(solver)
Nz = Nx+Nu
names = (:z,:ν,:λ,:μ,:r)
ind1 = TrajectoryOptimization.create_partition([Nz,Nx,Nh,Ng,2Nz],names)

ρ = 1000
meritfun(V) = al_lagrangian(V,1)

V = createV(res)
Z = V[ind1.z]
reg = KKT_reg(z=1e-2,ν=1e-2,λ=1e-2,μ=1e-2)
@btime newton_step(V,1,:kkt,projection=:jacobian,eps=1e-3,meritfun=meritfun)


mycost2,c,grad_J,_,c_jacob = gen_usrfun_newton(solver);
c.b(Z.Z)
newton_step2,buildKKT2,as2 = gen_newton_functions(solver)
V2 = NewtonVars(solver,res)
Z2 = primals(V2)
as2(V2)
inds = findall(V2.a[ind1.r])
X = V2.Z.X
A2,b2 = buildKKT2(V2,1)
newton_step2(V2,1,eps=1e-5)

obj = update_objective(obj_c,x_min=-5)
obj = obj_c
z_L,z_U = get_bounds(solver,false)
x_bnd = [isfinite.(obj.x_max); -isfinite.(obj.x_min)]
u_bnd = [isfinite.(obj.u_max); -isfinite.(obj.u_min)]
z_bnd = [u_bnd; x_bnd]
active_bnd = z_bnd .!= 0
jac_x_bnd= [Diagonal(isfinite.(obj.x_max));
           -Diagonal(isfinite.(obj.x_min))]
jac_u_bnd = [Diagonal(isfinite.(obj.u_max));
            -Diagonal(isfinite.(obj.u_min))]
jac_bnd_k = [spzeros(2m,n) jac_u_bnd;
             jac_x_bnd spzeros(2n,m)][active_bnd,:]
# jac_bnd_k = blockdiag(jac_x_bnd,jac_u_bnd)[active_bnd,:]
jac_bnd = blockdiag([jac_bnd_k for k = 1:N-1]...)
jac_bnd = blockdiag(jac_bnd,jac_x_bnd[active_bnd[2m+1:end],:])


ForwardDiff.gradient(c.I,Z2)
Array(jac_bnd_k)
Int.(Array(jac_bnd))

Int.(Array(G))
Int.(Array(Gfd))
