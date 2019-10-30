
# Quadrotor in Maze

# model
model = Dynamics.quadrotor
# model = Dynamics.quadrotor_euler
model_d = rk3(model)
n = model.n; m = model.m
q0 = [1.;0.;0.;0.] # unit quaternion

x0 = zeros(T,n)
x0[1:3] = [0.; 0.; 10.]
x0[4:7] = q0

xf = zero(x0)
xf[1:3] = [0.;60.; 10.]
xf[4:7] = q0;

# cost
Q = (1.0e-3)*Diagonal(I,n)
Q[4:7,4:7] = (1.0e-3)*Diagonal(I,4)
R = (1.0e-2)*Diagonal(I,m)
Qf = 1.0*Diagonal(I,n)

u_min = 0.
u_max = 50.
x_max = Inf*ones(model.n)
x_min = -Inf*ones(model.n)

x_max[1:3] = [25.0; Inf; 20]
x_min[1:3] = [-25.0; -Inf; 0.]
bnd_u = BoundConstraint(n,m,u_min=u_min, u_max=u_max)
bnd = BoundConstraint(n,m,u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

xf_no_quat_U = copy(xf)
xf_no_quat_L = copy(xf)
xf_no_quat_U[4:7] .= Inf
xf_no_quat_L[4:7] .= -Inf
xf_no_quat_U[8:10] .= 0.
xf_no_quat_L[8:10] .= 0.
bnd_xf = BoundConstraint(n,m,x_min=xf_no_quat_L,x_max=xf_no_quat_U)
# goal = goal_constraint(xf)

N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

U_hover = [0.5*9.81/4.0*ones(m) for k = 1:N-1] # initial hovering control trajectory
obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs

# Constraints
cyl_obs = [(0., 10., 3.),
          (10., 30., 3.),
          (-13., 25., 2.),
          (5.,50.,4.)]

sph_obs = [( 0., 40., 5., 2.),
           (-5., 15., 3., 1.),
           (10., 20., 7., 2.)]
r_quad = 2.0

cylinder_constraint = let cylinders=cyl_obs, spheres=sph_obs, r_quad=r_quad
    function cylinder_constraint(c,x,u)
        for (i,cyl) in enumerate(cylinders)
            c[i] = circle_constraint(x,cyl[1],cyl[2],cyl[3]+r_quad)
        end
    end
end

sphere_constraint = let cylinders=cyl_obs, spheres=sph_obs, r_quad=r_quad
    function sphere_constraint(c,x,u)
        for (i,sphere) in enumerate(spheres)
            c[i] = TrajectoryOptimization.sphere_constraint(x,sphere[1],sphere[2],sphere[3],sphere[3]+r_quad)
        end
    end
end

cyl_con = Constraint{Inequality}(cylinder_constraint,n,m,length(cyl_obs),:cylinders)
sph_con = Constraint{Inequality}(  sphere_constraint,n,m,length(sph_obs),:spheres)

constraints = Constraints(N)
constraints[1] += bnd_u
for k = 2:N-1
    constraints[k] += bnd + cyl_con + sph_con
end
constraints[N] += bnd_xf

quad_obs = Problem(model_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(quad_obs,U_hover); # initialize problem with control
