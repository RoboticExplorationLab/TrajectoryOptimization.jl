using Test
using Plots
function test_backwardpass_sqrt!(res::SolverIterResults,solver::Solver,bp::BackwardPass)
    # Get problem sizes
    res = results2
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    # Objective
    costfun = solver.obj.cost

    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; Su = res.S; s = res.s

    for k = 1:N-1
        res.S[k] = zeros(nn+mm,nn)
    end
    Su = res.S
    # # for now just re-instantiate
    # bp = BackwardPass(nn,mm,N)
    # Qx = bp.Qx; Qu = bp.Qu; Qxx = bp.Qxx; Quu = bp.Quu; Qux = bp.Qux
    # Quu_reg = bp.Quu_reg; Qux_reg = bp.Qux_reg

    # Boundary Conditions
    lxx,lx = cost_expansion(costfun, X[N][1:n])

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Terminal constraints
    if res isa ConstrainedIterResults
        C = res.C; Iμ = res.Iμ; λ = res.λ
        Cx = res.Cx; Cu = res.Cu
        Iμ_sqrt = sqrt.(Iμ[N])

        Su[N][1:nn,1:nn] = chol_plus(cholesky(lxx).U,Iμ_sqrt*Cx[N])
        s[N] = lx + Cx[N]'*(Iμ[N]*C[N] + λ[N])

        @test isapprox(lxx + Cx[N]'*Iμ[N]*Cx[N],Su[N]'*Su[N])
    else
        Su[N] = cholesky(lxx).U
        s[N] = lx
    end

    # Backward pass
    k = N-1
    while k >= 1

        x = X[k][1:n]
        u = U[k][1:m]
        h = sqrt(dt)

        expansion = cost_expansion(costfun,x,u)
        lxx,luu,lux,lx,_lu = expansion


        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = dt*lx + fdx'*s[k+1]
        Qu = dt*_lu + fdu'*s[k+1]
        Wxx = chol_plus(cholesky(dt*lxx).U, Su[k+1]*fdx)
        Wuu = chol_plus(cholesky(dt*luu).U, Su[k+1]*fdu)
        Qux = dt*lux + (fdu'*Su[k+1]')*(Su[k+1]*fdx)

        @test isapprox(dt*lxx + fdx'*Su[k+1]'*Su[k+1]*fdx, Wxx'*Wxx)
        @test isapprox(dt*luu + fdu'*Su[k+1]'*Su[k+1]*fdu, Wuu'*Wuu)
        # Constraints
        if res isa ConstrainedIterResults
            Iμ_sqrt = sqrt.(Iμ[k])

            Qx += Cx[k]'*(Iμ[k]*C[k] + λ[k])
            Qu += Cu[k]'*(Iμ[k]*C[k] + λ[k])
            Wxx = chol_plus(Wxx,Iμ_sqrt*Cx[k])
            Wuu = chol_plus(Wuu,Iμ_sqrt*Cu[k])
            Qux += Cu[k]'*Iμ[k]*Cx[k]

            @test isapprox(dt*lxx + fdx'*Su[k+1]'*Su[k+1]*fdx + Cx[k]'*Iμ[k]*Cx[k], Wxx'*Wxx)
            @test isapprox(dt*luu + fdu'*Su[k+1]'*Su[k+1]*fdu + Cu[k]'*Iμ[k]*Cu[k], Wuu'*Wuu)
        end
        #
        if solver.opts.bp_reg_type == :state
            Wuu_reg = chol_plus(Wuu,sqrt(res.ρ[1])*I*fdu)
            Qux_reg = Qux + res.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            Wuu_reg = chol_plus(Wuu,sqrt(res.ρ[1])*Matrix(I,m,m))
            Qux_reg = Qux
        end

        #TODO find better PD check for Wuu_reg
        # # Regularization
        # if !isposdef(Hermitian(Array(Wuu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
        #     # increase regularization
        #     regularization_update!(res,solver,:increase)
        #
        #     # reset backward pass
        #     k = N-1
        #     Δv[1] = 0.
        #     Δv[2] = 0.
        #     continue
        # end

        # Compute gains
        K[k] = -Wuu_reg\(Wuu_reg'\Qux_reg)
        d[k] = -Wuu_reg\(Wuu_reg'\Qu)

        # Calculate cost-to-go
        s[k] = Qx + (K[k]'*Wuu')*(Wuu*d[k]) + K[k]'*Qu + Qux'*d[k]

        tmp1 = (Wxx')\Qux'
        tmp2 = cholesky(Wuu'*Wuu - tmp1'*tmp1).U
        Su[k][1:nn,1:nn] = Wxx + tmp1*K[k]
        Su[k][nn+1:nn+mm,1:nn] = tmp2*K[k]

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu
        Δv[2] += 0.5*d[k]'*Wuu'*Wuu*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

function chol_plus(A,B)
    n1,m = size(A)
    n2 = size(B,1)
    P = zeros(n1+n2,m)
    P[1:n1,:] = A
    P[n1+1:end,:] = B
    return qr(P).R
end

# model_dubins, obj_uncon_dubins = TrajectoryOptimization.Dynamics.dubinscar
#
# # dubins car
# u_min_dubins = [-1; -1]
# u_max_dubins = [1; 1]
# x_min_dubins = [0; -100; -100]
# x_max_dubins = [1.0; 100; 100]
# obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)
#
# # -Constrained objective
#
# model = model_dubins
# obj = obj_con_dubins
# u_max = u_max_dubins
# u_min = u_min_dubins
# Solver options

# Set up quadrotor model, objective, solver
intergrator = :rk4
dt = 0.005
N = 201
tf = 10.0
r_quad = 3.0
model, = TrajectoryOptimization.Dynamics.quadrotor
n = model.n
m = model.m

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;40.;0.] # xyz position
xf[4:7] = q0

# -control limits
u_min = 0.0
u_max = 5.0

Q = (1e-1)*Matrix(I,n,n)
Q[4,4] = 1.0
Q[5,5] = 1.0
Q[6,6] = 1.0
Q[7,7] = 1.0
R = (1.0)*Matrix(I,m,m)
Qf = (1000.0)*Matrix(I,n,n)
# obstacles constraint
r_sphere = 3.0
spheres = ((0.,10.,0.,r_sphere),(0.,20.,0.,r_sphere),(0.,30.,0.,r_sphere))
n_spheres = 3

function cI(c,x,u)
    for i = 1:n_spheres
        c[i] = sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
    end
    c
end

# unit quaternion constraint
function cE(c,x,u)
    c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,cI=cI,cE=cE)

# Solver
solver = Solver(model,obj,integration=intergrator,N=N)
solver.opts.bp_reg_initial = 1.0
U0 = ones(solver.model.m,solver.N)

results1 = init_results(solver,Array{Float64}(undef,0,0),U0)
results2 = init_results(solver,Array{Float64}(undef,0,0),U0)
rollout!(results1,solver)
rollout!(results2,solver)
update_jacobians!(results1, solver)
update_jacobians!(results2, solver)

n,m,N = get_sizes(solver)
m̄,mm = get_num_controls(solver)
n̄,nn = get_num_states(solver)
bp = BackwardPass(nn,mm,solver.N)

v1 = _backwardpass!(results1,solver,bp)
v2 = test_backwardpass_sqrt!(results2,solver,bp)

@test isapprox(v1[1:2], v2[1:2])
@test isapprox(to_array(results1.K),to_array(results2.K))
@test isapprox(to_array(results1.d),to_array(results2.d))

S_sqrt = [zeros(nn,nn) for k = 1:N]
cond_normal = zeros(N)
cond_sqrt = zeros(N)

for k = 1:N
    S_sqrt[k] = results2.S[k]'*results2.S[k]
    cond_normal[k] = cond(results1.S[k])
    cond_sqrt[k] = cond(results2.S[k])
end

@test isapprox(to_array(results1.S),to_array(S_sqrt))

plot(cond_normal,label="bp")
plot(cond_sqrt,title="Condition number",label="sqrt bp")
