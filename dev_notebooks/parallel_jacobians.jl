using BenchmarkTools
using Distributed
addprocs()
@everywhere using TrajectoryOptimization
@everywhere using SharedArrays

# Set up problem
model, obj0 = Dynamics.cartpole_analytical
n,m = model.n, model.m

obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
obj_con = to_static(obj_con)
obj = to_static(obj)
dt = 0.1

# Initialize trajectory
solver = Solver(model,obj,dt=dt,integration=:rk3_foh)
n,m,N = get_sizes(solver)
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)


solver = Solver(model,obj,dt=dt,integration=:rk3)
res = UnconstrainedVectorResults(n,m,N)
reS = UnconstrainedStaticResults(n,m,N)

rollout!(res,solver)
rollout!(reS,solver)
J_prev2 = cost(solver, res)
J_prev3 = cost(solver, reS)


@btime TrajectoryOptimization.update_jacobians!(res, solver)
@btime TrajectoryOptimization.update_jacobians!(reS, solver)


X,U = res.X,res.U
fx,fu = res.fx, res.fu

fxu = [zeros(n,n+m) for i = 1:N-1]
fx_view = [view(F,1:n,1:n) for F in fxu]
fu_view = [view(F,1:n,n.+(1:m)) for F in fxu]


function calc_dyn_jacob(fx,fu,X,U)
    N = length(X)
    for k = 1:N-1
        res.fx[k], res.fu[k] = solver.Fd(res.X[k], res.U[k])
    end
    return nothing
end

function calc_dyn_jacob_map(fx,fu,X,U)
    N = solver.N
    jacob = map((x,u)->solver.Fd(x,u),res.X,res.U)
    for k = 1:N-1
        res.fx[k],res.fu[k] = jacob[k]
    end
end

function calc_dyn_jacob_map!(fxu,X,U)
    N = solver.N
    map!((x,u)->Fd(x,u),fxu,res.X,res.U)
    fx_view = [view(F,1:n,1:n) for F in fxu]
    fu_view = [view(F,1:n,n.+(1:m)) for F in fxu]
    return fx_view,fu_view
end

Xrand = rand(n,N)
Urand = rand(m,N)

copyto!(res.X,Xrand)
copyto!(res.U,Urand)

k = rand(1:N-1)
calc_dyn_jacob(fx,fu,X,U)
A,B = fx[k],fu[k]
calc_dyn_jacob_map(fx,fu,X,U)
fx[k] == A
fu[k] == B
fx_view,fu_view = calc_dyn_jacob_map!(fxu,X,U)
fx_view[k] == fx[k]
fu_view[k] == fu[k]
Fd(X[k],U[k]) == [A B]

@btime calc_dyn_jacob(fx,fu,res.X,res.U)
@btime calc_dyn_jacob_map(fx,fu,res.X,res.U)
@btime calc_dyn_jacob_map!(fxu,res.X,res.U)




###################################
#         PARALLEL STUFF          #
###################################
N = 1000

@everywhere Fd,Fc = TrajectoryOptimization.generate_dynamics_jacobians($model,$dt,TrajectoryOptimization.rk3,:zoh)
X = SharedArray{Float64,2}((n,N))
U = SharedArray{Float64,2}((m,N))
Fxu = SharedArray{Float64,3}((n,n+m,N))

X .= Xrand
U .= Urand


# Split the timesteps betwen processors
@everywhere function split_timesteps(N::Int,idx::Int=myid()-1)
    if idx == 0
        return 1:0
    end
    nchunks = length(workers())
    split = [round(Int,s) for s in range(0,stop=N,length=nchunks+1)]
    return split[idx]+1:split[idx+1]
end

@everywhere function jacobian_chunk!(Fxu::SharedArray,X::SharedArray,U::SharedArray,inds::UnitRange)
    n,N = size(X)
    m = size(U,1)
    for k in inds
        Fxu[:,:,k] = Fd(X[:,k],U[:,k])
    end
end

@everywhere function calc_jacobian_chunks(Fxu::SharedArray,X::SharedArray,U::SharedArray)
    N = size(X,2)
    @sync begin
        for w in workers()
            @async remotecall_wait(jacobian_chunk!,w,Fxu,X,U,split_timesteps(N,w-1))
        end
    end
end
inds = split_timesteps(N,1)
jacobian_chunk!(Fxu,X,U,inds)
@time calc_jacobian_chunks(Fxu,X,U)
Fxu[1:n,1:n,k] ≈ fx[k]

TrajectoryOptimization.to_array(fx) ≈ Fxu[1:n,1:n,1:end-1]

@btime calc_jacobian_chunks(Fxu,X,U)


function calculate_jacobians_parallel!(res::ConstrainedIterResults, solver::Solver)::Nothing #TODO change to inplace '!' notation throughout the code
    N = solver.N
    for k = 1:N-1
        if solver.control_integration == :foh
            res.fx[k], res.fu[k], res.fv[k] = solver.Fd(res.X[k], res.U[k], res.U[k+1])
            res.Ac[k], res.Bc[k] = solver.Fc(res.X[k], res.U[k])
        else
            res.fx[k], res.fu[k] = solver.Fd(res.X[k], res.U[k])
        end
        solver.c_jacobian(res.Cx[k], res.Cu[k], res.X[k],res.U[k])
    end

    if solver.control_integration == :foh
        res.Ac[N], res.Bc[N] = solver.Fc(res.X[N], res.U[N])
        solver.c_jacobian(res.Cx[N], res.Cu[N], res.X[N],res.U[N])
    end

    solver.c_jacobian(res.Cx_N, res.X[N])
    return nothing
end
