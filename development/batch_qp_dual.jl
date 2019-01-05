using Test

function generate_batch_qp_dual()
    nothing
end

u_bound = 3.
model, obj = TrajectoryOptimization.Dynamics.pendulum!
obj = Dynamics.pendulum_constrained[2]
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false

solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
U = zeros(model.m,solver.N)
results_c, = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)


costfun = solver.obj.cost
x = results_.X[1]
u = results_c.U[1]
expansion = taylor_expansion(costfun,x,u)
Q,R,H,q,r = taylor_expansion(costfun,x,u)
H

n = solver.model.n
m = solver.model.m
nm = n+m
N = solver.N
p,pI,pE = get_num_constraints(solver)

Nz = n*N + m*(N-1) # number of decision variables x, u
Nu = m*(N-1) # number of control decision variables u

Np = p*(N-1) + obj.p_N # number of constraints

Mz = N + N-1 # number of decision vectors
Mu = N-1 # number of control decision vectors

B̄ = zeros(Nz,Nu)
Ā = zeros(Nz,n)

Q̄ = zeros(Nz,Nz)
q̄ = zeros(Nz)

C̄ = zeros(Np,Nz)
c̄ = zeros(Np)

costfun = solver.obj.cost
results = results_c

for k = 1:N
    x = results.X[k]
    u = results.U[k]

    if k != N
        Q,R,H,q,r = taylor_expansion(costfun,x,u)
    else
        Qf,qf = taylor_expansion(costfun,x)
    end

    # Indices
    idx = ((k-1)*nm + 1):k*nm
    idx2 = ((k-1)*p + 1):k*p
    idx3 = ((k-1)*nm + 1):((k-1)*nm + n)
    idx4 = ((k-1)*nm + n + 1):k*nm
    idx5 = ((k-1)*m + 1):k*m

    # Calculate Ā
    k == 1 ? Ā[idx3,1:n] = 1.0*Matrix(I,n,n) : Ā[idx3,1:n] = prod(results.fdx[1:k-1])

    # Calculate B̄
    if k > 1
        for j = 1:k-1
            idx6 = ((j-1)*m + 1):j*m
            j == k-1 ? B̄[idx3,idx6] = results.fdu[k-1][1:n,1:m] : B̄[idx3,idx6] = prod(results.fdx[j+1:(k-1)])*results.fdu[k-1]
        end
    end

    # Calculate Q̄, q̄, C̄, c̄
    if k != N
        Q̄[idx,idx] = [Q H';H R]
        q̄[idx] = [q;r]

        G = results.Cx[k]
        H = results.Cu[k]
        C̄[idx2,idx] = [G H]
        c̄[idx2] = results.C[k]

        B̄[idx4,idx5] = 1.0*Matrix(I,m,m)
    else
        idx = ((k-1)*nm + 1):Nz
        Q̄[idx,idx] = Qf
        q̄[idx] = qf

        idx2 = ((k-1)*p + 1):Np
        C̄[idx2,idx] = results.Cx_N
        c̄[idx2] = results.CN
    end

end

# Tests
@test Ā[1:n,:] == Matrix(I,n,n)
k = 2
@test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:1])
k = 7
@test Ā[(k-1)*nm+1:(k-1)*nm+n,:] == prod(results.fdx[1:k-1])
@test Ā[(N-1)*nm+1:Nz,:] == prod(results.fdx[1:N-1])

@test B̄[1:n,1:n] == zeros(n,n)
@test B̄[n+1:nm,1:m] == 1.0*Matrix(I,m,m)
@test B̄[nm+1:nm+n,1:m] == results.fdu[1][1:n,1:m]
@test B̄[nm+n+1:2*nm,m+1:2*m] == 1.0*Matrix(I,m,m)
@test B̄[(N-1)*nm+1:Nz,1:m] == prod(results.fdx[2:N-1])*results.fdu[1]

k = 1
Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
@test Q̄[1:nm,1:nm] == [Q H'; H R]
@test q̄[1:nm] == [q;r]

k = 13
Q,R,H,q,r = taylor_expansion(costfun,results.X[k],results.U[k])
@test Q̄[(k-1)*nm+1:k*nm,(k-1)*nm+1:k*nm] == [Q H'; H R]
@test q̄[(k-1)*nm+1:k*nm] == [q;r]

k = N
Qf,qf = taylor_expansion(costfun,results.X[k])
@test Q̄[(k-1)*nm+1:Nz,(k-1)*nm+1:Nz] == Qf
@test q̄[(k-1)*nm+1:Nz] == qf
