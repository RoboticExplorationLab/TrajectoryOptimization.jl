using Test
using BenchmarkTools

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
dt = 0.1

# Initialize trajectory
solver = Solver(model,obj,dt=dt,integration=:rk3_foh)
n,m,N = get_sizes(solver)
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)

# Get results structures
obj_con = to_static(obj_con)
obj = to_static(obj)






#######################
#    BACKWARDS PASS   #
#######################

solver = Solver(model,obj,dt=dt,integration=:rk3)
res = UnconstrainedResults(n,m,N)
reS = UnconstrainedStaticResults(n,m,N)
reV = UnconstrainedVectorResults(n,m,N)

solver = Solver(model,obj_con,dt=dt,integration=:rk3)
res = ConstrainedResults(n,m,obj_con.p,N)
reS = ConstrainedStaticResults(n,m,obj_con.p,N)
reV = ConstrainedVectorResults(n,m,obj_con.p,N)

# println("Default")
rollout!(res,solver)
J_prev1 = cost(solver, res)
TrajectoryOptimization.calculate_jacobians!(res, solver)
update_constraints!(res, solver)
# println("Static")
rollout!(reS,solver)
J_prev2 = cost(solver, reS)
TrajectoryOptimization.calculate_jacobians!(reS, solver)
update_constraints!(reS, solver)
# println("Vecs")
rollout!(reV,solver)
J_prev3 = _cost(solver, reV)
TrajectoryOptimization.calculate_jacobians!(reV, solver)
update_constraints!(reV, solver)
to_array(reV.X) == to_array(reS.X) == res.X
to_array(reV.fx) == to_array(reS.fx) == res.fx
if res isa ConstrainedResults
    to_array(reS.C) == res.C == to_array(reV.C)
end

# Zero order hold
println("Original")
@btime Δv1 = backwardpass!(res, solver)

println("Vector Results")
@btime Δv3 = backwardpass!(reV, solver)
@btime Δv5 = backwardpass_new!(reV, solver)
# @btime Δv5 = backwardpass_new_ts!(reV, solver)

println("Static Results")
@btime Δv2 = backwardpass!(reS, solver)
@btime Δv4 = backwardpass_new!(reS, solver)
# @btime Δv4 = backwardpass_foh_new_ts!(reS, solver)



# First order hold
solver = Solver(model,obj,dt=dt,integration=:rk3_foh)
res = UnconstrainedResults(n,m,N)
reS = UnconstrainedStaticResults(n,m,N)
reV = UnconstrainedVectorResults(n,m,N)

solver = Solver(model,obj_con,dt=dt,integration=:rk3_foh)
res = ConstrainedResults(n,m,obj_con.p,N)
reS = ConstrainedStaticResults(n,m,obj_con.p,N)
reV = ConstrainedVectorResults(n,m,obj_con.p,N)

# println("Default")
rollout!(res,solver)
J_prev1 = cost(solver, res)
TrajectoryOptimization.calculate_jacobians!(res, solver)
update_constraints!(res, solver)
# println("Static")
rollout!(reS,solver)
J_prev2 = cost(solver, reS)
TrajectoryOptimization.calculate_jacobians!(reS, solver)
update_constraints!(reS, solver)
# println("Vecs")
rollout!(reV,solver)
J_prev3 = _cost(solver, reV)
TrajectoryOptimization.calculate_jacobians!(reV, solver)
update_constraints!(reV, solver)
to_array(reV.X) == to_array(reS.X) == res.X
to_array(reV.fx) == to_array(reS.fx) == res.fx
if res isa ConstrainedResults
    to_array(reS.C) == res.C == to_array(reV.C)
end


println("Original")
@btime Δv1 = backwardpass_foh!(res, solver)

println("Vector Results")
@btime Δv3 = backwardpass_foh!(reV, solver)
@btime Δv5 = backwardpass_foh_new!(reV, solver)
@btime Δv5 = backwardpass_foh_new_ts!(reV, solver)

println("Static Results")
@btime Δv2 = backwardpass_foh!(reS, solver)
@btime Δv4 = backwardpass_foh_new!(reS, solver)
@btime Δv4 = backwardpass_foh_new_ts!(reS, solver)



# Type Stability checks

# Zero order hold
Δv1 = backwardpass!(res, solver)
Δv2 = backwardpass!(reV, solver)
Δv3 = backwardpass!(reS, solver)
Δv4 = backwardpass_new!(reV, solver)
Δv5 = backwardpass_new!(reS, solver)
Δv1 ≈ Δv2 ≈ Δv3 ≈ Δv4 ≈ Δv5

@code_warntype backwardpass!(reS, solver)
@code_warntype backwardpass!(reV, solver)

@code_warntype backwardpass_new!(reS, solver)
@code_warntype backwardpass_new!(reV, solver)



Δv1 = backwardpass_foh!(res, solver)
Δv2 = backwardpass_foh_new!(reV, solver)
Δv3 = backwardpass_foh_new_ts!(reV, solver)
Δv4 = backwardpass_foh_new!(reS, solver)
Δv5 = backwardpass_foh_new_ts!(reS, solver)
Δv1 ≈ Δv2 ≈ Δv3 ≈ Δv4 ≈ Δv5

@code_warntype backwardpass_foh!(reS, solver)
@code_warntype backwardpass_foh!(reV, solver)


@code_warntype backwardpass_foh_new!(reS, solver)
@code_warntype backwardpass_foh_new!(reV, solver)

@code_warntype backwardpass_foh_new_ts!(reS, solver)
@code_warntype backwardpass_foh_new_ts!(reV, solver)







function backwardpass_new!(res::SolverVectorResults,solver::Solver)
    N = solver.N; n = solver.model.n; m = solver.model.m;
    Q = solver.obj.Q; Qf = solver.obj.Qf; xf = solver.obj.xf;
    R = getR(solver)
    dt = solver.dt

    if solver.model.m != length(res.U[1])
        m += n
    end

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s

    # Initialize expected change in cost-to-go
    S[N],s[N] = terminal_step(res,solver)
    Δv = [0.0 0.0]

    k = N-1

    # Backward pass
    while k >= 1
        fx, fu = res.fx[k], res.fu[k]
        if res isa ConstrainedIterResults
            Qfun = cost_to_go_expansion(S[k+1],s[k+1],X[k],U[k],fx,fu,Q,R,xf,dt,res.ρ[1],
                res.C[k],res.Cx[k],res.Cu[k],res.LAMBDA[k],res.Iμ[k])
        else
            Qfun = cost_to_go_expansion(S[k+1],s[k+1],X[k],U[k],fx,fu,Q,R,xf,dt,res.ρ[1])
        end

        K[k],d[k],S[k],s[k],Δv_,reg = calculate_optimal_gains(Qfun)
        Δv += Δv_

        if reg
            if solver.opts.verbose
                println("regularized (normal bp)")
            end
            regularization_update!(res,solver,true)
            k = N-1
            Δv = [0.0 0.0]
            S[N],s[N] = terminal_step(res,solver)
            continue
        end

        k = k - 1;
    end

    return Δv
end

function backwardpass_foh_new!(res::SolverVectorResults,solver::Solver)
    n,m,N = get_sizes(solver)
    dt = solver.dt

    if solver.model.m != length(res.U[1])
        m += n
    end

    Q = solver.obj.Q
    Qf = solver.obj.Qf
    xf = solver.obj.xf
    R = getR(solver)

    K = res.K
    b = res.b
    d = res.d

    X,U = res.X, res.U


    # Initialization
    S,s = terminal_step(res,solver)
    Δv = [0. 0.]

    k = N-1
    while k >= 1

        # Extract Variables from results
        Ac1, Bc1 = res.Ac[k], res.Bc[k]
        Ac2, Bc2 = res.Ac[k+1], res.Bc[k+1]
        Ad, Bd, Cd = res.fx[k], res.fu[k], res.fv[k]

        xdot1 = res.xdot[k]
        xdot2 = res.xdot[k+1]

        # Calculate the Midpoint
        x,u = X[k], U[k]
        xm = 0.5*X[k] + dt/8.0*xdot1 + 0.5*X[k+1] - dt/8.0*xdot2
        um = (U[k] + U[k+1])/2.0
        y,v = X[k+1],U[k+1]


        # Get second order Taylor series of the Lagrangian
        if res isa ConstrainedIterResults
            Qfun = cost_to_go_expansion(S,s,x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,res.ρ[1],
                res.C[k+1],res.Cx[k+1],res.Cu[k+1],res.LAMBDA[k+1],res.Iμ[k+1])
        else
            Qfun = cost_to_go_expansion(S,s,x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,res.ρ[1])
        end

        # Calculate the optimal control gains
        K[k+1],b[k+1],d[k+1],Δv_,reg = calculate_optimal_gains!(S,s,Qfun,res.ρ[1])
        Δv += Δv_

        # Check regularization flag
        if reg
            if solver.opts.verbose
                println("*NOT implemented* regularized (foh bp)")
            end
            regularization_update!(res,solver,true)
            k = N-1
            S,s = terminal_step(res,solver)
            Δv = [0. 0.]
            continue
        end


        # at last time step, optimize over final control
        if k == 1
            # Calculate gains at first time step
            if res isa ConstrainedIterResults
                K[k],b[k],d[k],Δv_,reg = calculate_optimal_gains!(S,s,res.C[k],res.Cx[k],res.Cu[k],res.LAMBDA[k],res.Iμ[k])
            else
                K[k],b[k],d[k],Δv_,reg = calculate_optimal_gains!(S,s)
            end

            # Check regularization flag
            if reg
                if solver.opts.verbose
                    println("regularized (foh bp)")
                end
                regularization_update!(res,solver::Solver,true)
                k = N-1
                S,s = terminal_step(res,solver)
                Δv = [0. 0.]
                continue
            end

            res.s[1] = s[1:n]
            Δv += Δv_
        end

        k = k - 1;
    end

    return Δv
end

function backwardpass_foh_new_ts!(res,solver)
    n,m,N = get_sizes(solver)
    dt = solver.dt

    if solver.model.m != length(res.U[1])
        m += n
    end

    Q = solver.obj.Q
    Qf = solver.obj.Qf
    xf = solver.obj.xf
    R = getR(solver)

    K = res.K
    b = res.b
    d = res.d

    X,U = res.X, res.U


    # Initialization
    S,s = terminal_step(res,solver)
    Δv = [0. 0.]

    k = N-1
    while k >= 1
        # Extract Variables from results
        Ac1, Bc1 = res.Ac[k], res.Bc[k]
        Ac2, Bc2 = res.Ac[k+1], res.Bc[k+1]
        Ad, Bd, Cd = res.fx[k], res.fu[k], res.fv[k]

        xdot1 = res.xdot[k]
        xdot2 = res.xdot[k+1]

        # Calculate the Midpoint
        x,u = X[k], U[k]
        xm = 0.5*X[k] + dt/8.0*xdot1 + 0.5*X[k+1] - dt/8.0*xdot2
        um = (U[k] + U[k+1])/2.0
        y,v = X[k+1],U[k+1]


        # Get second order Taylor series of the Lagrangian
        if res isa ConstrainedIterResults
            Qfun = cost_to_go_expansion(S,s,x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,res.ρ[1],
            res.C[k+1],res.Cx[k+1],res.Cu[k+1],res.LAMBDA[k+1],res.Iμ[k+1])
        else
            Qfun = cost_to_go_expansion(S,s,x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,res.ρ[1])
        end

        # Calculate the optimal control gains
        K[k+1],b[k+1],d[k+1],Δv_,reg = calculate_optimal_gains!(S,s,Qfun,res.ρ[1])
        Δv += Δv_

        # Check regularization flag
        if reg
            if solver.opts.verbose
                println("*NOT implemented* regularized (foh bp)")
            end
            regularization_update!(res,solver,true)
            k = N-1
            S,s = terminal_step(res,solver)
            Δv = [0. 0.]
            continue
        end

        # at last time step, optimize over final control
        if k == 1
            # Calculate gains at first time step
            if res isa ConstrainedIterResults
                Qbar = S_to_Qbar(S,s,res.C[k],res.Cx[k],res.Cu[k],res.LAMBDA[k],res.Iμ[k])
            else
                Qbar = S_to_Qbar(S,s,n,m)
            end
            K[k],b[k],d[k],Δv_,reg = calculate_optimal_gains!(S,s,Qbar)

            # Check regularization flag
            if reg
                if solver.opts.verbose
                    println("regularized (foh bp)")
                end
                regularization_update!(res,solver::Solver,true)
                k = N-1
                S,s = terminal_step(res,solver)
                Δv = [0. 0.]
                continue
            end

            res.s[1] = s[1:n]
            Δv += Δv_
        end

        k = k - 1;

    end

    return Δv
end


"""
$(SIGNATURES)
Calculate the cost to go at the terminal time step
"""
function terminal_step(res::SolverVectorResults,solver::Solver)::Tuple{Matrix{Float64},Vector{Float64}}
    n,m,N = get_sizes(solver)
    Qf = solver.obj.Qf
    xf = solver.obj.xf
    if solver.control_integration == :foh
        # Initialization
        S = zeros(n+m,n+m)
        s = zeros(n+m)

        # Boundary conditions
        S[1:n,1:n] = Qf
        s[1:n] = Qf*(res.X[N]-xf)

        # Terminal constraints
        if res isa ConstrainedIterResults
            C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
            CxN = res.Cx_N
            S[1:n,1:n] += CxN'*res.IμN*CxN
            s[1:n] += CxN'*res.IμN*res.CN + CxN'*res.λN
        end
    elseif solver.control_integration == :zoh

        # Boundary Conditions
        S = Array(Qf)
        s = Array(Qf*(res.X[N] - xf))

        # Terminal constraints
        if res isa ConstrainedIterResults
            C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
            CxN = res.Cx_N
            S += CxN'*res.IμN*CxN
            s += CxN'*res.IμN*res.CN + CxN'*res.λN
        end
    end
    return S,s
end


#########################################
#        COST TO GO EXPANSIONS          #
#########################################
"""
$(SIGNATURES)
Perform a 2nd order taylor series expansion of the the stage cost for first
    order hold
"""
function _lagrangian_2nd_order(S::Matrix{Float64},s::Vector{Float64},x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,ρ)
    # Expansion of stage cost L(x,u,y,v) -> dL(dx,du,dy,dv)
    Lx = dt/6*Q*(x - xf) + 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(xm - xf)
    Lu = dt/6*R*u + 4*dt/6*((dt/8*Bc1)'*Q*(xm - xf) + 0.5*R*um)
    Ly = dt/6*Q*(y - xf) + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(xm - xf)
    Lv = dt/6*R*v + 4*dt/6*((-dt/8*Bc2)'*Q*(xm - xf) + 0.5*R*um)

    Lxx = dt/6*Q + 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 + dt/8*Ac1)
    Luu = dt/6*R + 4*dt/6*((dt/8*Bc1)'*Q*(dt/8*Bc1) + 0.5*R*0.5)
    Lyy = dt/6*Q + 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(I/2 - dt/8*Ac2)
    Lvv = dt/6*R + 4*dt/6*((-dt/8*Bc2)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)

    Lxu = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(dt/8*Bc1)
    Lxy = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(I/2 - dt/8*Ac2)
    Lxv = 4*dt/6*(I/2 + dt/8*Ac1)'*Q*(-dt/8*Bc2)
    Luy = 4*dt/6*(dt/8*Bc1)'*Q*(I/2 - dt/8*Ac2)
    Luv = 4*dt/6*((dt/8*Bc1)'*Q*(-dt/8*Bc2) + 0.5*R*0.5)
    Lyv = 4*dt/6*(I/2 - dt/8*Ac2)'*Q*(-dt/8*Bc2)

    # Unpack cost-to-go P, then add L + P
    Sy = s[1:n]
    Sv = s[n+1:n+m]
    Syy = S[1:n,1:n]
    Svv = S[n+1:n+m,n+1:n+m]
    Syv = S[1:n,n+1:n+m]

    Ly += Sy
    Lv += Sv
    Lyy += Syy
    Lvv += Svv
    Lyv += Syv

    TX,TU = typeof(Lx),typeof(Lu)
    TXX,TUU,TXU = typeof(Lxx),typeof(Luu),typeof(Lxu)

    return Lx,Lu,Ly::TX,Lv::TU, Lxx,Luu,Lyy::TXX,Lvv::TUU,Lxu,Lxy,Lxv,Luy,Luv,Lyv::TXU
end

"""
$(SIGNATURES)
Find the 2nd order taylor series expansion of the cost to go at the current
    time step
"""
# ZERO ORDER HOLD
function cost_to_go_expansion(S,s,x,u,fx,fu,Q,R,xf,dt,ρ)
    Qx,Qu,Qxx,Quu,Qux = _cost_to_go_expansion(S,s,x,u,fx,fu,Q,R,xf,dt,ρ)
    return (x=Qx,u=Qu, xx=Qxx,uu=Quu,ux=Qux)
end

# (with constraints)
function cost_to_go_expansion(S,s,x,u,fx,fu,Q,R,xf,dt,ρ,
        C,Cx,Cu,λ,Iμ)
    Qx,Qu,Qxx,Quu,Qux = _cost_to_go_expansion(S,s,x,u,fx,fu,Q,R,xf,dt,ρ)

    Qx +=  Cx'Iμ*C + Cx'λ
    Qu +=  Cu'Iμ*C + Cu'λ
    Qxx += Cx'Iμ*Cx
    Quu += Cu'Iμ*Cu
    Qux += Cu'Iμ*Cx

    return (x=Qx,u=Qu, xx=Qxx,uu=Quu,ux=Qux)
end

function _cost_to_go_expansion(S,s,x,u,fx,fu,Q,R,xf,dt,ρ)
    lx = dt*Q*(x - xf)
    lu = dt*R*u
    lxx = dt*Q
    luu = dt*R

    # Gradients and Hessians of Taylor Series Expansion of Q
    Qx = lx + fx's
    Qu = lu + fu's
    Qxx = lxx + fx'S*fx

    Quu = luu + fu'S*fu + ρ*I
    Qux = fu'S*fx

    return Qx,Qu,Qxx,Quu,Qux
end


# FIRST ORDER HOLD
function cost_to_go_expansion(S,s,x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,ρ)

    Lx,Lu,Ly,Lv, Lxx,Luu,Lyy,Lvv, Lxu,Lxy,Lxv,Luy,Luv,Lyv = _lagrangian_2nd_order(S,s,
        x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,ρ)

    # Substitute in discrete dynamics dx = (Ad)dx + (Bd)du1 + (Cd)du2
    Qx = vec(Lx) + Ad'*vec(Ly)
    Qu = vec(Lu) + Bd'*vec(Ly)
    Qv = vec(Lv) + Cd'*vec(Ly)

    Qxx = Lxx + Lxy*Ad + Ad'Lxy' + Ad'Lyy*Ad
    Quu = Luu + Luy*Bd + Bd'Luy' + Bd'Lyy*Bd + ρ*I
    Qvv = Lvv + Lyv'Cd + Cd'Lyv  + Cd'Lyy*Cd + ρ*I
    Qxu = Lxu + Lxy*Bd + Ad'Luy' + Ad'Lyy*Bd
    Qxv = Lxv + Lxy*Cd + Ad'Lyv  + Ad'Lyy*Cd
    Quv = Luv + Luy*Cd + Bd'Lyv  + Bd'Lyy*Cd

    # return (x=Qx,u=Qu,v=Qv, xx=Qxx,uu=Quu,vv=Qvv, xu=Qxu,xv=Qxv,uv=Quv)
    return Qx,Qu,Qv, Qxx,Quu,Qvv, Qxu,Qxv,Quv
end

# (with constraints)
function cost_to_go_expansion(S,s,x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,ρ,
        C,Cy,Cv,LAMBDA,Iμ)

    Lx,Lu,Ly,Lv, Lxx,Luu,Lyy,Lvv, Lxu,Lxy,Lxv,Luy,Luv,Lyv = _lagrangian_2nd_order(S,s,
        x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,ρ)

    Ly += (Cy'*Iμ*C + Cy'*LAMBDA)
    Lv += (Cv'*Iμ*C + Cv'*LAMBDA)
    Lyy += Cy'*Iμ*Cy
    Lvv += Cv'*Iμ*Cv
    Lyv += Cy'*Iμ*Cv

    # Substitute in discrete dynamics dx = (Ad)dx + (Bd)du1 + (Cd)du2
    Qx = vec(Lx) + Ad'*vec(Ly)
    Qu = vec(Lu) + Bd'*vec(Ly)
    Qv = vec(Lv) + Cd'*vec(Ly)

    Qxx = Lxx + Lxy*Ad + Ad'Lxy' + Ad'Lyy*Ad
    Quu = Luu + Luy*Bd + Bd'Luy' + Bd'Lyy*Bd + ρ*I
    Qvv = Lvv + Lyv'Cd + Cd'Lyv  + Cd'Lyy*Cd + ρ*I
    Qxu = Lxu + Lxy*Bd + Ad'Luy' + Ad'Lyy*Bd
    Qxv = Lxv + Lxy*Cd + Ad'Lyv  + Ad'Lyy*Cd
    Quv = Luv + Luy*Cd + Bd'Lyv  + Bd'Lyy*Cd

    # return (x=Qx,u=Qu,v=Qv, xx=Qxx,uu=Quu,vv=Qvv, xu=Qxu,xv=Qxv,uv=Quv)
    return Qx,Qu,Qv, Qxx,Quu,Qvv, Qxu,Qxv,Quv
end



#############################################
#     CALCULATE OPTIMAL CONTROL GAINS       #
#############################################
"""
$(SIGNATURES)
Calculate the optimal control gains given the 2nd order expansion of the cost-to-go
"""
# FIRST ORDER HOLD
function calculate_optimal_gains(Q::NamedTuple)
    # regularization
    if !isposdef(Hermitian(Array(Q.uu)))  # need to wrap Array since isposdef doesn't work for static arrays
        K = zeros(Q.ux)
        d = zeros(Q.u)
        S = zeros(Q.xx)
        s = zeros(Q.x)

        reg = true
    else
        # Compute gains
        K = Q.uu\Q.ux
        d = Q.uu\Q.u
        S = Q.xx - Q.ux'K
        s = Q.x - Q.ux'd

        reg = false
    end

    Δv = [vec(Q.u)'*vec(d) 0.5*vec(d)'*Q.uu*vec(d)]
    return K,d,S,s,Δv,reg
end

# SECOND ORDER HOLD
function calculate_optimal_gains!(S,s,Q::Tuple,ρ::Float64)
    Qx,Qu,Qv, Qxx,Quu,Qvv, Qxu,Qxv,Quv = Q
    n = length(Qx)
    m = length(Qu)

    # Qvv = Hermitian(Qvv)
    # regularization
    if !isposdef(Array(Qvv))
        K = zeros(Qxv')
        b = zeros(Quv')
        d = zeros(Qv)
        Δv = [0. 0.]
        reg = true
        return K,b,d,Δv,reg
    else
        reg = false
    end

    K = -Qvv\Qxv'
    b = -Qvv\Quv'
    d = -Qvv\vec(Qv)

    Qx_ = vec(Qx) + K'*vec(Qv) + Qxv*vec(d) + K'Qvv*d
    Qu_ = vec(Qu) + b'*vec(Qv) + Quv*vec(d) + b'*Qvv*d
    Qxx_ = Qxx + Qxv*K + K'*Qxv' + K'*Qvv*K
    Quu_ = Quu + Quv*b + b'*Quv' + b'*Qvv*b + ρ*I
    Qxu_ = Qxu + K'*Quv' + Qxv*b + K'*Qvv*b

    # cache (approximate) cost-to-go at timestep k
    s[1:n] = Qx_
    s[n+1:n+m] = Qu_
    S[1:n,1:n] = Qxx_
    S[n+1:n+m,n+1:n+m] = Quu_
    S[1:n,n+1:n+m] = Qxu_
    S[n+1:n+m,1:n] = Qxu_'

    # line search terms
    Δv = [-vec(Qv)'vec(d) 0.5*vec(d)'Qvv*vec(d)]

    return K,b,d,Δv,reg
end

# (Final time step)
function calculate_optimal_gains!(S,s,Q::NTuple{4,Any})
    Qx_,Qu_,Quu_,Qxu_ = Q
    n = length(Qx_)
    m = length(Qu_)

    if !isposdef(Array(Quu_))
        K = zero(Qxu_')
        b = zero(Quu_)
        d = zero(Qu_)
        Δv = [0. 0.]
        reg = true
        return K,b,d,Δv,reg
    else
        reg = false
    end

    K = -Quu_\Qxu_'
    b = zero(Quu_)
    d = -Quu_\vec(Qu_)

    s[1:n] = Qx_ + Qxu_*vec(d)

    Δv = [-vec(Qu_)'*vec(d) 0.5*vec(d)'*Quu_*vec(d)]

    return K,b,d,Δv,reg
end

"""
$(SIGNATURES)
Calculate the 2nd order expansion of the final (k=1) cost-to-go for first order hold case
"""
function S_to_Qbar(S::Matrix{Float64},s::Vector{Float64},n::Int,m::Int)
    Qx_ = s[1:n]
    Qu_ = s[n+1:n+m]
    Quu_ = S[n+1:n+m,n+1:n+m]
    Qxu_ = S[n+1:n+m,1:n]'
    return Qx_,Qu_,Quu_,Qxu_
end

# (with constraints)
function S_to_Qbar(S::Matrix{Float64},s::Vector{Float64},C,Cx,Cu,LAMBDA,Iμ)
    n = size(Cx,2)
    m = size(Cu,2)

    Qx_ = s[1:n]
    Qu_ = s[n+1:n+m]
    Quu_ = S[n+1:n+m,n+1:n+m]
    Qxu_ = S[n+1:n+m,1:n]'

    Qx_ += (Cx'*Iμ*C + Cx'*LAMBDA)
    Qu_ += (Cu'*Iμ*C + Cu'*LAMBDA)
    Quu_ += Cu'*Iμ*Cu
    Qxu_ += Cx'*Iμ*Cu

    return Qx_,Qu_,Quu_,Qxu_
end




# function cost_to_go_expansion(S,s,x::MVector{N,Float64},u::MVector{M,Float64},xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,ρ) where {N,M}
#     println("Static version")
#     Lx,Lu,Ly,Lv, Lxx,Luu,Lyy,Lvv, Lxu,Lxy,Lxv,Luy,Luv,Lyv = _lagrangian_2nd_order(S,s,
#         x,u,xm,um,y,v,Ac1,Bc1,Ac2,Bc2,Ad,Bd,Cd,Q,R,xf,dt,ρ)
#
#     # Substitute in discrete dynamics dx = (Ad)dx + (Bd)du1 + (Cd)du2
#     Qx = vec(Lx) + Ad'*vec(Ly)
#     Qu = vec(Lu) + Bd'*vec(Ly)
#     Qv = vec(Lv) + Cd'*vec(Ly)
#
#     Qxx = Lxx + Lxy*Ad + Ad'Lxy' + Ad'Lyy*Ad
#     Quu = Luu + Luy*Bd + Bd'Luy' + Bd'Lyy*Bd + ρ*I
#     Qvv = Lvv + Lyv'Cd + Cd'Lyv  + Cd'Lyy*Cd + ρ*I
#     Qxu = Lxu + Lxy*Bd + Ad'Luy' + Ad'Lyy*Bd
#     Qxv = Lxv + Lxy*Cd + Ad'Lyv  + Ad'Lyy*Cd
#     Quv = Luv + Luy*Cd + Bd'Lyv  + Bd'Lyy*Cd
#
#     # return (x=Qx,u=Qu,v=Qv, xx=Qxx,uu=Quu,vv=Qvv, xu=Qxu,xv=Qxv,uv=Quv)
#     return Qx::SVector{N,Float64},Qu::SVector{M,Float64},Qv::SVector{M,Float64},
#         Qxx::SMatrix{N,N,Float64},Quu::SMatrix{M,M,Float64},Qvv::SMatrix{M,M,Float64}, Qxu::SMatrix{N,M,Float64},Qxv::SMatrix{N,M,Float64},Quv::SMatrix{M,M,Float64}
# end

# function cost_to_go_expansion(S,s,x::MVector{N,Float64},u::MVector{M,Float64},fx,fu,Q,R,xf,dt,ρ) where {N,M}
#     Qx,Qu,Qxx,Quu,Qux = _cost_to_go_expansion(S,s,x,u,fx,fu,Q,R,xf,dt,ρ)
#     return (x=Qx,u=Qu, xx=Qxx,uu=Quu,ux=Qux)
# end


# function calculate_optimal_gains!(S,s,Q::NamedTuple,ρ::Float64)
#     n = length(Q.x)
#     m = length(Q.u)
#
#     # Qvv = Hermitian(Qvv)
#     # regularization
#     if !isposdef(Array(Q.vv))
#         K = zeros(Q.xv')
#         b = zeros(Q.uv')
#         d = zeros(Q.v)
#         Δv = [0. 0.]
#         reg = true
#         return K,b,d,Δv,reg
#     else
#         reg = false
#     end
#
#     K = -Q.vv\Q.xv'
#     b = -Q.vv\Q.uv'
#     d = -Q.vv\vec(Q.v)
#
#     Qx_ = vec(Q.x) + K'*vec(Q.v) + Q.xv*vec(d) + K'Q.vv*d
#     Qu_ = vec(Q.u) + b'*vec(Q.v) + Q.uv*vec(d) + b'*Q.vv*d
#     Qxx_ = Q.xx + Q.xv*K + K'*Q.xv' + K'*Q.vv*K
#     Quu_ = Q.uu + Q.uv*b + b'*Q.uv' + b'*Q.vv*b + ρ*I
#     Qxu_ = Q.xu + K'*Q.uv' + Q.xv*b + K'*Q.vv*b
#
#     # cache (approximate) cost-to-go at timestep k
#     s[1:n] = Qx_
#     s[n+1:n+m] = Qu_
#     S[1:n,1:n] = Qxx_
#     S[n+1:n+m,n+1:n+m] = Quu_
#     S[1:n,n+1:n+m] = Qxu_
#     S[n+1:n+m,1:n] = Qxu_'
#
#     # line search terms
#     Δv = [-vec(Q.v)'vec(d) 0.5*vec(d)'Q.vv*vec(d)]
#
#     return K,b,d,Δv,reg
# end
