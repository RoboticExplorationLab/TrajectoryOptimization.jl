abstract type BackwardPass end

struct BackwardPassZOH <: BackwardPass
    Qx::Vector{Vector{Float64}}
    Qu::Vector{Vector{Float64}}
    Qxx::Vector{Matrix{Float64}}
    Qux::Vector{Matrix{Float64}}
    Quu::Vector{Matrix{Float64}}

    function BackwardPassZOH(n::Int,m::Int,N::Int)
        Qx = [zeros(n) for i = 1:N-1]
        Qu = [zeros(m) for i = 1:N-1]
        Qxx = [zeros(n,n) for i = 1:N-1]
        Qux = [zeros(m,n) for i = 1:N-1]
        Quu = [zeros(m,m) for i = 1:N-1]

        new(Qx,Qu,Qxx,Qux,Quu)
    end
end

struct BackwardPassFOH <: BackwardPass
    Qx::Vector{Float64}
    Qu::Vector{Float64}
    Qy::Vector{Float64}
    Qv::Vector{Float64}

    Qxx::Matrix{Float64}
    Quu::Matrix{Float64}
    Qyy::Matrix{Float64}
    Qvv::Matrix{Float64}

    Qxu::Matrix{Float64}
    Qxy::Matrix{Float64}
    Qxv::Matrix{Float64}
    Quy::Matrix{Float64}
    Quv::Matrix{Float64}
    Qyv::Matrix{Float64}

    Lx::Vector{Float64}
    Lu::Vector{Float64}
    Ly::Vector{Float64}
    Lv::Vector{Float64}

    Lxx::Matrix{Float64}
    Luu::Matrix{Float64}
    Lyy::Matrix{Float64}
    Lvv::Matrix{Float64}

    Lxu::Matrix{Float64}
    Lxy::Matrix{Float64}
    Lxv::Matrix{Float64}
    Luy::Matrix{Float64}
    Luv::Matrix{Float64}
    Lyv::Matrix{Float64}

    function BackwardPassFOH(n::Int,m::Int)
        Qx = zeros(n)
        Qu = zeros(m)
        Qy = zeros(n)
        Qv = zeros(m)

        Qxx = zeros(n,n)
        Quu = zeros(m,m)
        Qyy = zeros(n,n)
        Qvv = zeros(m,m)

        Qxu = zeros(n,m)
        Qxy = zeros(n,n)
        Qxv = zeros(n,m)
        Quy = zeros(m,n)
        Quv = zeros(m,m)
        Qyv = zeros(n,m)

        Lx = zeros(n)
        Lu = zeros(m)
        Ly = zeros(n)
        Lv = zeros(m)

        Lxx = zeros(n,n)
        Luu = zeros(m,m)
        Lyy = zeros(n,n)
        Lvv = zeros(m,m)

        Lxu = zeros(n,m)
        Lxy = zeros(n,n)
        Lxv = zeros(n,m)
        Luy = zeros(m,n)
        Luv = zeros(m,m)
        Lyv = zeros(n,m)

        new(Qx,Qu,Qy,Qv,Qxx,Quu,Qyy,Qvv,Qxu,Qxy,Qxv,Quy,Quv,Qyv,Lx,Lu,Ly,Lv,Lxx,Luu,Lyy,Lvv,Lxu,Lxy,Lxv,Luy,Luv,Lyv)
    end
end

"""
$(SIGNATURES)
Solve the dynamic programming problem, starting from the terminal time step
Computes the gain matrices K and d by applying the principle of optimality at
each time step, solving for the gradient (s) and Hessian (S) of the cost-to-go
function. Also returns parameters Δv for line search (see Synthesis and Stabilization of Complex Behaviors through
Online Trajectory Optimization)
"""
function backwardpass!(results::SolverVectorResults,solver::Solver,bp::BackwardPass)
    if solver.control_integration == :foh
        Δv = _backwardpass_foh!(results,solver)
    elseif solver.opts.square_root
        Δv = _backwardpass_sqrt!(results, solver) #TODO option to help avoid ill-conditioning [see algorithm xx]
    else
        Δv = _backwardpass!(results, solver, bp)
    end

    return Δv
end

function _backwardpass!(res::SolverVectorResults,solver::Solver,bp::BackwardPass)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf

    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d;

    S = res.S; s = res.s

    bp = BackwardPassZOH(n,mm,N)
    Qx = bp.Qx; Qu = bp.Qu; Qxx = bp.Qxx; Qux = bp.Qux; Quu = bp.Quu

    # Boundary Conditions
    S[N] = Wf
    s[N] = Wf*(X[N] - xf)

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Terminal constraints
    if solver.opts.constrained
        gs = res.gs; gc = res.gc; hs = res.hs; hc = res.hc
        λs = res.λs; λc = res.λc; κs = res.κs; κc = res.κc
        Iμs = res.Iμs; Iμc = res.Iμc; Iνs = res.Iνs; Iνc = res.Iνc
        gsx = res.gsx; gcu = res.gcu; hsx = res.hsx; hcu = res.hcu

        S[N] += gsx[N]'*Iμs[N]*gsx[N] + hsx[N]'*Iνs[N]*hsx[N]
        s[N] += gsx[N]'*(Iμs[N]*gs[N] + λs[N]) + hsx[N]'*(Iνs[N]*hs[N] + κs[N])
    end

    # Backward pass
    k = N-1

    # for the special case of the tracking quadratic cost:
    ℓxx = W
    ℓuu = R
    ℓux = zeros(m,n)

    while k >= 1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        x = X[k]
        u = U[k][1:m]

        # for the special case of the tracking quadratic cost:
        ℓ1 = ℓ(x,u,W,R,xf)
        ℓx = W*(x - xf)
        ℓu = R*u

        # Assemble expansion
        Qx[k][1:n] = dt*ℓx
        Qu[k][1:m] = dt*ℓu
        Qxx[k][1:n,1:n] = dt*ℓxx
        Quu[k][1:m,1:m] = dt*ℓuu
        Qux[k][1:m,1:n] = dt*ℓux

        # Minimum time expansion components
        if solver.opts.minimum_time
            h = U[k][m̄]

            Qu[k][m̄] = 2*h*(ℓ1 + R_minimum_time)

            tmp = 2*h*ℓu
            Quu[k][1:m,m̄] = tmp
            Quu[k][m̄,1:m] = tmp'
            Quu[k][m̄,m̄] = 2*(ℓ1 + R_minimum_time)

            Qux[k][m̄,1:n] = 2*h*ℓx'
        end

        # Infeasible expansion components
        if solver.opts.infeasible
            Qu[k][m̄+1:mm] = R_infeasible*U[k][m̄+1:m̄+n]
            Quu[k][m̄+1:mm,m̄+1:mm] = R_infeasible
        end

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Constraints
        if solver.opts.constrained
            k != 1 ? Qx[k] += gsx[k]'*(Iμs[k]*gs[k] + λs[k]) + hsx[k]'*(Iνs[k]*hs[k] + κs[k]) : nothing
            Qu[k] += gcu[k]'*(Iμc[k]*gc[k] + λc[k]) + hcu[k]'*(Iνc[k]*hc[k] + κc[k])
            k != 1 ? Qxx[k] += gsx[k]'*Iμs[k]*gsx[k] + hsx[k]'*Iνs[k]*hsx[k] : nothing
            Quu[k] += gcu[k]'*Iμc[k]*gcu[k] + hcu[k]'*Iνc[k]*hcu[k]
            # Qux[k] += # no coupling between constraints
        end

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx[k] += fdx'*s[k+1]
        Qu[k] += fdu'*s[k+1]
        Qxx[k] += fdx'*S[k+1]*fdx
        Quu[k] += fdu'*S[k+1]*fdu
        Qux[k] += fdu'*S[k+1]*fdx

        if solver.opts.regularization_type == :state #TODO combined into one
            Quu_reg = Quu[k] + res.ρ[1]*fdu'*fdu
            Qux_reg = Qux[k] + res.ρ[1]*fdu'*fdx
        elseif solver.opts.regularization_type == :control
            Quu_reg = Quu[k] + res.ρ[1]*I
            Qux_reg = Qux[k]
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            regularization_update!(res,solver,:increase)

            # reset backward pass
            k = N-1
            Δv[1] = 0.
            Δv[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -Quu_reg\Qux_reg
        d[k] = -Quu_reg\Qu[k]

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Qx[k] + K[k]'*Quu[k]*d[k] + K[k]'*Qu[k] + Qux[k]'*d[k]
        S[k] = Qxx[k] + K[k]'*Quu[k]*K[k] + K[k]'*Qux[k] + Qux[k]'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu[k]
        Δv[2] += 0.5*d[k]'*Quu[k]*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end

function _backwardpass_foh!(results::SolverVectorResults,solver::Solver)
    # Problem dimensions
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    # Objective parameters
    W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf

    dt = solver.dt

    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    # Pull out results
    X = results.X; U = results.U; K = results.K; b = results.b; d = results.d; s = results.s; S = results.S
    L = results.L; l = results.l

    # Useful linear indices
    idx = Array(1:2*(n+mm))
    Idx = reshape(Array(1:(n+mm)^2),n+mm,n+mm)

    x_idx = idx[1:n]
    u_idx = idx[n+1:n+m]
    ū_idx = idx[n+1:n+mm]
    y_idx = idx[n+mm+1:n+mm+n]
    v_idx = idx[n+mm+n+1:n+mm+n+m]
    v̄_idx = idx[n+mm+n+1:n+mm+n+mm]

    xx_idx = Idx[1:n,1:n]
    ūū_idx = Idx[n+1:n+mm,n+1:n+mm]
    xū_idx = Idx[1:n,n+1:n+mm]
    ūx_idx = Idx[n+1:n+mm,1:n]

    # Boundary conditions
    # S = zeros(n+mm,n+mm)
    # s = zeros(n+mm)
    # S[xx_idx] = Wf
    # s[x_idx] = Wf*(X[N] - xf)
    S[N][xx_idx] = Wf
    s[N][x_idx] = Wf*(X[N] - xf)


    # Terminal constraints
    if results isa ConstrainedIterResults
        C = results.C; Iμ = results.Iμ; λ = results.λ
        CxN = results.Cx_N

        # S[xx_idx] += CxN'*results.IμN*CxN
        # s[x_idx] += CxN'*results.IμN*results.CN + CxN'*results.λN
        S[N][xx_idx] += CxN'*results.IμN*CxN
        s[N][x_idx] += CxN'*results.IμN*results.CN + CxN'*results.λN

        # Include the the k = N expansions here for a cleaner backward pass
        Cx, Cu = results.Cx[N], results.Cu[N]

        # s[x_idx] += Cx'*Iμ[N]*C[N] + Cx'*λ[N]
        # s[ū_idx] += Cu'*Iμ[N]*C[N] + Cu'*λ[N]
        #
        # S[xx_idx] += Cx'*Iμ[N]*Cx
        # S[ūū_idx] += Cu'*Iμ[N]*Cu
        # tmp = Cx'*Iμ[N]*Cu
        # S[xū_idx] += tmp
        # S[ūx_idx] += tmp'
        s[N][x_idx] += Cx'*Iμ[N]*C[N] + Cx'*λ[N]
        s[N][ū_idx] += Cu'*Iμ[N]*C[N] + Cu'*λ[N]

        S[N][xx_idx] += Cx'*Iμ[N]*Cx
        S[N][ūū_idx] += Cu'*Iμ[N]*Cu
        tmp = Cx'*Iμ[N]*Cu
        S[N][xū_idx] += tmp
        S[N][ūx_idx] += tmp'
    end

    # Create a copy of boundary conditions in case of regularization
    # SN = copy(S)
    # sN = copy(s)

    # Backward pass
    k = N-1
    Δv = zeros(2) # Initialization of expected change in cost-to-go
    while k >= 1
        # Check for minimum time solve
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        # Unpack results
        fcx, fcu = results.fcx[k], view(results.fcu[k],:,1:m)
        fcy, fcv = results.fcx[k+1], view(results.fcu[k+1],:,1:m)
        fdx, fdu, fdv = results.fdx[k], results.fdu[k], results.fdv[k]

        x = results.X[k]
        y = results.X[k+1]
        u = view(results.U[k],1:m)
        v = view(results.U[k+1],1:m)
        xm = results.xm[k]
        um = view(results.um[k],1:m)
        dx = results.dx[k]
        dy = results.dx[k+1]

        ## L(x,u,y,v) = L(x,u) + L(xm,um) + L(y,v) = L1 + L2 + L3
        # repeated derivatives
        xmx = 0.5*I + dt/8*fcx
        xmu = dt/8*fcu
        xmy = 0.5*I - dt/8*fcy
        xmv = -dt/8*fcv

        ℓxm = W*(xm - xf)
        ℓum = 0.5*R*um

        # ℓ(x,u) expansion
        ℓ1 = ℓ(x,u,W,R,xf)
        ℓ1x = W*(x - xf)
        ℓ1u = R*u

        ℓ1xx = W
        ℓ1uu = R

        # ℓ(xm,um) expansion
        ℓ2 = ℓ(xm,um,W,R,xf)
        ℓ2x = xmx'*ℓxm #(I/2 + dt/8*fcx)'*W*(xm - xf)
        ℓ2u = xmu'*ℓxm + ℓum #((dt/8*fcu[:,1:m])'*W*(xm - xf) + 0.5*R*um[1:m])
        ℓ2y = xmy'*ℓxm #(I/2 - dt/8*fcy)'*W*(xm - xf)
        ℓ2v = xmv'*ℓxm + ℓum #((-dt/8*fcv[:,1:m])'*W*(xm - xf) + 0.5*R*um[1:m])

        ℓ2xx = xmx'*W*xmx #(I/2.0 + dt/8.0*fcx)'*W*(I/2.0 + dt/8.0*fcx)
        ℓ2uu = xmu'*W*xmu + 0.25*R #((dt/8*fcu[:,1:m])'*W*(dt/8*fcu[:,1:m]) + 0.5*R*0.5)
        ℓ2yy = xmy'*W*xmy #(I/2 - dt/8*fcy)'*W*(I/2 - dt/8*fcy)
        ℓ2vv = xmv'*W*xmv + 0.25*R #((-dt/8*fcv[:,1:m])'*W*(-dt/8*fcv[:,1:m]) + 0.5*R*0.5)

        ℓ2xu = xmx'*W*xmu #(I/2 + dt/8*fcx)'*W*(dt/8*fcu[:,1:m])
        ℓ2xy = xmx'*W*xmy #(I/2 + dt/8*fcx)'*W*(I/2 - dt/8*fcy)
        ℓ2xv = xmx'*W*xmv #(I/2 + dt/8*fcx)'*W*(-dt/8*fcv[:,1:m])
        ℓ2uy = xmu'*W*xmy #(dt/8*fcu[:,1:m])'*W*(I/2 - dt/8*fcy)
        ℓ2uv = xmu'*W*xmv + 0.25*R #((dt/8*fcu[:,1:m])'*W*(-dt/8*fcv[:,1:m]) + 0.5*R*0.5)
        ℓ2yv = xmy'*W*xmv #(I/2 - dt/8*fcy)'*W*(-dt/8*fcv[:,1:m])

        # ℓ(y,v) expansion
        ℓ3 = ℓ(y,v,W,R,xf)

        ℓ3y = W*(y - xf)
        ℓ3v = R*v

        ℓ3yy = W
        ℓ3vv = R

        # Assemble expansion
        Lx = dt/6*(ℓ1x + 4*ℓ2x)
        Lu = dt/6*(ℓ1u + 4*ℓ2u)
        Ly = dt/6*(4*ℓ2y + ℓ3y)
        Lv = dt/6*(4*ℓ2v + ℓ3v)
        Lxx = dt/6*(ℓ1xx + 4*ℓ2xx)
        Luu = dt/6*(ℓ1uu + 4*ℓ2uu)
        Lyy = dt/6*(4*ℓ2yy + ℓ3yy)
        Lvv = dt/6*(4*ℓ2vv + ℓ3vv)

        Lxu = 4*dt/6*ℓ2xu
        Lxy = 4*dt/6*ℓ2xy
        Lxv = 4*dt/6*ℓ2xv
        Luy = 4*dt/6*ℓ2uy
        Luv = 4*dt/6*ℓ2uv
        Lyv = 4*dt/6*ℓ2yv

        # Assemble minimum time δL expansion terms
        if solver.opts.minimum_time
            h = results.U[k][m̄]

            # additional, useful expansion terms
            Δd = (dx - dy)
            xmh = 2/8*h*Δd
            ℓ2h = xmh'*ℓxm
            L2h = 4/6*(2*h*ℓ2 + dt*ℓ2h)
            ℓ2hh = 2/8*Δd'*(ℓxm + h*W*xmh)
            L2hh = 4/6*(4*h*ℓ2h + 2*ℓ2 + dt*ℓ2hh)
            L2xh = 4/6*(2*h*ℓ2x + dt*xmx'*W*xmh + 2/8*h*fcx'*ℓxm)
            L2uh = 4/6*(2*h*ℓ2u + dt*xmu'*W*xmh + 2/8*h*fcu'*ℓxm)
            L2hu = 4/6*(2*h*ℓ2u + 2/8*(h^3)*(fcu'*ℓxm + xmu'*W*Δd))
            L2hy = 4/6*(2*h*ℓ2y + 2/8*(h^3)*(-fcy'*ℓxm + xmy'*W*Δd))
            L2hv = 4/6*(2*h*ℓ2v + 2/8*(h^3)*(-fcv'*ℓxm + xmv'*W*Δd))

            # Assemble expansion
            # Lu = [Lu; (2/6*h*(ℓ1 + ℓ3) + L2h + 2*R_minimum_time*h)]
            # Lv = [Lv; 0]
            #
            # Luu = [Luu (2/6*h*ℓ1u + L2uh); (2/6*h*ℓ1u + L2hu)' (2/6*(ℓ1 + ℓ3) + L2hh + 2*R_minimum_time)]
            # Lvv = [Lvv zeros(m); zeros(m)' 0]
            #
            # Lxu = [Lxu (2/6*h*ℓ1x + L2xh)]
            # Lxv = [Lxv zeros(n)]
            # Luy = [Luy; (L2hy + 2/6*h*ℓ3y)']
            # Luv = [Luv zeros(m); (L2hv + 2*h/6*ℓ3v)' 0]
            # Lyv = [Lyv zeros(n)]

            # Lu = [Lu; (2/6*h*(ℓ1 + ℓ3) + L2h + 2*R_minimum_time*h)]
            # l[n+1:n+m] = Lu
            l[n+m̄] = (2/6*h*(ℓ1 + ℓ3) + L2h + 2*R_minimum_time*h)

            # Lv = [Lv; 0]
            # l[n+mm+n+1:n+mm+n+m] = Lv

            # Luu = [Luu (2/6*h*ℓ1u + L2uh); (2/6*h*ℓ1u + L2hu)' (2/6*(ℓ1 + ℓ3) + L2hh + 2*R_minimum_time)]
            # L[n+1:n+m,n+1:n+m] = Luu
            L[u_idx,n+m̄] = 2/6*h*ℓ1u + L2uh
            L[n+m̄,u_idx] = (2/6*h*ℓ1u + L2hu)'
            L[n+m̄,n+m̄] = 2/6*(ℓ1 + ℓ3) + L2hh + 2*R_minimum_time

            # Lvv = [Lvv zeros(m); zeros(m)' 0]
            # L[n+mm+n+1:n+mm+n+m,n+mm+n+1:n+mm+n+m] = Lvv

            # Lxu = [Lxu (2/6*h*ℓ1x + L2xh)]
            # L[1:n,n+1:n+m] = Lxu
            L[x_idx,n+m̄] = 2/6*h*ℓ1x + L2xh

            # Lxv = [Lxv zeros(n)]
            # L[1:n,n+mm+n+1:n+mm+n+m] = Lxv

            # Luy = [Luy; (L2hy + 2/6*h*ℓ3y)']
            # L[n+1:n+m,n+mm+1:n+mm+n] = Luy
            L[n+m̄,y_idx] = (L2hy + 2/6*h*ℓ3y)'

            # Luv = [Luv zeros(m); (L2hv + 2*h/6*ℓ3v)' 0]
            # L[n+1:n+m,n+mm+n+1:n+mm+n+m] = Luv
            L[n+m̄,v_idx] = (L2hv + 2*h/6*ℓ3v)'
            # Lyv = [Lyv zeros(n)]
            # L[n+mm+1:n+mm+n,n+mm+n+1:n+mm+n+m] = Lyv

            # Lu = view(l,n+1:n+m̄)
            # Lv = view(l,n+mm+n+1:n+mm+n+m̄)
            # Luu = view(L,n+1:n+m̄,n+1:n+m̄)
            # Lvv = view(L,n+mm+n+1:n+mm+n+m̄,n+mm+n+1:n+mm+n+m̄)
            # Lxu = view(L,1:n,n+1:n+m̄)
            # Lxv = view(L,1:n,n+mm+n+1:n+mm+n+m̄)
            # Luy = view(L,n+1:n+m̄,n+mm+1:n+mm+n)
            # Luv = view(L,n+1:n+m̄,n+mm+n+1:n+mm+n+m̄)
            # Lyv = view(L,n+mm+1:n+mm+n,n+mm+n+1:n+mm+n+m̄)
        end

        if solver.opts.infeasible
            # Lu = [Lu; R_infeasible*results.U[k][m̄+1:m̄+n]]
            l[n+m̄+1:n+mm] = R_infeasible*results.U[k][m̄+1:m̄+n]
            # Lv = [Lv; zeros(n)]

            # Luu = [Luu zeros(m̄,n); zeros(n,m̄) R_infeasible]
            L[n+m̄+1:n+mm,n+m̄+1:n+mm] = R_infeasible
            # Lvv = [Lvv zeros(m̄,n); zeros(n,m̄) zeros(n,n)]
            #
            # Lxu = [Lxu zeros(n,n)]
            # Lxv = [Lxv zeros(n,n)]
            # Luy = [Luy; zeros(n,n)']
            # Luv = [Luv zeros(m̄,n); zeros(n,m̄) zeros(n,n)]
            # Lyv = [Lyv zeros(n,n)]
        end
        if solver.opts.minimum_time || solver.opts.infeasible
            l[u_idx] = Lu
            l[v_idx] = Lv

            L[u_idx,u_idx] = Luu
            L[v_idx,v_idx] = Lvv
            L[x_idx,u_idx] = Lxu
            L[x_idx,v_idx] = Lxv
            L[u_idx,y_idx] = Luy
            L[u_idx,v_idx] = Luv
            L[y_idx,v_idx] = Lyv

            Lu = view(l,n+1:n+mm)
            Lv = view(l,n+mm+n+1:n+mm+n+mm)
            Luu = view(L,n+1:n+mm,n+1:n+mm)
            Lvv = view(L,n+mm+n+1:n+mm+n+mm,n+mm+n+1:n+mm+n+mm)
            Lxu = view(L,1:n,n+1:n+mm)
            Lxv = view(L,1:n,n+mm+n+1:n+mm+n+mm)
            Luy = view(L,n+1:n+mm,n+mm+1:n+mm+n)
            Luv = view(L,n+1:n+mm,n+mm+n+1:n+mm+n+mm)
            Lyv = view(L,n+mm+1:n+mm+n,n+mm+n+1:n+mm+n+mm)
        end

        # Constraints
        if results isa ConstrainedIterResults
            Cx, Cu = results.Cx[k], results.Cu[k]
            Lx += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            Lu += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            Lxx += Cx'*Iμ[k]*Cx
            Luu += Cu'*Iμ[k]*Cu
            Lxu += Cx'*Iμ[k]*Cu

            if solver.opts.minimum_time
                p,pI,pE = get_num_constraints(solver)
                #TODO Simplify
                if k < N-1
                    Cv = zeros(p,mm)
                    Cv[p,m̄] = -1
                    Lv += Cv'*Iμ[k]*C[k] + Cv'*λ[k]
                    Lvv += Cv'*Iμ[k]*Cv
                    Lxv += Cx'*Iμ[k]*Cv
                    Luv += Cu'*Iμ[k]*Cv
                end
            end
        end

        # Unpack cost-to-go P
        # Sy = view(s,1:n)
        # Sv = view(s,n+1:n+mm)
        # Syy = view(S,1:n,1:n)
        # Svv = view(S,n+1:n+mm,n+1:n+mm)
        # Syv = view(S,1:n,n+1:n+mm)

        Sy = view(s[k+1],1:n)
        Sv = view(s[k+1],n+1:n+mm)
        Syy = view(S[k+1],1:n,1:n)
        Svv = view(S[k+1],n+1:n+mm,n+1:n+mm)
        Syv = view(S[k+1],1:n,n+1:n+mm)

        # Substitute in discrete dynamics (second order approximation)
        tmp0 = Ly + Sy
        Qx = Lx + fdx'*tmp0
        Qu = Lu + fdu'*tmp0
        Qv = Lv + fdv'*tmp0 + Sv

        # calculate components only once
        tmp1 = Lxy*fdx; tmp2 = Luy*fdu; tmp3 = Lyv'*fdv; tmp4 = fdv'*Syv; tmp5 = Lyy + Syy

        # calculate Q
        Qxx = Lxx + tmp1 + tmp1' + fdx'*tmp5*fdx
        Quu = Luu + tmp2 + tmp2' + fdu'*tmp5*fdu
        Qvv = Lvv + tmp3 + tmp3' + fdv'*tmp5*fdv + tmp4 + tmp4' + Svv
        Qxu = Lxu + Lxy*fdu + fdx'*Luy' + fdx'*tmp5*fdu
        Qxv = Lxv + Lxy*fdv + fdx'*Lyv + fdx'*tmp5*fdv + fdx'*Syv
        Quv = Luv + Luy*fdv + fdu'*Lyv + fdu'*tmp5*fdv + fdu'*Syv

        # Regularization
        Qvv_reg = Qvv + results.ρ[1]*I
        Qxv_reg = Qxv
        Quv_reg = Quv

        if !isposdef(Hermitian(Array(Qvv_reg)))

            regularization_update!(results,solver,:increase)

            # Reset BCs
            # S = copy(SN)
            # s = copy(sN)
            ############
            k = N-1
            Δv[1] = 0.
            Δv[2] = 0.
            continue
        end

        # calculate gains
        K[k+1] = -Qvv_reg\Qxv_reg'
        b[k+1] = -Qvv_reg\Quv_reg'
        d[k+1] = -Qvv_reg\Qv

        # calculate optimized values
        Q̄x = Qx + K[k+1]'*Qv + Qxv*d[k+1] + K[k+1]'Qvv*d[k+1]
        Q̄u = Qu + b[k+1]'*Qv + Quv*d[k+1] + b[k+1]'*Qvv*d[k+1]

        # calculate transposed terms only once
        tmp6 = Qxv*K[k+1]
        tmp7 = Quv*b[k+1]
        Q̄xx = Qxx + tmp6 + tmp6' + K[k+1]'*Qvv*K[k+1]
        Q̄uu = Quu + tmp7 + tmp7' + b[k+1]'*Qvv*b[k+1]
        Q̄xu = Qxu + K[k+1]'*Quv' + Qxv*b[k+1] + K[k+1]'*Qvv*b[k+1]

        # cache (approximate) cost-to-go at timestep k
        # s[x_idx] = Q̄x
        # s[ū_idx] = Q̄u
        # S[xx_idx] = Q̄xx
        # S[ūū_idx] = Q̄uu
        # S[xū_idx] = Q̄xu
        # S[ūx_idx] = Q̄xu'
        s[k][x_idx] = Q̄x
        s[k][ū_idx] = Q̄u
        S[k][xx_idx] = Q̄xx
        S[k][ūū_idx] = Q̄uu
        S[k][xū_idx] = Q̄xu
        S[k][ūx_idx] = Q̄xu'

        # line search terms
        Δv[1] += Qv'*d[k+1]
        Δv[2] += 0.5*d[k+1]'*Qvv*d[k+1]

        # at last time step, optimize over final control
        if k == 1
            Q̄uu_reg = Q̄uu + results.ρ[1]*I

            if !isposdef(Array(Hermitian(Q̄uu_reg)))
                regularization_update!(results,solver,:increase)

                ## Reset BCs ##
                # S = copy(SN)
                # s = copy(sN)
                ################
                k = N-1
                Δv[1] = 0.
                Δv[2] = 0.
                continue
            end

            K[1] = -Q̄uu_reg\Q̄xu'
            # b[1] = zeros(mm,mm)
            d[1] = -Q̄uu_reg\Q̄u

            # results.s[1] = Q̄x + K[1]'*Q̄uu*d[1] + K[1]'*Q̄u + Q̄xu*d[1] # calculate for gradient check in solve

            Δv[1] += Q̄u'*d[1]
            Δv[2] += 0.5*d[1]'*Q̄uu*d[1]
        end

        k = k - 1;
    end

    # if successful backward pass, reduce regularization
    regularization_update!(results,solver,:decrease)
    return Δv
end

"""
$(SIGNATURES)
Perform the operation sqrt(A-B), where A and B are Symmetric Matrices
"""
function chol_minus(A,B::Matrix)
    AmB = Cholesky(A,:U,0)
    for i = 1:size(B,1)
        lowrankdowndate!(AmB,B[i,:])
    end
    U = AmB.U
end

function chol_plus(A,B)
    n1,m = size(A)
    n2 = size(B,1)
    P = zeros(n1+n2,m)
    P[1:n1,:] = A
    P[n1+1:end,:] = B
    qr(P)
end



# Outdated
"""
$(SIGNATURES)
Perform a backwards pass with Cholesky Factorizations of the Cost-to-Go to
avoid ill-conditioning.
"""
function _backwardpass_sqrt!(res::SolverVectorResults,solver::Solver)
    N = solver.N
    n = solver.model.n
    m = solver.model.m

    if solver.model.m != length(res.U[1])
        m += n
    end

    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf
    dt = solver.dt

    Uq = cholesky(Q).U
    Ur = cholesky(R).U

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; Su = res.S; s = res.s

    # Terminal Cost-to-go
    if isa(solver.obj, ConstrainedObjective)
        Cx = res.Cx_N
        Su[N] = cholesky(Qf + Cx'*res.IμN*Cx).U
        s[N] = Qf*(X[N] - xf) + Cx'*res.IμN*res.CN + Cx'*res.λN
    else
        Su[N] = cholesky(Qf).U
        s[N] = Qf*(X[N] - xf)
    end

    # Initialization of expected change in cost-to-go
    Δv = [0. 0.]

    k = N-1

    # Backward pass
    while k >= 1
        lx = dt*Q*(X[k] - xf)
        lu = dt*R*(U[k])
        lxx = dt*Q
        luu = dt*R

        fx, fu = res.fdx[k], res.fdu[k]

        Qx = lx + fx'*s[k+1]
        Qu = lu + fu'*s[k+1]

        Wxx = chol_plus(Su[k+1]*fx, cholesky(lxx).U)
        Wuu = chol_plus(Su[k+1]*fu, cholesky(luu).U)

        Qxu = (fx'*Su[k+1]')*(Su[k+1]*fu)

        # Constraints
        if isa(solver.obj, ConstrainedObjective)
            Iμ = res.Iμ; C = res.C; λ = res.λ;
            Cx, Cu = res.Cx[k], res.Cu[k]
            Iμ2 = sqrt.(Iμ[k])
            Qx += (Cx'*Iμ[k]*C[k] + Cx'*λ[k])
            Qu += (Cu'*Iμ[k]*C[k] + Cu'*λ[k])
            Qxu += Cx'*Iμ[k]*Cu

            Wxx = chol_plus(Wxx.R, Iμ2*Cx)
            Wuu = chol_plus(Wuu.R, Iμ2*Cu)
        end

        K[k] = -Wuu.R\(Array(Wuu.R')\Array(Qxu'))
        d[k] = -Wuu.R\(Wuu.R'\Qu)

        s[k] = Qx - Qxu*(Wuu.R\(Wuu.R'\Qu))

        try  # Regularization
            Su[k] = chol_minus(Wxx.R,(Array(Wuu.R'))\Array(Qxu'))
        catch ex
            error("sqrt bp not implemented")
        end

        # Expected change in cost-to-go
        Δv += [vec(Qu)'*vec(d[k]) 0.5*vec(d[k])'*Wuu.R'*Wuu.R*vec(d[k])]

        k = k - 1;
    end

    return Δv
end

function _backwardpass_foh_old!(results::SolverVectorResults,solver::Solver)
    # Problem dimensions
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    # Objective parameters
    W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf

    dt = solver.dt

    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    # Pull out results
    X = results.X; U = results.U; K = results.K; b = results.b; d = results.d; s = results.s; # S = results.S

    # Boundary conditions
    S = zeros(n+mm,n+mm)
    s = zeros(n+mm)
    S[1:n,1:n] = Wf
    s[1:n] = Wf*(X[N]-xf)

    # Terminal constraints
    if results isa ConstrainedIterResults
        C = results.C; Iμ = results.Iμ; λ = results.λ
        CxN = results.Cx_N
        S[1:n,1:n] += CxN'*results.IμN*CxN
        s[1:n] += CxN'*results.IμN*results.CN + CxN'*results.λN

        # Include the the k = N expansions here for a cleaner backward pass
        Cx, Cu = results.Cx[N], results.Cu[N]
        s[1:n] += Cx'*Iμ[N]*C[N] + Cx'*λ[N]
        s[n+1:n+mm] += Cu'*Iμ[N]*C[N] + Cu'*λ[N]
        S[1:n,1:n] += Cx'*Iμ[N]*Cx
        S[n+1:n+mm,n+1:n+mm] += Cu'*Iμ[N]*Cu
        S[1:n,n+1:n+mm] += Cx'*Iμ[N]*Cu
        S[n+1:n+mm,1:n] += Cu'*Iμ[N]*Cx
    end

    # Create a copy of boundary conditions in case of regularization
    SN = copy(S)
    sN = copy(s)

    # Backward pass
    k = N-1
    Δv = [0. 0.] # Initialization of expected change in cost-to-go
    while k >= 1
        # Check for minimum time solve
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        # Unpack results
        fcx, fcu = results.fcx[k], results.fcu[k][:,1:m]
        fcy, fcv = results.fcx[k+1], results.fcu[k+1][:,1:m]
        fdx, fdu, fdv = results.fdx[k], results.fdu[k], results.fdv[k]

        x = results.X[k]
        y = results.X[k+1]
        u = results.U[k]
        v = results.U[k+1]
        xm = results.xm[k]
        um = results.um[k]
        dx = results.dx[k]
        dy = results.dx[k+1]

        ## L(x,u,y,v) = L(x,u) + L(xm,um) + L(y,v) = L1 + L2 + L3
        # ℓ(x,u) expansion
        ℓ1 = ℓ(x,u[1:m],W,R,xf)
        ℓ1x = W*(x - xf)
        ℓ1u = R*u[1:m]

        ℓ1xx = W
        ℓ1uu = R

        # ℓ(xm,um) expansion
        ℓ2 = ℓ(xm,um[1:m],W,R,xf)
        ℓ2x = (I/2 + dt/8*fcx)'*W*(xm - xf)
        ℓ2u = ((dt/8*fcu[:,1:m])'*W*(xm - xf) + 0.5*R*um[1:m])
        ℓ2y = (I/2 - dt/8*fcy)'*W*(xm - xf)
        ℓ2v = ((-dt/8*fcv[:,1:m])'*W*(xm - xf) + 0.5*R*um[1:m])

        ℓ2xx = (I/2.0 + dt/8.0*fcx)'*W*(I/2.0 + dt/8.0*fcx)
        ℓ2uu = ((dt/8*fcu[:,1:m])'*W*(dt/8*fcu[:,1:m]) + 0.5*R*0.5)
        ℓ2yy = (I/2 - dt/8*fcy)'*W*(I/2 - dt/8*fcy)
        ℓ2vv = ((-dt/8*fcv[:,1:m])'*W*(-dt/8*fcv[:,1:m]) + 0.5*R*0.5)

        ℓ2xu = (I/2 + dt/8*fcx)'*W*(dt/8*fcu[:,1:m])
        ℓ2xy = (I/2 + dt/8*fcx)'*W*(I/2 - dt/8*fcy)
        ℓ2xv = (I/2 + dt/8*fcx)'*W*(-dt/8*fcv[:,1:m])
        ℓ2uy = (dt/8*fcu[:,1:m])'*W*(I/2 - dt/8*fcy)
        ℓ2uv = ((dt/8*fcu[:,1:m])'*W*(-dt/8*fcv[:,1:m]) + 0.5*R*0.5)
        ℓ2yv = (I/2 - dt/8*fcy)'*W*(-dt/8*fcv[:,1:m])

        # ℓ(y,v) expansion
        ℓ3 = ℓ(y,v[1:m],W,R,xf)

        ℓ3y = W*(y - xf)
        ℓ3v = R*v[1:m]

        ℓ3yy = W
        ℓ3vv = R

        # Assemble δL expansion
        if solver.opts.minimum_time
            h = u[m̄]

            # fdxditional expansion terms
            xmh = 2/8*h*(dx - dy)
            xmu = (h^2)/8*fcu[:,1:m]
            xmy = 0.5*Matrix(I,n,n) - (h^2)/8*fcy
            xmv = -(h^2)/8*fcv[:,1:m]
            ℓ2h = xmh'*W*(xm-xf)
            L2h = 4/6*(2*h*ℓ2 + (h^2)*ℓ2h)
            ℓ2hh = 2/8*((dx - dy)'*W*(xm - xf) + h*(dx - dy)'*W*xmh)
            L2hh = 4/6*(2*h*ℓ2h + 2*ℓ2 + (h^2)*ℓ2hh + 2*h*ℓ2h)
            L2xh = 4/6*(2*h*ℓ2x + (h^2)*(0.5*Matrix(I,n,n) + (h^2)/8*fcx)'*W*xmh + 2/8*h*fcx'*W*(xm - xf))
            L2uh = 4/6*(2*h*ℓ2u + (h^2)*((h^2)/8*fcu[:,1:m])'*W*xmh + 2/8*h*fcu[:,1:m]'*W*(xm - xf))
            L2hu = 4/6*(2*h*ℓ2u + 2/8*(h^3)*(fcu[:,1:m]'*W*(xm - xf) + xmu'*W*(dx - dy)))
            L2hy = 4/6*(2*h*ℓ2y + 2/8*(h^3)*(-fcy'*W*(xm - xf) + xmy'*W*(dx - dy)))
            L2hv = 4/6*(2*h*ℓ2v + 2/8*(h^3)*(-fcv[:,1:m]'*W*(xm - xf) + xmv'*W*(dx - dy)))

            # Assemble expansion
            Lx = (h^2)/6*ℓ1x + 4/6*(h^2)*ℓ2x
            Lu = [(h^2)/6*ℓ1u + 4/6*(h^2)*ℓ2u; (2/6*h*ℓ1 + L2h + 2/6*h*ℓ3 + 2*R_minimum_time*h)]
            Ly = 4/6*(h^2)*ℓ2y + (h^2)/6*ℓ3y
            Lv = [4/6*(h^2)*ℓ2v + (h^2)/6*ℓ3v; 0]

            Lxx = (h^2)/6*ℓ1xx + 4/6*(h^2)*ℓ2xx
            Luu = [((h^2)/6*ℓ1uu + 4/6*(h^2)*ℓ2uu) (2/6*h*ℓ1u + L2uh); (2/6*h*ℓ1u + L2hu)' (2/6*ℓ1 + L2hh + 2/6*ℓ3 + 2*R_minimum_time)]
            Lyy =  4/6*(h^2)*ℓ2yy + (h^2)/6*ℓ3yy
            Lvv = [(4/6*(h^2)*ℓ2vv + (h^2)/6*ℓ3vv) zeros(m); zeros(m)' 0]

            Lxu = [4/6*(h^2)*ℓ2xu (2/6*h*ℓ1x + L2xh)]
            Lxy = 4/6*(h^2)*ℓ2xy
            Lxv = [4/6*(h^2)*ℓ2xv zeros(n)]
            Luy = [4/6*(h^2)*ℓ2uy; (L2hy + 2*h/6*ℓ3y)']
            Luv = [4/6*(h^2)*ℓ2uv zeros(m); (L2hv + 2*h/6*ℓ3v)' 0]
            Lyv = [4/6*(h^2)*ℓ2yv zeros(n)]
        else
            Lx = dt/6*ℓ1x + 4*dt/6*ℓ2x
            Lu = dt/6*ℓ1u + 4*dt/6*ℓ2u
            Ly = 4*dt/6*ℓ2y + dt/6*ℓ3y
            Lv = 4*dt/6*ℓ2v + dt/6*ℓ3v
            Lxx = dt/6*ℓ1xx + 4*dt/6*ℓ2xx
            Luu = dt/6*ℓ1uu + 4*dt/6*ℓ2uu
            Lyy = 4*dt/6*ℓ2yy + dt/6*ℓ3yy
            Lvv = 4*dt/6*ℓ2vv + dt/6*ℓ3vv

            Lxu = 4*dt/6*ℓ2xu
            Lxy = 4*dt/6*ℓ2xy
            Lxv = 4*dt/6*ℓ2xv
            Luy = 4*dt/6*ℓ2uy
            Luv = 4*dt/6*ℓ2uv
            Lyv = 4*dt/6*ℓ2yv
        end

        if solver.opts.infeasible
            Lu = [Lu; R_infeasible*u[m̄+1:m̄+n]]
            Lv = [Lv; zeros(n)]

            Luu = [Luu zeros(m̄,n); zeros(n,m̄) R_infeasible]
            Lvv = [Lvv zeros(m̄,n); zeros(n,m̄) zeros(n,n)]

            Lxu = [Lxu zeros(n,n)]
            Lxv = [Lxv zeros(n,n)]
            Luy = [Luy; zeros(n,n)']
            Luv = [Luv zeros(m̄,n); zeros(n,m̄) zeros(n,n)]
            Lyv = [Lyv zeros(n,n)]
        end

        # Constraints
        if results isa ConstrainedIterResults
            Cx, Cu = results.Cx[k], results.Cu[k]
            Lx += Cx'*Iμ[k]*C[k] + Cx'*λ[k]
            Lu += Cu'*Iμ[k]*C[k] + Cu'*λ[k]
            Lxx += Cx'*Iμ[k]*Cx
            Luu += Cu'*Iμ[k]*Cu
            Lxu += Cx'*Iμ[k]*Cu

            if solver.opts.minimum_time
                p,pI,pE = get_num_constraints(solver)
                #TODO Simplify
                if k < N-1
                    Cv = zeros(p,mm)
                    Cv[p,m̄] = -1
                    Lv += Cv'*Iμ[k]*C[k] + Cv'*λ[k]
                    Lvv += Cv'*Iμ[k]*Cv
                    Lxv += Cx'*Iμ[k]*Cv
                    Luv += Cu'*Iμ[k]*Cv
                end
            end
        end

        # Unpack cost-to-go P
        Sy = s[1:n]
        Sv = s[n+1:n+mm]
        Syy = S[1:n,1:n]
        Svv = S[n+1:n+mm,n+1:n+mm]
        Syv = S[1:n,n+1:n+mm]

        # Substitute in discrete dynamics (second order approximation)
        Qx = vec(Lx) + fdx'*vec(Ly) + fdx'*vec(Sy)
        Qu = vec(Lu) + fdu'*vec(Ly) + fdu'*vec(Sy)
        Qv = vec(Lv) + fdv'*vec(Ly) + fdv'*vec(Sy) + Sv

        Qxx = Lxx + Lxy*fdx + fdx'*Lxy' + fdx'*Lyy*fdx + fdx'*Syy*fdx
        Quu = Luu + Luy*fdu + fdu'*Luy' + fdu'*Lyy*fdu + fdu'*Syy*fdu
        Qvv = Lvv + Lyv'*fdv + fdv'*Lyv + fdv'*Lyy*fdv + fdv'*Syy*fdv + fdv'*Syv + Syv'*fdv + Svv
        Qxu = Lxu + Lxy*fdu + fdx'*Luy' + fdx'*Lyy*fdu + fdx'*Syy*fdu
        Qxv = Lxv + Lxy*fdv + fdx'*Lyv + fdx'*Lyy*fdv + fdx'*Syy*fdv + fdx'*Syv
        Quv = Luv + Luy*fdv + fdu'*Lyv + fdu'*Lyy*fdv + fdu'*Syy*fdv + fdu'*Syv

        # Regularization
        Qvv_reg = Qvv + results.ρ[1]*I
        Qxv_reg = Qxv
        Quv_reg = Quv

        if !isposdef(Hermitian(Array(Qvv_reg)))

            regularization_update!(results,solver,:increase)

            # Reset BCs
            S = copy(SN)
            s = copy(sN)
            ############
            k = N-1
            Δv = [0. 0.]
            continue
        end

        # calculate gains
        K[k+1] = -Qvv_reg\Qxv_reg'
        b[k+1] = -Qvv_reg\Quv_reg'
        d[k+1] = -Qvv_reg\vec(Qv)

        # calculate optimized values
        Q̄x = vec(Qx) + K[k+1]'*vec(Qv) + Qxv*vec(d[k+1]) + K[k+1]'Qvv*d[k+1]
        Q̄u = vec(Qu) + b[k+1]'*vec(Qv) + Quv*vec(d[k+1]) + b[k+1]'*Qvv*d[k+1]
        Q̄xx = Qxx + Qxv*K[k+1] + K[k+1]'*Qxv' + K[k+1]'*Qvv*K[k+1]
        Q̄uu = Quu + Quv*b[k+1] + b[k+1]'*Quv' + b[k+1]'*Qvv*b[k+1]
        Q̄xu = Qxu + K[k+1]'*Quv' + Qxv*b[k+1] + K[k+1]'*Qvv*b[k+1]

        # cache (approximate) cost-to-go at timestep k
        s[1:n] = Q̄x
        s[n+1:n+mm] = Q̄u
        S[1:n,1:n] = Q̄xx
        S[n+1:n+mm,n+1:n+mm] = Q̄uu
        S[1:n,n+1:n+mm] = Q̄xu
        S[n+1:n+mm,1:n] = Q̄xu'

        # line search terms
        Δv += [vec(Qv)'*vec(d[k+1]) 0.5*vec(d[k+1])'*Qvv*vec(d[k+1])]

        # at last time step, optimize over final control
        if k == 1
            Q̄uu_reg = Q̄uu + results.ρ[1]*I

            if !isposdef(Array(Hermitian(Q̄uu_reg)))
                regularization_update!(results,solver,:increase)

                ## Reset BCs ##
                S = copy(SN)
                s = copy(sN)
                ################
                k = N-1
                Δv = [0. 0.]
                continue
            end

            K[1] = -Q̄uu_reg\Q̄xu'
            b[1] = zeros(mm,mm)
            d[1] = -Q̄uu_reg\vec(Q̄u)

            results.s[1] = vec(Q̄x) + K[1]'*Q̄uu*vec(d[1]) + K[1]'*vec(Q̄u) + Q̄xu*vec(d[1]) # calculate for gradient check in solve

            Δv += [vec(Q̄u)'*vec(d[1]) 0.5*vec(d[1])'*Q̄uu*vec(d[1])]
        end

        k = k - 1;
    end

    # if successful backward pass, reduce regularization
    regularization_update!(results,solver,:decrease)
    return Δv
end

function _backwardpass_old!(res::SolverVectorResults,solver::Solver)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)

    W = solver.obj.Q; R = solver.obj.R; Wf = solver.obj.Qf; xf = solver.obj.xf

    solver.opts.minimum_time ? R_minimum_time = solver.opts.R_minimum_time : nothing
    solver.opts.infeasible ? R_infeasible = solver.opts.R_infeasible*Matrix(I,n,n) : nothing

    dt = solver.dt

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; S = res.S; s = res.s
    L = zeros(n+mm,n+mm)
    l = zeros(n+mm)

    # Useful linear indices
    idx = Array(1:2*(n+mm))
    Idx = reshape(Array(1:(n+mm)^2),n+mm,n+mm)

    x_idx = idx[1:n]
    u_idx = idx[n+1:n+m]
    ū_idx = idx[n+1:n+mm]

    # Boundary Conditions
    S[N] = Wf
    s[N] = Wf*(X[N] - xf)

    # Initialize expected change in cost-to-go
    Δv = zeros(2)

    # Terminal constraints
    if res isa ConstrainedIterResults
        gs = res.gs
        gc = res.gc
        hs = res.hs
        hc = res.hc
        λs = res.λs
        λc = res.λc
        κs = res.κs
        κc = res.κc
        Iμs = res.Iμs
        Iμc = res.Iμc
        Iνs = res.Iνs
        Iνc = res.Iνc
        gsx = res.gsx
        gcu = res.gcu
        hsx = res.hsx
        hcu = res.hcu
        S[N] += gsx[N]'*Iμs[N]*gsx[N] + hsx[N]'*Iνs[N]*hsx[N]
        s[N] += gsx[N]'*(Iμs[N]*gs[N] + λs[N]) + hsx[N]'*(Iνs[N]*hs[N] + κs[N])
    end

    # Backward pass
    k = N-1
    while k >= 1
        solver.opts.minimum_time ? dt = U[k][m̄]^2 : nothing

        x = X[k]
        u = U[k][1:m]

        ℓ1 = ℓ(x,u,W,R,xf)
        ℓx = W*(x - xf)
        ℓu = R*u
        ℓxx = W
        ℓuu = R
        ℓux = zeros(m,n)

        # Assemble expansion

        Lx = dt*ℓx
        Lu = dt*ℓu
        Lxx = dt*ℓxx
        Luu = dt*ℓuu
        Lux = dt*ℓux

        # Minimum time expansion components
        if solver.opts.minimum_time
            h = U[k][m̄]

            l[n+m̄] = 2*h*(ℓ1 + R_minimum_time)

            tmp = 2*h*ℓu
            L[u_idx,m̄] = tmp
            L[m̄,u_idx] = tmp'
            L[m̄,m̄] = 2*(ℓ1 + R_minimum_time)

            L[m̄,x_idx] = 2*h*ℓx'
        end

        # Infeasible expansion components
        if solver.opts.infeasible
            l[n+m̄+1:n+mm] = R_infeasible*U[k][m̄+1:m̄+n]
            L[n+m̄+1:n+mm,n+m̄+1:n+mm] = R_infeasible
        end

        # Final expansion terms
        if solver.opts.minimum_time || solver.opts.infeasible
            l[u_idx] = Lu

            L[u_idx,u_idx] = Luu
            L[u_idx,x_idx] = Lux

            Lu = l[n+1:n+mm]
            Luu = L[n+1:n+mm,n+1:n+mm]
            Lux = L[n+1:n+mm,1:n]
        end

        # Compute gradients of the dynamics
        fdx, fdu = res.fdx[k], res.fdu[k]

        # Constraints
        if res isa ConstrainedIterResults
            k != 1 ? Lx .+= gsx[k]'*(Iμs[k]*gs[k] + λs[k]) + hsx[k]'*(Iνs[k]*hs[k] + κs[k]) : nothing
            Lu .+= gcu[k]'*(Iμc[k]*gc[k] + λc[k]) + hcu[k]'*(Iνc[k]*hc[k] + κc[k])
            k != 1 ? Lxx .+= gsx[k]'*Iμs[k]*gsx[k] + hsx[k]'*Iνs[k]*hsx[k] : nothing
            Luu .+= gcu[k]'*Iμc[k]*gcu[k] + hcu[k]'*Iνc[k]*hcu[k]
            # Lux .+= Cu'*Iμ[k]*Cx # no coupling between constraints
        end

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = Lx + fdx'*s[k+1]
        Qu = Lu + fdu'*s[k+1]
        Qxx = Lxx + fdx'*S[k+1]*fdx
        Quu = Luu + fdu'*S[k+1]*fdu
        Qux = Lux + fdu'*S[k+1]*fdx

        if solver.opts.regularization_type == :state #TODO combined into one
            Quu_reg = Quu + res.ρ[1]*fdu'*fdu
            Qux_reg = Qux + res.ρ[1]*fdu'*fdx
        elseif solver.opts.regularization_type == :control
            Quu_reg = Quu + res.ρ[1]*I
            Qux_reg = Qux
        end

        # Regularization
        if !isposdef(Hermitian(Array(Quu_reg)))  # need to wrap Array since isposdef doesn't work for static arrays
            # increase regularization
            regularization_update!(res,solver,:increase)

            # reset backward pass
            k = N-1
            Δv[1] = 0.
            Δv[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -Quu_reg\Qux_reg
        d[k] = -Quu_reg\Qu

        # Calculate cost-to-go (using unregularized Quu and Qux)
        s[k] = Qx + K[k]'*Quu*d[k] + K[k]'*Qu + Qux'*d[k]
        S[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'*K[k]
        S[k] = 0.5*(S[k] + S[k]')

        # calculated change is cost-to-go over entire trajectory
        Δv[1] += d[k]'*Qu
        Δv[2] += 0.5*d[k]'*Quu*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    regularization_update!(res,solver,:decrease)

    return Δv
end
