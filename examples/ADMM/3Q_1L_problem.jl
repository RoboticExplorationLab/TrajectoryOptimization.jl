include("methods.jl")
include("models.jl")

function quad_obstacles(door=:middle)
    r_cylinder = 0.1
    _cyl = []
    h = 3 - 0*1.8  # x-loc [-1.8,2.0]
    w = 0.5      # doorway width [0.1, inf)
    off = 0.0    # y-offset [0, 0.6]
    door_width = 1.0
    off += door_location(door)
    push!(_cyl,(h,  w+off, r_cylinder))
    push!(_cyl,(h, -w+off, r_cylinder))
    push!(_cyl,(h,  w+off+3r_cylinder, 3r_cylinder))
    push!(_cyl,(h, -w+off-3r_cylinder, 3r_cylinder))
    push!(_cyl,(h,  w+off+3r_cylinder+3r_cylinder, 4r_cylinder))
    push!(_cyl,(h, -w+off-3r_cylinder-3r_cylinder, 4r_cylinder))
    push!(_cyl,(h,  w+off+3r_cylinder+9r_cylinder, 6r_cylinder))
    push!(_cyl,(h, -w+off-3r_cylinder-9r_cylinder, 6r_cylinder))
    # push!(_cyl,(h, -w+off-3r_cylinder, 3r_cylinder))
    x_door = [h, off, 0]
    return _cyl, x_door
end

function door_location(door, door_width=1.0)
    if door == :left
        off = door_width
    elseif door == :middle
        off = 0.0
    elseif door == :right
        off = -door_width
    else
        error(string(door) * " not a defined door")
    end
    return off
end

"""
Return the 3D positions of the quads given the position of the load
Default Config:
    Distribute quads evenly around a circle centered around the load, each at a distance `d` from the load.
    The angle `α` specifies the angle between the rope and vertical (i.e. α=pi/2 puts the quads in plane with the load)

Doorway Config:
    Distribute quads evenly over an arc of `2α` degrees, centered at vertical, in the x-z plane
"""
function get_quad_locations(x_load::Vector, d::Real, α=π/4, num_lift=3;
        config=:default, r_cables=[zeros(3) for i = 1:num_lift])
    if config == :default
        h = d*cos(α)
        r = d*sin(α)
        z = x_load[3] + h
        circle(θ) = [x_load[1] + r*cos(θ), x_load[2] + r*sin(θ)]
        θ = range(0,2π,length=num_lift+1)
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            if num_lift == 2
                x_lift[i][1:2] = circle(θ[i] + pi/2)
            else
                x_lift[i][1:2] = circle(θ[i])
            end
            x_lift[i][3] = z
            x_lift[i] += r_cables[i]  # Shift by attachment location
        end
    elseif config == :doorway
        y = x_load[2]
        fan(θ) = [x_load[1] - d*sin(θ), y, x_load[3] + d*cos(θ)]
        θ = range(-α,α, length=num_lift)
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            x_lift[i][1:3] = fan(θ[i])
        end
    end
    return x_lift
end

function build_quad_problem(agent, x0_load=zeros(3), xf_load=[7.5,0,0],
        quat::Bool=false, obstacles::Bool=true, num_lift::Int=3;
        infeasible=false, doors=false, rigidbody=false)
    n_lift = quadrotor_lift.n
    m_lift = quadrotor_lift.m

    # Params
    N = 101          # number of knot points
    dt = 0.1         # time step
    d = 1.55         # rope length
    α = deg2rad(60)  # angle between vertical and ropes
    α2 = deg2rad(60) # arc angle for doorway
    ceiling = 2.1    # ceiling height
    r_ped = 0.15     # last fraction of traj to increase lower bound on load

    # Robot sizes (for obstacles)
    r_lift = 0.275
    r_load = 0.2

    # Calculated Params
    tf = (N-1)*dt           # total time
    Nmid = Int(floor(N/2))  # midpint at which to set the doorway configuration


    # Model
    if rigidbody
        load_dims = (0.5, 0.5, 0.2)
        load_params = let (l,w,h) = load_dims
            (mass=0.350, inertia=Diagonal(1.0I,3),
                gravity=SVector(0,0,-9.81),
                r_cables=[(@SVector [ l/2,    0, h/2])*1,
                          (@SVector [-l/2,  w/2, h/2])*1,
                          (@SVector [-l/2, -w/2, h/2])*1])
        end
        info = Dict{Symbol,Any}(:quat=>4:7, :dims=>load_dims, :r_cables=>load_params.r_cables)

        n_load = 13
        m_load = 3*num_lift
        load_model = Model(Dynamics.load_dynamics!, n_load, m_load, load_params, info)
    else
        _doubleintegrator3D_load = gen_di_load_dyn(num_lift)
        n_load = _doubleintegrator3D_load.n
        m_load = _doubleintegrator3D_load.m
        load_model = _doubleintegrator3D_load
        load_model.info[:r_cables] = [zeros(3) for i = 1:num_lift]
    end
    r_cables = load_model.info[:r_cables]
    load_model.info[:radius] = 0.2
    load_model.info[:rope_length] = d

    door = :middle
    door_width = 1.0
    if doors
        if xf_load[2] == door_width
            door = :left
        elseif xf_load[2] == -door_width
            door = :right
        end
    end
    _cyl, x_door = quad_obstacles(door)




    #~~~~~~~~~~~~~~~~~~~~~~~~~ INITIAL & FINAL POSITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    x0_lift = get_quad_locations(x0_load, d, α, num_lift, r_cables=r_cables)
    xf_lift = get_quad_locations(xf_load, d, α, num_lift, r_cables=r_cables)

    xlift0 = [zeros(n_lift) for i = 1:num_lift]
    xliftf = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        xlift0[i][1:3] = x0_lift[i]
        xliftf[i][1:3] = xf_lift[i]

        # Set quaternion
        xlift0[i][4] = 1.0
        xliftf[i][4] = 1.0
    end

    xload0 = zeros(n_load)
    xload0[1:3] = x0_load
    xloadf = zeros(n_load)
    xloadf[1:3] = xf_load

    if n_load == 13
        xload0[4] = 1.0
        xloadf[4] = 1.0
        xloadf[4:7] = SVector(Quaternion(RotX(pi/2)))
    end

    # midpoint desired configuration
    xm_load = (x0_load + xf_load)/2       # average middle and end positions to get load height at doorway
    xm_load = copy(x0_load)
    xm_load[1:2] = x_door[1:2]            # set x,y location of load to be the location of the door
    xliftmid = [zeros(n_lift) for i = 1:num_lift]
    xm_lift = get_quad_locations(xm_load, d, α2, num_lift, config=:doorway)
    for i = 1:num_lift
        xliftmid[i][1:3] = xm_lift[i]
    end


    #~~~~~~~~~~~~~~~~~~~~~~~~~ OBJECTIVES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    q_diag = ones(n_lift)
    _q_diag = copy(q_diag)
    # q_diag1 = copy(q_diag)
    # q_diag2 = copy(q_diag)
    # q_diag3 = copy(q_diag)
    _q_diag[1] = 1.0e-3
    # q_diag2[1] = 1.0e-3
    # q_diag3[1] = 1.0e-3

    _q_diag[2] = 1.0e-1
    # q_diag2[2] = 1.0e-1
    # q_diag3[2] = 1.0e-1

    _q_diag[3] = 1.0e-3
    # q_diag2[3] = 1.0e-3
    # q_diag3[3] = 1.0e-3

    r_diag = ones(m_lift)
    r_diag[1:4] .= 1.0e-2
    r_diag[5:7] .= 1.0e-2

    # Quads
    Q_lift = [1.0e-1*Diagonal(_q_diag) for i = 1:num_lift]#, 1.0e-1*Diagonal(q_diag2), 1.0e-1*Diagonal(q_diag3)]
    Qf_lift = [1.0*Diagonal(q_diag) for i = 1:num_lift]
    R_lift = Diagonal(r_diag)

    obj_lift = [LQRObjective(Q_lift[i],R_lift,Qf_lift[i],xliftf[i],N) for i = 1:num_lift]

    # Load
    q_load = zeros(n_load)
    if rigidbody
        q_load[4:7] .= 1e-1
    end
    Q_load = Diagonal(q_load)
    Qf_load = 0.0*Diagonal(q_load)
    Qf_load[3,3] = 1.0  # needed to get good initial guess with pedestal constraints. Turned off after initial solve
    R_load = 1.0e-6*Diagonal(I,m_load)
    obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N)

    # Set cost at midpoint
    n_mid = 10
    q_mid = zeros(n_lift)
    q_mid[1:3] .= 100.0 / n_mid

    Q_mid = Diagonal(q_mid)
    cost_mid = [LQRCost(Q_mid,R_lift,xliftmid[i]) for i = 1:num_lift]


    if obstacles
        q_diag = ones(n_lift)

        q_diag1 = copy(q_diag)
        q_diag2 = copy(q_diag)
        q_diag3 = copy(q_diag)
        q_diag1[1] = 1.0e-3
        q_diag2[1] = 1.0e-3
        q_diag3[1] = 1.0e-6
        q_diag2[2] = 1.0e-3
        q_diag3[2] = 1.0e-3
        Q_lift2 = [1.0e-1*Diagonal(q_diag1), 1.0e-1*Diagonal(q_diag2), 1.0e-1*Diagonal(q_diag3)]
        cost2 = [LQRCost(Q_lift2[i], R_lift, xliftf[i]) for i = 1:num_lift]

        for i = 1:num_lift
            for k = 1:N-1
                if k == Nmid
                    obj_lift[i].cost[k] = cost_mid[i]
                elseif k > Nmid
                    # obj_lift[i].cost[k] = cost2[i]
                end
            end
        end

        qm_load = zeros(n_load)
        qm_load[1:3] .= 100.0
        Qm_load = Diagonal(qm_load)
        xloadmid = zeros(n_load)
        xloadmid[1:3] = xm_load
        cost_load_mid = LQRCost(Qm_load, R_load, xloadmid)
        obj_load.cost[Nmid] = cost_load_mid
    end



    #~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Bound quad controls between 0 and 3.0 N
    u_lim_l = -Inf*ones(m_lift)
    u_lim_u =  Inf*ones(m_lift)
    u_lim_l[1:4] .= 0.
    u_lim_u[1:4] .= 12.0/4.0

    # Set floor limit
    x_lim_l_lift = -Inf*ones(n_lift)
    x_lim_l_lift[3] = 0.

    x_lim_l_load = -Inf*ones(n_load)
    x_lim_l_load[3] = 0.

    # Set ceiling limit
    x_lim_u_lift = Inf*ones(n_lift)
    x_lim_u_lift[3] = ceiling

    # Set pedestal limit
    x_min_pedestal = copy(x_lim_l_load)
    x_min_pedestal[3] = xf_load[3]

    bnd1 = BoundConstraint(n_lift, m_lift, u_min=u_lim_l, u_max=u_lim_u)                    # Control bounds on quads
    bnd2 = BoundConstraint(n_lift, m_lift, u_min=u_lim_l, u_max=u_lim_u,
        x_min=x_lim_l_lift, x_max=x_lim_u_lift)                                             # all bounds on quads
    bnd3 = BoundConstraint(n_load, m_load, x_min=x_lim_l_load)                              # all bounds on load
    bnd_ped = BoundConstraint(n_load, m_load, x_min=x_min_pedestal)                         # pedestal bounds on load

    # Obstacles
    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_lift)
        end
    end
    obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_load)
        end
    end
    obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)

    # Generate constraints
    constraints_lift = []
    for i = 1:num_lift
        con = Constraints(N)
        con[1] += bnd1
        for k = 2:N-1
            obstacles ? con[k] += bnd2 + obs_lift : con[k] += bnd2
        end
        con[N] += bnd2
        push!(constraints_lift,copy(con))
    end

    constraints_load = Constraints(N)
    for k = 2:N-1
        obstacles ? constraints_load[k] +=  obs_load : nothing
        if k < floor((1-r_ped)*N)
            constraints_load[k] += bnd3
        else
            obstacles ? constraints_load[k] += bnd_ped : nothing
        end
    end
    constraints_load[N] += goal_constraint(xloadf)


    #~~~~~~~~~~~~~~~~~~~~~~~~~ INITALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    u_load = [0.;0.;-9.81*0.35/num_lift]

    u_lift = zeros(m_lift)
    u_lift[1:4] .= 9.81*(quad_params.m)/4.
    u_lift[5:7] = u_load
    U0_lift = [u_lift for k = 1:N-1]
    U0_load = [-1.0*vcat([u_load for i = 1:num_lift]...) for k = 1:N-1]

    X0_lift = Vector{Vector{Vector{Float64}}}(undef, num_lift)
    for i = 1:num_lift
        X0_lift[i] = to_dvecs(interp_rows(N, tf, [xlift0[i] xliftmid[i] xliftf[i]]))
    end

    #~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE PROBLEMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Create problems
    if agent ∈ 1:num_lift
        i = agent
        prob= Problem(quadrotor_lift,
                    obj_lift[i],
                    U0_lift,
                    integration=:midpoint,
                    constraints=constraints_lift[i],
                    x0=xlift0[i],
                    xf=xliftf[i],
                    N=N,
                    dt=dt)
        if infeasible
            initial_states!(prob, X0_lift[i])
        end

    elseif agent ∈ [0, :load]
        prob = Problem(load_model,
                        obj_load,
                        U0_load,
                        integration=:midpoint,
                        constraints=constraints_load,
                        x0=xload0,
                        xf=xloadf,
                        N=N,
                        dt=dt)
    end


    return prob
end
