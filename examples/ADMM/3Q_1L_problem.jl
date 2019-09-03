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
    The angle `ϕ` specifies how much the formation is rotated about Z

Doorway Config:
    Distribute quads evenly over an arc of `2α` degrees, centered at vertical, in the x-z plane
"""
function get_quad_locations(x_load::Vector, d::Real, α=π/4, num_lift=3;
        config=:default, r_cables=[zeros(3) for i = 1:num_lift], ϕ=0.0)
    if config == :default
        h = d*cos(α)
        r = d*sin(α)
        z = x_load[3] + h
        circle(θ) = [x_load[1] + r*cos(θ), x_load[2] + r*sin(θ)]
        θ = range(0,2π,length=num_lift+1) .+ ϕ
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



function quad_objective_weights(model::Model, num_lift)
    n_lift, m_lift = model.n, model.m

    r_control = 1.0e-3*ones(4)
    r_slack = 1.0*ones(1)

    Q_lift = [1.0e-1*Diagonal(ones(n_lift)) for i = 1:num_lift]

    for ii = 1:num_lift
        for i = 1:3
            Q_lift[ii][i,i] = 1.0e-3
        end
    end
    Qf_lift = [100.0*Diagonal(ones(n_lift)) for i = 1:num_lift]
    R_lift = Diagonal([r_control;r_slack])

    return Q_lift, R_lift, Qf_lift
end

function build_quad_problem(agent, x0_load=zeros(3), xf_load=[7.5,0,0],
        quat::Bool=false, obstacles::Bool=true, num_lift::Int=3;
        infeasible=false, doors=false, rigidbody=false)
    build_lift_problem(quadrotor_lift, agent, x0_load, xf_load,
        quat, obstacles, num_lift;
        infeasible=infeasible, doors=doors, rigidbody=rigidbody)
end

function build_lift_problem(lift_model::Model, agent, x0_load=zeros(3), xf_load=[7.5,0,0],
        quat::Bool=false, obstacles::Bool=true, num_lift::Int=3;
        infeasible=false, doors=false, rigidbody=false)

    n_lift = lift_model.n
    m_lift = lift_model.m

    is_quad = n_lift == 13

    # Params
    N = 101         # number of knot points
    dt = 0.1        # time step
    d = 1.55         # rope length
    α = deg2rad(60)  # angle between vertical and ropes
    α2 = deg2rad(60) # arc angle for doorway
    ceiling = 2.1    # ceiling height
    r_ped = 0.15     # last fraction of traj to increase lower bound on load
    ϕ = 0*pi/4            # load rotation angle

    # Robot sizes (for obstacles)
    r_lift = 0.275
    r_load = 0.2

    # Calculated Params
    tf = (N-1)*dt           # total time
    Nmid = Int(floor(N/2))  # midpint at which to set the doorway configuration

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
    x0_lift = get_quad_locations(x0_load, d, α, num_lift)
    xf_lift = get_quad_locations(xf_load, d, α, num_lift, ϕ=ϕ)

    xlift0 = [zeros(n_lift) for i = 1:num_lift]
    xliftf = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        xlift0[i][1:3] = x0_lift[i]
        xliftf[i][1:3] = xf_lift[i]

        # Set quaternion
        if is_quad
            xlift0[i][4] = 1.0
            xliftf[i][4] = 1.0
        end
    end

    # Model
    n_load = 6
    m_load = 3
    xload0 = zeros(n_load)
    xload0[1:3] = x0_load
    xloadf = zeros(n_load)
    xloadf[1:3] = xf_load
    load_model = gen_load_model_initial(xload0,xlift0)
    n_load, m_load = load_model.n, load_model.m
    load_model.info[:rope_length] = d
    load_mass = load_model.info[:mass]


    # if n_load == 13
    #     xload0[4] = 1.0
    #     xloadf[4] = 1.0
    #     xloadf[4:7] = SVector(Quaternion(RotZ(ϕ)))
    # end

    # midpoint desired configuration
    xm_load = (x0_load + xf_load)/2       # average middle and end positions to get load height at doorway
    xm_load = copy(x0_load)
    xm_load[1:2] = x_door[1:2]            # set x,y location of load to be the location of the door
    xliftmid = [zeros(n_lift) for i = 1:num_lift]
    xm_lift = get_quad_locations(xm_load, d, α2, num_lift, config=:doorway)
    for i = 1:num_lift
        xliftmid[i][1:3] = xm_lift[i]
    end

    xliftmid[2][2] += 0.01
    xliftmid[3][2] -= 0.01

    #~~~~~~~~~~~~~~~~~~~~~~~~~ INITALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Initial controls
    f1 = (x0_lift[1][1:3] - xload0[1:3])/norm(x0_lift[1][1:3] - xload0[1:3])
    f2 = (x0_lift[2][1:3] - xload0[1:3])/norm(x0_lift[2][1:3] - xload0[1:3])
    f3 = (x0_lift[3][1:3] - xload0[1:3])/norm(x0_lift[3][1:3] - xload0[1:3])
    f_mag = hcat(f1, f2, f3)\[0;0;9.81*load_mass]
    ff = [f_mag[1]*f1, f_mag[2]*f2, f_mag[3]*f3]

    thrust = 9.81*(lift_model.info[:mass] + load_mass/num_lift)/4
    ulift = [[thrust;thrust;thrust;thrust;f_mag[i]] for i = 1:num_lift]

    uload = vcat(f_mag...)

    f1m = (xliftmid[1][1:3] - xm_load[1:3])/norm(xliftmid[1][1:3] - xm_load[1:3])
    f2m = (xliftmid[2][1:3] - xm_load[1:3])/norm(xliftmid[2][1:3] - xm_load[1:3])
    f3m = (xliftmid[3][1:3] - xm_load[1:3])/norm(xliftmid[3][1:3] - xm_load[1:3])
    f_magm = hcat(f1m, f2m, f3m)\[0;0;9.81*load_mass]
    ffm = [f_magm[1]*f1m, f_magm[2]*f2m, f_magm[3]*f3m]

    thrustm = 9.81*(lift_model.info[:mass] + load_mass/num_lift)/4
    uliftm = [[thrustm;thrustm;thrustm;thrustm;f_mag[i]] for i = 1:num_lift]

    uloadm = vcat(f_mag...)

    U0_load = [uload for k = 1:N-1]
    U0_lift = [[ulift[i] for k = 1:N-1] for i = 1:num_lift]

    #~~~~~~~~~~~~~~~~~~~~~~~~~ OBJECTIVES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    Q_lift, R_lift, Qf_lift = quad_objective_weights(lift_model, num_lift)

    obj_lift = [LQRObjective(Q_lift[i],R_lift,Qf_lift[i],xliftf[i],N) for i = 1:num_lift]

    # Load

    Q_load = 0.0*Diagonal(ones(n_load))
    Qf_load = 0.0*Diagonal(ones(n_load))
    # Qf_load[3,3] = 1.0  # needed to get good initial guess with pedestal constraints. Turned off after initial solve
    R_load = Diagonal(ones(num_lift))
    obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N)


    if obstacles
        # Set cost at midpoint
        q_mid = 1.0e-1*ones(n_lift)
        q_mid[1:3] .= 100.0

        Q_mid = Diagonal(q_mid)
        cost_mid = [LQRCost(Q_mid,R_lift,xliftmid[i]) for i = 1:num_lift]

        for i = 1:num_lift
            obj_lift[i].cost[Nmid] = cost_mid[i]
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
    if is_quad
        u_lim_l = -Inf*ones(m_lift)
        u_lim_u =  Inf*ones(m_lift)
        u_lim_l[1:4] .= 0.
        u_lim_u[1:4] .= 12.0/4.0
        u_lim_l[5] = 0.
    else
        u_lim_u = Inf*ones(m_lift)
        u_lim_u[1:3] .= 12/.850
        u_lim_l = -Inf*ones(m_lift)
        u_lim_l[3] = 0.
    end

    # Set floor limit
    x_lim_l_lift = -Inf*ones(n_lift)
    x_lim_l_lift[3] = 0.

    x_lim_l_load = -Inf*ones(n_load)
    x_lim_l_load[3] = 0.

    # Set ceiling limit
    x_lim_u_lift = Inf*ones(n_lift)
    x_lim_u_lift[3] = ceiling

    u_lim_l_load = zeros(m_load)

    # Set pedestal limit
    x_min_pedestal = copy(x_lim_l_load)
    x_min_pedestal[3] = xf_load[3]

    bnd1 = BoundConstraint(n_lift, m_lift, u_min=u_lim_l, u_max=u_lim_u)                    # Control bounds on quads
    bnd2 = BoundConstraint(n_lift, m_lift, u_min=u_lim_l, u_max=u_lim_u,
        x_min=x_lim_l_lift, x_max=x_lim_u_lift)                                             # all bounds on quads
    # bnd3 = BoundConstraint(n_load, m_load, x_min=x_lim_l_load)#, u_min=u_lim_l_load)                              # all bounds on load
    # bnd_ped = BoundConstraint(n_load, m_load, x_min=x_min_pedestal)#, u_min=u_lim_l_load)                         # pedestal bounds on load

    # Add constraints
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
            if obstacles
                con[k] += bnd2 + obs_lift
            else
                con[k] += bnd2
            end
        end
        con[N] += bnd2
        push!(constraints_lift,copy(con))
    end

    constraints_load = Constraints(N)
    # for k = 2:N-1
    #     if obstacles
    #         constraints_load[k] += obs_load
    #         if k < floor((1-r_ped)*N)
    #             constraints_load[k] += bnd3
    #         else
    #             constraints_load[k] += bnd_ped
    #         end
    #     elseif k < N-1
    #         constraints_load[k] += bnd3
    #     end
    # end
    constraints_load[N] += goal_constraint(xloadf)

    #~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE PROBLEMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Create problems
    if agent ∈ 1:num_lift
        i = agent
        prob= Problem(gen_lift_model_initial(agent,xload0,xlift0[agent]),
                    obj_lift[i],
                    U0_lift[i],
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
