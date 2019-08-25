free_body_params = (mass=1.0, inertia=Diagonal(1.0I,3))
function free_body_dynamics!(ẋ,x,u,params)
    J = params.inertia

    ω = x[1:3]
    q = x[4:7]

    ẋ[1:3] = J\(u[1:3] - ω × (J*ω))
    ẋ[4:7] = SVector(0.5*Quaternion(q)*Quaternion(zero(eltype(x)),ω))
end

free_body_model = Model(free_body_dynamics!,7,3,free_body_params)

function rigid_body_dynamics!(ẋ,x,u,params)
    J = params.inertia

    r = x[1:3]
    q = Quaternion(x[4:7])
    v = x[8:10]
    ω = x[11:13]

    F = u[1:3]  # force in body frame
    M = u[4:6]  # torque in body frame


    ẋ[1:3] = v
    ẋ[4:7] = SVector(0.5*q*Quaternion(zero(eltype(x)),ω))
    ẋ[8:10] = (q*F)/m
    ẋ[11:13] = J\(M - ω × (J*ω))
end

function load_dynamics!(ẋ,x,u,params)
    r_cables = params.r_cables
    n_cables = length(r_cables)

    # Get quaternion
    q = Quaternion(x[4:7])

    # Get input forces
    F_ = reshape(u,3,n_cables)
    F = [col for col in eachcol(F_)]

    # Convert to body frame
    F_body = [inv(q)*f for f in F]

    # Calculate Torque
    M_body = [r × f for (r,f) in zip(r_cables, F_body)]

    # Total torque and force
    F_total = sum(F_body)
    M_total = sum(M_body)

    u_new = zeros(6)
    u_new[1:3] = F_total
    u_new[4:6] = M_total
    rigid_body_dynamics!(ẋ,x,[F_total; M_total], params)
end
