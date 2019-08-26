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
    m = params.mass
    g = params.gravity
    J = params.inertia

    r = x[1:3]
    q = Quaternion(x[4:7])
    v = x[8:10]
    ω = x[11:13]

    F = u[1:3]  # force in body frame
    M = u[4:6]  # torque in body frame


    ẋ[1:3] = v
    ẋ[4:7] = SVector(0.5*q*Quaternion(zero(eltype(x)),ω))
    ẋ[8:10] = g + (q*F)/m
    ẋ[11:13] = J\(M - ω × (J*ω))
end
