free_body_params = (mass=1.0, inertia=Diagonal(1.0I,3))
function free_body_dynamics!(ẋ,x,u,params)
    J = params.inertia

    ω = x[1:3]
    q = x[4:7]

    ẋ[1:3] = J\(u[1:3] - ω × (J*ω))
    ẋ[4:7] = SVector(0.5*Quaternion(q)*Quaternion(zero(eltype(x)),ω))
end

free_body_model = Model(free_body_dynamics!,7,3,free_body_params)
