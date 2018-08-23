module Dynamics

using TrajectoryOptimization: Model, UnconstrainedObjective, ConstrainedObjective

export
    pendulum,
    doublependulum,
    cartpole,
    ballonbeam,
    acrobot

include("../dynamics/pendulum.jl")
include("../dynamics/doublependulum.jl")
include("../dynamics/acrobot.jl")
include("../dynamics/ballonbeam.jl")
include("../dynamics/cartpole.jl")
include("../dynamics/dubinscar.jl")
include("../dynamics/quadrotor.jl")

end
