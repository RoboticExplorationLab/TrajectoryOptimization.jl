@deprecate Base.length(con::AbstractConstraint) RobotDynaimcs.output_dim(con::AbstractConstraint)

@deprecate Base.length(conval::AbstractConstraintValues) RobotDynamics.output_dim(conval::AbstractConstraintValues)