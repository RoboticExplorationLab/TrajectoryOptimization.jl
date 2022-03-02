import Base: copy, +

#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

"""
Abstract type that represents a scalar-valued function that accepts a state and control
at a single knot point.
"""
abstract type CostFunction <: RobotDynamics.ScalarFunction end
