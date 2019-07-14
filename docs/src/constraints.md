# [3. Creating Constraints](@id constraint_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["constraints.md"]
```

# Constraint Type
```@docs
ConstraintType
Equality
Inequality
```
All constraints inherit from `AbstractConstraint` and are parameterized by [`ConstraintType`](@ref), which specifies the type of constraint, `Inequality` or `Equality`. This allows the software to easily dispatch over the type of constraint. Each constraint type represents a vector-valued constraint. The intention is that each constraint type represent one line in constraints of problem definition (where they may be vector or scalar-valued). Each constraint has the following interface:

* `evaluate!(v, con, x, u)`: Stage constraint  
* `evaluate!(v, con, x)`: Terminal constraint  
* `jacobian!(V, con, x, u)`: Jacobian wrt states and controls at a stage time step  
* `jacobian!(V, con, x)`: Jacobian wrt terminal state at terminal time step  
* `is_terminal(con)`: Boolean true if the constraint is defined at the terminal time step  
* `is_stage(con)`: Boolean true if the constraint is defined at the stage time steps  
* `length(con)`: Number of constraints (length of the output vector)  

There are currently two types of constraints implemented
```@docs
Constraint
BoundConstraint
```

## General Constraints
### Fields
Each `Constraint` contains the following fields:
* `c`: the in-place constraint function. Methods dispatch over constraint functions of the form `c(v,x,u)` and `c(v,x)`.
* `∇c`: the in-place constraint jacobian function defined as `∇c(Z,x,u)` where `Z` is the p × (n+m) concatenated Jacobian. `p`: number of elements in the constraint vector
* `label`: a Symbol for identifying the constraint
* `type`: a Symbol identifying where the constraint applies. One of `[:stage, :terminal, :all]`

### Basic Constructor
Let's say we have a problem with 3 states and 2 controls, the following constraints ``x_1^2 + x_2^2 - 1 \leq 0``, ``x_2 + u_1  = 0``, and ``x_3 + u_2 = 0``. These two constraints could be created as follows:
```julia
# Problem size
n,m = 3,2

# Inequality Constraint
c(v,x,u) = v[1] = x[1]^2 + x[2]^2 - 1
p1 = 1
con = Constraint{Inequality}(c, n, m, p1, :mycon1)

# Equality Constraint
c_eq(v,x,u) = v[1:2] = x[2:3] + u
p2 = 2
con_eq = Constraint{Equality}(c_eq, n, m, p2, :mycon2)
```

### Analytical Jacobians
Previously, we let the constructor build the Jacobians using ForwardDiff. We can alternatively specify them explicitly:
```julia
# Jacobian for inequality constraint
∇c(V,x,u) = begin V[1,1] = 2x[1]; V[1,2] = 2x[2]; end
con = Constraint{Inequality}(c, ∇c, n, m, p1, :mycon1)

# Jacobian for equality constraint
∇c_eq(V,x,u) = begin V[1,2] = 1; V[1,4] = 1;
                     V[2,3] = 1; V[2,5] = 1; end
con_eq = Constraint{Equality}(c_eq, ∇c_eq, n, m, p2, :mycon2)
```

### Terminal Constraints
We can also build a terminal constraint that only depends on the terminal state, ``x_N``. Let's say we have a terminal constraint ``x_1 + x_2 + x_3 = 1``. We can create it using similar constructors:
```julia
# Build terminal constraint
c_term(v,x) = sum(x) - 1
p_term = 1
con_term = Constraint{Equality}(c_term, n, p_term, :con_term)

# We can also optionally give it an analytical Jacobian
∇c_term(V,x) = ones(1,n)
con_term = Constraint{Equality}(c_term, ∇c_term, n, p_term, :con_term)
```

### Both Stage and Terminal Constraints
Every constraint can be applied to both stage and terminal time steps. The constructor automatically determines which is applied based on the methods defined for the provided function. You can check this by inspecting the `type` field, which will be one of `[:stage, :terminal, :all]`.

Notice our first constraint is only dependent on the state. If we want to enforce it at the terminal time step we can simply use multiple dispatch:
```julia
con.label == :stage  # true
c(v,x) = v[1] = x[1]^2 + x[2]^2 - 1
con = Constraint{Inequality}(c, n, m, p1, :mycon1)
con.label == :all  # true
```

The type can easily be checked with the `is_terminal` and `is_stage` commands.


### Methods
Given that constraints can apply at any time step, we assume that there is the same number of constraints for both stage and terminal time steps. The `length` method can also accept either `:stage` or `:terminal` as a second argument to specify which length you want (since one may be zero). e.g.
```julia
length(con, :stage) == length(con, :terminal)  == 1  # true
length(con_eq, :stage) == 2     # true
length(con_eq, :terminal) == 0  # true
```

### Special Constraints
A few constructors for common constraints have been provided:

```@docs
goal_constraint
planar_obstacle_constraint
```

## Bound Constraints
Bound constraints define simple bounds on the states and controls, allowing the solver to efficiently dispatch on methods to handle these simple constraints (especially when using direct methods). The constructor is very simple

```julia
BoundConstraint(n, m; x_min, x_max, u_min, x_max)
```
The bounds can be given by vectors of the appropriate length, using ±Inf for unbounded values, or a scalar can be passed in when all the states have the same bound. If left blank, the value is assumed to be unbounded.

Working from the previous examples, let's say we have ``-1 \leq x_1 \leq 1, x_3 \leq 10, -15 \leq u \leq 12``:
```julia
# Create a bound constraint
bnd = BoundConstraint(n, m, x_min=[-1,-Inf,-Inf], x_max=[1,Inf,10],
                            u_min=-15, u_max=12)
```

Note that bound constraints are automatically valid at both stage and terminal time steps, i.e. `evaluate!(v, bnd, x, u)` and `evaluate!(v, bnd, x)` are both defined.



# Constraint Sets
A `ConstraintSet` is simply a vector of constraints, and represents a set of constraints at a particular time step. There are some convenient methods for creating and working with constraint sets.

Let's say we combine the previous constraints into a single constraint set. We can do this easily using the `+` method:
```julia
# Create a constraint set for stage time steps
constraints = con + con_eq + bnd

# Create a constraint set for terminal constraints
constraints_term = con + con_term + bnd
```

There are several functions provided to work with `ConstraintSets`
```@docs
TrajectoryOptimization.num_constraints(C::ConstraintSet, type)
Base.pop!(C::ConstraintSet, label::Symbol)
evaluate!(c::PartedVector, C::ConstraintSet, x, u)
evaluate!(c::PartedVector, C::ConstraintSet, x)
jacobian!(c::PartedMatrix, C::ConstraintSet, x, u)
jacobian!(c::PartedMatrix, C::ConstraintSet, x)
```

The `PartedVector` and `PartedMatrix` needed for the `evaluate!` and `jacobian!` methods can be generated using
```julia
PartedVector(C::ConstraintSet, type=:stage)
PartedMatrix(C::ConstraintSet, type=:stage)
```

# Problem Constraints
A `Problem` is made up of individual `ConstraintSet`s at each of the `N` time steps, allowing for different constraints along the trajectory. The collection of `ConstraintSet`s is captured in the `Constraints` type. There are several methods for constructing `Constraints`:
```julia
# Create an empty set
Constraints(N)

# Copy a single ConstraintSet over every time step
Constraints(constraints, N)

# Use a different set at the terminal time step
Constraints(constraints, constraints_term, N)

# Create from a Vector of Constraint Sets
Constraints([constraints, constraints, constraints, constraints_term])
```

You can easily add or remove constraints from time steps using `+` and `pop!` on the appropriate time step:
```julia
pcon = Constraints(N)
pcon[1] += con
pcon[2] += con + con_eq
pop!(pcon[2], :mycon2)
pcon[N] += con_term
```

## Methods
```@docs
TrajectoryOptimization.num_constraints(::Constraints)
```
