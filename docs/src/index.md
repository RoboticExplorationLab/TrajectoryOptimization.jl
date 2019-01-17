# TrajectoryOptimization.jl

Documentation for TrajectoryOptimization.jl

```@contents
```

# Overview
The purpose of this package is to provide a testbed for state-of-the-art trajectory optimization algorithms. In general, this package focuses on trajectory optimization problems of the form
(put LaTeX here)

This package currently implements both indirect and direct methods for trajectory optimization:
* Iterative LQR (iLQR): indirect method based on differential dynamic programming
* Direct Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solver

The primary focus of this package is developing the iLQR algorithm, although we hope this will extend to many algorithms in the future.


# Getting Started
In order to set up a trajectory optimization problem, the user needs to create a Model and Objective

## Creating a Model
There are two ways of creating a model:
1) An in-place analytic function of the form f(ẋ,x,u)
2) A URDF

### Analytic Models
To create an analytic model we just need to 

```@docs
Solver
Model
SolverResults
```
