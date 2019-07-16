
##########################
#     SOLVE METHODS      #
##########################

#
# function solve!(prob::Problem{T,Discrete}, solver::SequentialNewtonSolver) where T
#     V_ = newton_step!(prob, solver)
#     copyto!(prob.X, V_.X)
#     copyto!(prob.U, V_.U)
#     projection!(prob)
#     return solver
# end
#
# """
# Take a Projected Newton step using a purely sequential method (never forms block arrays)
# """
# function newton_step!(prob, solver::SequentialNewtonSolver)
#     V = solver.V
#     verbose = solver.opts.verbose
#
#     # Initial stats
#     update!(prob, solver)
#     J0 = cost(prob, V)
#     res0 = res2(solver, V)
#     viol0 = max_violation(solver)
#
#     # Projection
#     verbose ? println("\nProjection:") : nothing
#     projection!(prob, solver, V)
#     res1, = multiplier_projection!(solver, V)
#
#     # Solve KKT
#     J1 = cost(prob, V)
#     viol1 = max_violation(solver)
#     δx, δu, δλ = solveKKT(solver, V)
#
#     # Line Search
#     verbose ? println("\nLine Search") : nothing
#     V_ = line_search(prob, solver, δx, δu, δλ)
#     J_ = cost(prob, V_)
#     res_ = res2(solver, V_)
#     viol_ = max_violation(solver)
#
#     # Print Stats
#     if verbose
#         println("\nStats")
#         println("cost: $J0 → $J1 → $J_")
#         println("res: $res0 → $res1 → $res_")
#         println("viol: $viol0 → $viol1 → $viol_")
#     end
#
#     return V_
# end
#
# function line_search(prob::Problem, solver::SequentialNewtonSolver, δx, δu, δλ)
#     α = 1.0
#     s = 0.01
#     V = solver.V
#     J0 = cost(prob, V)
#     update!(prob, solver)
#     res0 = res2(solver, V)
#     count = 0
#     solver.opts.verbose ? println("res0: $res0") : nothing
#     while count < 10
#         V_ = V + (α.*(δx,δu,δλ),solver.active_set)
#
#         # Calculate residual
#         projection!(prob, solver, V_)
#         res, = multiplier_projection!(solver, V_)
#         J = cost(prob, V_)
#
#         # Calculate max violation
#         viol = max_violation(solver)
#
#         if solver.opts.verbose
#             println("cost: $J \t residual: $res \t feas: $viol")
#         end
#         if res < (1-α*s)*res0
#             solver.opts.verbose ? println("α: $α") : nothing
#             return V_
#         end
#         count += 1
#         α /= 2
#     end
#     return solver.V
# end
#
# """
# Calculates Hinv*Y'*((Y*Hinv*Y')\\y), which is the least norm solution to the problem
#     min xᵀHx
#      st Y*x = y
# where y is the vector of active constraints and Y is the constraint jacobian
# stores the result in vals.x and vals.u, with the intermediate lagrange multiplier stored in vals.λ
# """
# function _projection(solver::SequentialNewtonSolver)
#     y = active_constraints(solver)
#     calc_factors!(solver)
#     δλ = solve_cholesky(solver, y)
#
#     δx,δu = jac_T_mult(solver, δλ)
#     δx = -solver.Qinv .* δx
#     δu = -solver.Rinv .* δu
#     return δx, δu, δλ
# end
#
#
# function projection!(prob, solver::SequentialNewtonSolver, V::PrimalDualVars, active_set_update=true)
#     X,U = V.X, V.U
#     eps_feasible = solver.opts.feasibility_tolerance
#     count = 0
#     # cost_expansion!(prob, solver, V)
#     feas = Inf
#     while true
#         dynamics_constraints!(prob, solver, V)
#         update_constraints!(prob, solver, V)
#         dynamics_jacobian!(prob, solver, V)
#         constraint_jacobian!(prob, solver, V)
#         if active_set_update
#             active_set!(prob, solver)
#         end
#         y = active_constraints(solver)
#
#         viol = maximum(norm.(y,Inf))
#         if solver.opts.verbose
#             println("feas: ", viol)
#         end
#         if viol < eps_feasible || count > 10
#             break
#         else
#             δx, δu = _projection(solver)
#             _update_primals!(V, δx, δu)
#             count += 1
#         end
#     end
# end
#
# """
# Solve the least-squares problem to find the best multipliers,
#     given the current active multipliers λ
# """
# function _mult_projection(solver::SequentialNewtonSolver, λ)
#     N = length(solver.Q)
#
#     # Calculate Y'λ
#     δx,δu = jac_T_mult(solver, λ)
#
#     # Calculate g + Y'λ
#     for k = 1:N-1
#         δx[k] += solver.Q[k].x
#         δu[k] += solver.Q[k].u
#     end
#     δx[N] += solver.Q[N].x
#
#     # Calculate Y*(g + Y'λ)
#     r = jac_mult(solver, δx, δu)
#
#     # Solve (Y*Y')\r
#     eyes = [I for k = 1:N]
#     calc_factors!(solver, eyes, eyes)
#     δλ = solve_cholesky(solver, r)
#
#     return -δλ
# end
#
# function multiplier_projection!(solver::SequentialNewtonSolver, V)
#     λa = active_duals(V, solver.active_set)
#     δλ = _mult_projection(solver, λa)
#     _update_duals!(V, δλ, solver.active_set)
#     return res2(solver,V), δλ
# end
#
# """
# Solve the KKT system [H Y'][δz] = [g + Y'λ]
#                      [Y 0 ][δλ]   [   y   ]
#     using a sequential cholesky factorization of the Shur compliment S = (Y*Hinv*Y')
#     '''
#     r = g + Y'λ
#     δλ = Sinv*(y - Y*Hinv*r)
#     δz = -Hinv*(r + Y'δλ)
#     '''
# """
# function _solveKKT(solver::SequentialNewtonSolver, λ)
#     N = length(solver.Q)
#
#     # Calculate r = g + Y'λ
#     rx,ru = jac_T_mult(solver, λ)
#     for k = 1:N-1
#         rx[k] = solver.Q[k].x + rx[k]
#         ru[k] = solver.Q[k].u + ru[k]
#     end
#     rx[N] = solver.Q[N].x + rx[N]
#
#     # Calculate b0 = Y*Hinv*r
#     δx = solver.Qinv .* rx
#     δu = solver.Rinv .* ru
#     b0 = jac_mult(solver, δx, δu)
#
#     # Calculate y - Y*Hinv*r
#     y = active_constraints(solver)
#     b = y - b0
#
#     # Solve for δλ = (Y*Hinv*Y')\b
#     calc_factors!(solver)
#     δλ = solve_cholesky(solver, b)
#
#     # Solve for δz = -Hinv*(r + Y'δλ)
#     δx, δu = jac_T_mult(solver, δλ)
#     for k = 1:N-1
#         δx[k] = -solver.Qinv[k]*(δx[k] + rx[k])
#         δu[k] = -solver.Rinv[k]*(δu[k] + ru[k])
#     end
#     δx[N] = -solver.Qinv[N]*(δx[N] + rx[N])
#     return δx, δu, δλ
# end
#
# function solveKKT(solver::SequentialNewtonSolver, V)
#     λa = active_duals(V, solver.active_set)
#     δx, δu, δλ = _solveKKT(solver, λa)
#     return δx, δu, δλ
# end
