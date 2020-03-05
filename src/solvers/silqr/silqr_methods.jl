
#
# """
# Take one step of iLQR algorithm (non-allocating)
# """
# function step!(solver::iLQRSolver, J)
#     Z = solver.Z
#     state_diff_jacobian!(solver.G, solver.model, Z)
#     discrete_jacobian!(solver.∇F, solver.model, Z)
#     cost_expansion!(solver.Q, solver.G, solver.obj, solver.model, solver.Z)
#     ΔV = backwardpass!(solver)
#     forwardpass!(solver, ΔV, J)
# end
#
#
# """
# $(SIGNATURES)
# Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
# Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
# """
# function backwardpass!(solver::StaticiLQRSolver{T,QUAD}) where {T,QUAD<:QuadratureRule}
#     n,m,N = size(solver)
#
#     # Objective
#     obj = solver.obj
#     model = solver.model
#
#     # Extract variables
#     Z = solver.Z; K = solver.K; d = solver.d;
#     G = solver.G
#     S = solver.S
#     Q = solver.Q
#
#     # Terminal cost-to-go
#     S.xx[N] = Q.xx[N]
#     S.x[N] = Q.x[N]
#
#     # Initialize expecte change in cost-to-go
#     ΔV = @SVector zeros(2)
#
#
#     k = N-1
#     while k > 0
#         ix = Z[k]._x
#         iu = Z[k]._u
#
#         # fdx = G[k+1]'solver.∇F[k][ix,ix]*G[k]
#         # fdu = G[k+1]'solver.∇F[k][ix,iu]
#         fdx,fdu = dynamics_expansion(solver.∇F[k], G[k], G[k+1], model, Z[k])
#         # fdx, fdu = dynamics_expansion(QUAD, model, Z[k])
#
#         Qx =  Q.x[k] + fdx'S.x[k+1]
#         Qu =  Q.u[k] + fdu'S.x[k+1]
#         Qxx = Q.xx[k] + fdx'S.xx[k+1]*fdx
#         Quu = Q.uu[k] + fdu'S.xx[k+1]*fdu
#         Qux = Q.ux[k] + fdu'S.xx[k+1]*fdx
#
#         if solver.opts.bp_reg_type == :state
#             Quu_reg = Quu + solver.ρ[1]*fdu'fdu
#             Qux_reg = Qux + solver.ρ[1]*fdu'fdx
#         elseif solver.opts.bp_reg_type == :control
#             Quu_reg = Quu + solver.ρ[1]*I
#             Qux_reg = Qux
#         end
#
#         # Regularization
#         if solver.opts.bp_reg
#             vals = eigvals(Hermitian(Quu_reg))
#             if minimum(vals) <= 0
#                 @warn "Backward pass regularized"
#                 regularization_update!(solver, :increase)
#                 k = N-1
#                 ΔV = @SVector zeros(2)
#                 continue
#             end
#         end
#
#         # Compute gains
#         K[k] = -(Quu_reg\Qux_reg)
#         d[k] = -(Quu_reg\Qu)
#
#         # Calculate cost-to-go (using unregularized Quu and Qux)
#         S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
#         S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
#         S.xx[k] = 0.5*(S.xx[k] + S.xx[k]')
#
#         # calculated change is cost-to-go over entire trajectory
#         ΔV += @SVector [d[k]'*Qu, 0.5*d[k]'*Quu*d[k]]
#
#         k -= 1
#     end
#
#     regularization_update!(solver, :decrease)
#
#     return ΔV
#
# end
#

#
# """
# $(SIGNATURES)
# Simulate forward the system with the optimal feedback gains from the iLQR backward pass.
# (non-allocating)
# """
# function rollout!(solver::StaticiLQRSolver{T,Q}, α) where {T,Q}
#     Z = solver.Z; Z̄ = solver.Z̄
#     K = solver.K; d = solver.d;
#
#     Z̄[1].z = [solver.x0; control(Z[1])]
#
#     temp = 0.0
#
#
#     for k = 1:solver.N-1
#         δx = state_diff(solver.model, state(Z̄[k]), state(Z[k]))
#         ū = control(Z[k]) + K[k]*δx + α*d[k]
#         set_control!(Z̄[k], ū)
#
#         # Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
#         Z̄[k+1].z = [discrete_dynamics(Q, solver.model, Z̄[k]);
#             control(Z[k+1])]
#
#         temp = norm(Z̄[k+1].z)
#         if temp > solver.opts.max_state_value
#             return false
#         end
#     end
#     return true
# end
