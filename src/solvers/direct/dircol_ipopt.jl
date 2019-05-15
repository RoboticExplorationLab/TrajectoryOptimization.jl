
function gen_ipopt_functions(prob::Problem, solver::DIRCOLSolver)

    n,m,N = size(prob)
    p_colloc = num_colloc(prob)
    p_custom = sum(num_constraints(prob))
    p_total = p_colloc + p_custom

    # Create initial primals
    Z0 = Primals(prob,true)
    NN = length(Z0)
    X0,U0 = Z0.X, Z0.U

    # Get constraint jacobian sparsity structure
    jac_structure = spzeros(p_total, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)

    #################
    # COST FUNCTION #
    #################
    function eval_f(Z)
        Z = Primals(Z,X0,U0)
        cost(prob, Z)
    end

    ###########################
    # COLLOCATION CONSTRAINTS #
    ###########################
    function eval_g(Z, g)
        Z = Primals(Z, X0, U0)
        dynamics!(prob, solver, Z)
        traj_points!(prob, solver, Z)
        update_constraints!(g, prob, solver, Z)
    end


    #################
    # COST GRADIENT #
    #################
    function eval_grad_f(Z, grad_f)
        Z = Primals(Z, X0, U0)
        cost_gradient!(grad_f, prob, solver, Z)
    end

    #######################
    # CONSTRAINT JACOBIAN #
    #######################
    function eval_jac_g(Z, mode, rows, cols, vals)
        if mode == :Structure
            copyto!(rows, r)
            copyto!(cols, c)
        else
            Z = Primals(Z, X0, U0)
            dynamics!(prob, solver, Z)
            traj_points!(prob, solver, Z)
            calculate_jacobians!(prob, solver, Z)
            constraint_jacobian!(vals, prob, solver, Z)
        end
    end

    return eval_f, eval_g, eval_grad_f, eval_jac_g
end

function remove_bounds!(prob::Problem)
    bounds = BoundConstraint[]
    for con in prob.constraints
        bnd = remove_bounds!(con)
        push!(bounds, bnd)
    end
end

function solve!(prob::Problem, solver::DIRCOLSolver)
    eval_f, eval_g, eval_grad_f, eval_jac_g = gen_ipopt_functions(prob, solver)

    NN = N*(n+m)
    p_colloc = num_colloc(prob)
    p_

    problem = Ipopt.createProblem(NN, x_L, x_U, P, g_L, g_U, nG, nH,
        eval_f, eval_g, eval_grad_f, eval_jac_g)
    prob.x = Z0
end
