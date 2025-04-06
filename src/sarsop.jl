"""
    solve(solver, momdp, policy)

Runs pomdpsol using the options in 'solver' on 'pomdp',
and writes out a .policy xml file specified by 'policy'.
"""
function POMDPs.solve(solver::SARSOPSolver, momdp::MOMDP; kwargs...)
    if !isempty(kwargs)
        @warn("Keyword args for solve(::SARSOPSolver, ::MOMDP) are no longer supported. Use the options in the SARSOPSolver")
    end
    pomdp_file = POMDPFile(momdp, solver.pomdp_filename, verbose=solver.verbose)
    options = SARSOP.get_solver_options(solver)
    if solver.verbose
        run(`$(SARSOP.pomdpsol()) $(pomdp_file.filename) --output $(solver.policy_filename) $options`)
    else
        success(`$(SARSOP.pomdpsol()) $(pomdp_file.filename) --output $(solver.policy_filename) $options`)
    end
    alphas = MOMDPAlphas(solver.policy_filename)
    action_map = broadcast(x -> getindex(ordered_actions(momdp), x), alphas.alpha_actions .+ 1)
    vis_state_map = broadcast(x -> getindex(ordered_states_x(momdp), x), alphas.alpha_visible_states .+ 1)
    
    return MOMDPAlphaVectorPolicy(momdp, alphas.alpha_vectors, action_map, vis_state_map)
end

"""
    load_policy(momdp::MOMDP, file_name::AbstractString)

Load a policy from an xml file output by SARSOP.
"""
function load_policy(momdp::MOMDP, file_name::AbstractString)
    alphas = nothing
    if isfile(file_name)
        alphas = MOMDPAlphas(file_name)
    else
        error("Policy file ", file_name, " does not exist")
    end
    action_map = broadcast(x -> getindex(ordered_actions(momdp), x), alphas.alpha_actions .+ 1)
    vis_state_map = broadcast(x -> getindex(ordered_states_x(momdp), x), alphas.alpha_visible_states .+ 1)
    return MOMDPAlphaVectorPolicy(momdp, alphas.alpha_vectors, action_map, vis_state_map)
end
