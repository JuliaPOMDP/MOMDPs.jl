struct POMDP_of_Discrete_MOMDP{X,Y,A,O} <: POMDP{Tuple{X,Y},A,O}
    momdp::MOMDP{X,Y,A,O}
end

POMDPs.states(p::POMDP_of_Discrete_MOMDP) = states(p.momdp)
POMDPs.stateindex(p::POMDP_of_Discrete_MOMDP, s) = stateindex(p.momdp, s)
POMDPs.actions(p::POMDP_of_Discrete_MOMDP) = actions(p.momdp)
POMDPs.actionindex(p::POMDP_of_Discrete_MOMDP, a) = actionindex(p.momdp, a)
POMDPs.transition(p::POMDP_of_Discrete_MOMDP, s, a) = transition(p.momdp, s, a)
POMDPs.reward(p::POMDP_of_Discrete_MOMDP, s, a) = reward(p.momdp, s, a)
POMDPs.discount(p::POMDP_of_Discrete_MOMDP) = discount(p.momdp)
POMDPs.isterminal(p::POMDP_of_Discrete_MOMDP, s) = isterminal(p.momdp, s)
POMDPs.initialstate(p::POMDP_of_Discrete_MOMDP) = initialstate(p.momdp)

function POMDPs.observations(p::POMDP_of_Discrete_MOMDP)
    x_states = states_x(p.momdp)
    obs = observations(p.momdp)
    return vec([(x, o) for x in x_states for o in obs])
end

function POMDPs.obsindex(p::POMDP_of_Discrete_MOMDP, o)
    n_x_states = length(states_x(p.momdp))
    n_obs_momdp = length(observations(p.momdp))
    
    x_idx = stateindex_x(p.momdp, o[1])
    o_idx = obsindex(p.momdp, o[2])

    return LinearIndices((n_x_states, n_obs_momdp))[x_idx, o_idx]
end

function POMDPs.observation(p::POMDP_of_Discrete_MOMDP{X, Y, A, O}, a::A, s::Tuple{X, Y}) where {X, Y, A, O}
    obs_dist = observation(p.momdp, a, s)
    poss_obs = support(obs_dist)
    new_obs = [(s[1], o) for o in poss_obs]
    new_weights = [pdf(obs_dist, o) for o in poss_obs]
    return SparseCat(new_obs, new_weights)
end

POMDPTools.ordered_observations(p::POMDP_of_Discrete_MOMDP) = observations(p)
