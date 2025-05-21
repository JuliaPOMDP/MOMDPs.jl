"""
    POMDP_of_Discrete_MOMDP{X,Y,A,O} <: POMDP{Tuple{X,Y},A,O}

This type is used to convert a Discrete MOMDP into a POMDP. There are functions defined for this POMDP that use the functions defined for the MOMDP type. 

The only difference in the spaces is the observation space. If the original observation space was of size ``\\mathcal{O}`` and the visible state space was of size ``\\mathcal{X}``, then the observation space of the POMDP is ``\\mathcal{X} \\times \\mathcal{O}``.
"""
struct POMDP_of_Discrete_MOMDP{X,Y,A,O} <: POMDP{Tuple{X,Y},A,O}
    momdp::MOMDP{X,Y,A,O}
end

POMDPs.states(p::POMDP_of_Discrete_MOMDP) = states(p.momdp)
POMDPs.stateindex(p::POMDP_of_Discrete_MOMDP{X,Y,A,O}, s::Tuple{X,Y}) where {X,Y,A,O} = stateindex(p.momdp, s)
POMDPs.actions(p::POMDP_of_Discrete_MOMDP) = actions(p.momdp)
POMDPs.actionindex(p::POMDP_of_Discrete_MOMDP{X,Y,A,O}, a::A) where {X,Y,A,O} = actionindex(p.momdp, a)
POMDPs.transition(p::POMDP_of_Discrete_MOMDP{X,Y,A,O}, s::Tuple{X,Y}, a::A) where {X,Y,A,O} = transition(p.momdp, s, a)
POMDPs.reward(p::POMDP_of_Discrete_MOMDP{X,Y,A,O}, s::Tuple{X,Y}, a::A) where {X,Y,A,O} = reward(p.momdp, s, a)
POMDPs.discount(p::POMDP_of_Discrete_MOMDP) = discount(p.momdp)
POMDPs.isterminal(p::POMDP_of_Discrete_MOMDP{X,Y,A,O}, s::Tuple{X,Y}) where {X,Y,A,O} = isterminal(p.momdp, s)
POMDPs.initialstate(p::POMDP_of_Discrete_MOMDP) = initialstate(p.momdp)

"""
    POMDPs.observations(p::POMDP_of_Discrete_MOMDP)

Returns the full observation space of a POMDP_of_Discrete_MOMDP. The observations are `Tuple{X,O}` where `X` is the visible state and `O` is the observation.
"""
function POMDPs.observations(p::POMDP_of_Discrete_MOMDP)
    x_states = states_x(p.momdp)
    obs = observations(p.momdp)
    return vec([(x, o) for x in x_states for o in obs])
end

"""
    POMDPs.obsindex(p::POMDP_of_Discrete_MOMDP, o)

Returns the index of the `Tuple{X,O}` observation for a POMDP_of_Discrete_MOMDP.
"""
function POMDPs.obsindex(p::POMDP_of_Discrete_MOMDP, o)
    n_x_states = length(states_x(p.momdp))
    n_obs_momdp = length(observations(p.momdp))
    
    x_idx = stateindex_x(p.momdp, o[1])
    o_idx = obsindex(p.momdp, o[2])

    return LinearIndices((n_x_states, n_obs_momdp))[x_idx, o_idx]
end

"""
    POMDPs.observation(p::POMDP_of_Discrete_MOMDP{X, Y, A, O}, a::A, s::Tuple{X, Y}) where {X, Y, A, O}

Returns the full observation distribution for a POMDP_of_Discrete_MOMDP. The observations are `Tuple{X,O}` where `X` is the visible state and `O` is the observation.
"""
function POMDPs.observation(p::POMDP_of_Discrete_MOMDP{X, Y, A, O}, a::A, s::Tuple{X, Y}) where {X, Y, A, O}
    obs_dist = observation(p.momdp, a, s)
    poss_obs = support(obs_dist)
    new_obs = [(s[1], o) for o in poss_obs]
    new_weights = [pdf(obs_dist, o) for o in poss_obs]
    return SparseCat(new_obs, new_weights)
end

POMDPTools.ordered_observations(p::POMDP_of_Discrete_MOMDP) = observations(p)
