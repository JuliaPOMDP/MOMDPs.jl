"""
    POMDPs.states(p::MOMDP)

Helper function to return the full state space for discrete MOMDPs. The states are
    Tuple{X,Y} where X is the visible state and Y is the hidden state.
"""
function POMDPs.states(p::MOMDP)
    x_states = states_x(p)
    y_states = states_y(p)
    return vec([(x, y) for x in x_states, y in y_states])
end

"""
    POMDPs.stateindex(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}) where {X,Y,A,O}

Helper function to return the index of the Tuple{X,Y} state for discrete MOMDPs.
"""
function POMDPs.stateindex(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}) where {X,Y,A,O}
    x_idx = stateindex_x(p, s)
    y_idx = stateindex_y(p, s)
    n_states_x = length(states_x(p))
    n_states_y = length(states_y(p))
    return LinearIndices((n_states_x, n_states_y))[x_idx, y_idx]
end

"""
    POMDPs.transition(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}, a::A) where {X,Y,A,O}

Helper function to return the full transition distribution for discrete MOMDPs. The states 
    are Tuple{X,Y}. It uses `transition_x` and `transition_y` to construct the
    distribution.
"""
function POMDPs.transition(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}, a::A) where {X,Y,A,O}    
    x_dist = transition_x(p, s, a)
    
    # Use dictionary to accumulate weights
    state_weights = Dict{Tuple{X,Y}, Float64}()
    
    for (x_i, w_i) in weighted_iterator(x_dist)
        y_dist = transition_y(p, s, a, x_i)
        for (y_i, w_y) in weighted_iterator(y_dist)
            new_state = (x_i, y_i)
            weight = w_i * w_y
            state_weights[new_state] = get(state_weights, new_state, 0.0) + weight
        end
    end
    
    # Convert to vectors for SparseCat
    new_states = collect(keys(state_weights))
    new_weights = collect(values(state_weights))
    
    return SparseCat(new_states, new_weights)
end

"""
    initialstate(p::MOMDP{X,Y,A,O}) where {X,Y,A,O}

Helper function to return the initial state distribution for discrete MOMDPs. The states 
    are Tuple{X,Y}. It uses `initialstate_x` and `initialstate_y` to construct the
    distribution.
"""
function POMDPs.initialstate(p::MOMDP{X,Y,A,O}) where {X,Y,A,O}
    x_support = support(initialstate_x(p))
    
    probs = Float64[]
    poss_states = Tuple{X,Y}[]
    
    for xi in x_support
        xi_p = pdf(initialstate_x(p), xi)
        y_support = support(initialstate_y(p, xi))
        for yi in y_support
            yi_p = pdf(initialstate_y(p, xi), yi)
            push!(poss_states, (xi, yi))
            push!(probs, xi_p * yi_p)
        end
    end
    return SparseCat(poss_states, probs)
end
