"""
    MOMDPDiscreteUpdater

An updater type for maintaining and updating discrete beliefs over hidden states in a Mixed Observability Markov Decision Process (MOMDP).

# Constructor
    MOMDPDiscreteUpdater(momdp::MOMDP)

Create a discrete belief updater for the given MOMDP.

# Fields
- `momdp <: MOMDP`: The MOMDP problem instance for which beliefs will be updated

# Description
In a MOMDP, the state space is factored into visible states `x` (fully observable) and 
hidden states `y` (partially observable). This updater maintains beliefs only over the 
hidden states, since the visible states are assumed to be directly observed.

The updater implements the discrete Bayesian filter for belief updates, assuming:
- Finite, discrete hidden state spaces
- Known visible state transitions (x → x')
- Probabilistic hidden state transitions that may depend on visible states
- Observations that depend on both visible and hidden states

# Usage
```julia
momdp = YourMOMDPProblem()
updater = MOMDPDiscreteUpdater(momdp)

# Initialize belief over hidden states
initial_dist = initialstate_y(momdp)

# Update belief after taking action and receiving observation
new_belief = update(updater, current_belief, action, observation, x, xp)
```

# See Also
- `uniform_belief_y`: Create uniform beliefs over hidden states
- `initialize_belief`: Initialize beliefs from distributions
- `update`: Perform belief updates using the MOMDP filter
"""
mutable struct MOMDPDiscreteUpdater{P<:MOMDP} <: Updater
    momdp::P
end

"""
     uniform_belief_y(momdp)
     uniform_belief_y(up::MOMDPDiscreteUpdater)

Return a uniform DiscreteBelief over all hidden states in the MOMDP.

# Arguments
- `momdp`: A MOMDP problem instance, or
- `up::MOMDPDiscreteUpdater`: A MOMDP discrete belief updater

# Returns
- A `DiscreteBelief` with equal probability `1/|Y|` for each hidden state `y ∈ Y`

# Description
This function creates a uniform prior belief over the hidden state space, which is often 
used as an initial belief when the true hidden state is unknown. In MOMDP problems, this 
represents maximum uncertainty about the hidden state while the visible state is assumed 
to be known.

The uniform belief is particularly useful for:
- Initializing belief when no prior information is available
- Baseline comparisons in experiments
- Worst-case analysis of belief-dependent policies

# Example
```julia
momdp = YourMOMDPProblem()
initial_belief = uniform_belief_y(momdp)
# or
updater = MOMDPDiscreteUpdater(momdp)
initial_belief = uniform_belief_y(updater)
```
"""
function uniform_belief_y(momdp)
    state_list = ordered_states_y(momdp)
    ny = length(state_list)
    return DiscreteBelief(momdp, state_list, ones(ny) / ny)
end
uniform_belief_y(up::MOMDPDiscreteUpdater) = uniform_belief_y(up.momdp)

"""
    initialize_belief(bu::MOMDPDiscreteUpdater, dist::Any)

Initialize a discrete belief over hidden states from a given distribution.

# Arguments
- `bu::MOMDPDiscreteUpdater`: The MOMDP discrete belief updater
- `dist`: A distribution over hidden states to initialize from (supports various distribution types)

# Returns
- A `DiscreteBelief` over the hidden state space with probabilities initialized from `dist`

# Description
This function creates a discrete belief representation over the hidden states `y` suitable 
for use with MOMDP belief update operations. The conversion process:

1. Creates a zero-initialized probability vector over all hidden states in the MOMDP
2. For each state `y` in the support of the input distribution, extracts the probability 
   `pdf(dist, y)` and assigns it to the corresponding index in the belief vector
3. Returns a `DiscreteBelief` object that can be used with the MOMDP updater

# Supported Distribution Types
The function can handle various distribution types through the generic `pdf` interface:
- Discrete distributions (e.g., `Categorical`, `DiscreteUniform`)
- Custom distributions that implement `pdf` and `support`
- Sparse distributions with limited support

# Usage Examples
```julia
updater = MOMDPDiscreteUpdater(momdp)

# From a uniform distribution
uniform_dist = DiscreteUniform(1, length(states_y(momdp)))
belief = initialize_belief(updater, uniform_dist)

# From a sparse categorical distribution  
sparse_dist = SparseCat([state1, state3], [0.7, 0.3])
belief = initialize_belief(updater, sparse_dist)
```

# Implementation Notes
- Uses `stateindex_y` to map hidden states to belief vector indices
- Assumes the distribution is over individual hidden states, not joint (x,y) states
- The resulting belief is properly normalized if the input distribution is normalized
"""
function POMDPTools.BeliefUpdaters.initialize_belief(bu::MOMDPDiscreteUpdater, dist::Any)
    state_list = ordered_states_y(bu.momdp)
    ns = length(state_list)
    b = zeros(ns)
    belief = DiscreteBelief(bu.momdp, state_list, b)
    temp_x = first(states_x(bu.momdp))
    for y in support(dist)
        yidx = stateindex_y(bu.momdp, (temp_x, y))
        belief.b[yidx] = pdf_y(dist, y)
    end
    return belief
end

pdf_y(dist::Any, y) = pdf(dist, y)
pdf_y(b::DiscreteBelief, x, y) = b.b[stateindex_y(b.pomdp, (x, y))]
function pdf_y(b::DiscreteBelief, y)
    x_temp = first(states_x(b.pomdp))
    return pdf_y(b, x_temp, y)
end


"""
    update(bu::MOMDPDiscreteUpdater, b::DiscreteBelief, a, o, x, xp)

Update a discrete belief over hidden states using the MOMDP belief update equation.

# Arguments
- `bu::MOMDPDiscreteUpdater`: The MOMDP discrete belief updater
- `b::DiscreteBelief`: The current belief over hidden states `y`
- `a`: The action taken
- `o`: The observation received after taking action `a`
- `x`: The previous visible state
- `xp`: The current visible state

# Returns
- A new `DiscreteBelief` representing the updated belief over hidden states

# Description
This function implements the discrete Bayesian filter for MOMDPs, which updates beliefs
over hidden states given knowledge of visible state transitions.

The function iterates through all current hidden states with non-zero probability,
computes transition probabilities to next hidden states, weights by observation
probabilities, and normalizes the result.

# Errors
Throws an error if the updated belief probabilities sum to zero, which indicates
an impossible observation given the current belief and action.
"""
function POMDPTools.BeliefUpdaters.update(bu::MOMDPDiscreteUpdater, b::DiscreteBelief, a, o, x, xp)
    momdp = bu.momdp
    state_space_y = b.state_list
    bp = zeros(length(state_space_y))

    for (yi, y) in enumerate(state_space_y)
        if b.b[yi] > 0.0
            td = transition_y(momdp, (x, y), a, xp)

            for (yp, tp) in weighted_iterator(td)
                ypi = stateindex_y(momdp, (xp, yp))
                op = obs_weight(momdp, (x, y), a, (xp, yp), o) # shortcut for observation probability from POMDPModelTools

                bp[ypi] += op * tp * b.b[yi]
            end
        end
    end

    bp_sum = sum(bp)

    if bp_sum == 0.0
        error("""
              Failed discrete belief update: new probabilities sum to zero.

              b = $b
              a = $a
              o = $o
              x = $x
              xp = $xp

              Failed discrete belief update: new probabilities sum to zero.
              """)
    end

    # Normalize
    bp ./= bp_sum

    return DiscreteBelief(momdp, b.state_list, bp)
end

"""
    update(bu::MOMDPDiscreteUpdater, b::Any, a, o, x, xp)

This is a convenience method that handles arbitrary belief types by first calling
`initialize_belief` to convert them to a `DiscreteBelief`, then performing the
standard discrete belief update.
"""
POMDPTools.BeliefUpdaters.update(bu::MOMDPDiscreteUpdater, b::Any, a, o, x, xp) = update(bu, initialize_belief(bu, b), a, o, x, xp)
