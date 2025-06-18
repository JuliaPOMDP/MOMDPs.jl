# Belief Updaters

The only updater currently implemented is the [`MOMDPs.MOMDPDiscreteUpdater`](@ref).

## `MOMDPDiscreteUpdater`

The [`MOMDPs.MOMDPDiscreteUpdater`](@ref) is a discrete belief updater for MOMDPs. The [`update`](@ref) function implements the discrete Bayesian filter for MOMDPs, which updates beliefs over hidden states given knowledge of visible state transitions. The `update` function requires the current belief `b`, the action taken `a`, the observation received `o`, the visible state from which the action was taken `x`, and the visible state that we transitioned to `xp`.

### Example Usage
```julia
momdp = YourMOMDPProblem()
updater = MOMDPDiscreteUpdater(momdp)

# Initialize belief over hidden states
initial_dist = initialstate_y(momdp)

# Update belief after taking action and receiving observation
new_belief = update(updater, current_belief, action_taken, observation_received, x, xp)
```