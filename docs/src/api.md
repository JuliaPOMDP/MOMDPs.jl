# API

## Exported Types

```@docs
MOMDPs.MOMDP
MOMDPs.POMDP_of_Discrete_MOMDP
MOMDPs.MOMDPAlphaVectorPolicy
MOMDPs.MOMDPDiscreteUpdater
```

## Exported Functions

```@docs
MOMDPs.transition_x
MOMDPs.transition_y
MOMDPs.states_x
MOMDPs.states_y
MOMDPs.stateindex_x
MOMDPs.stateindex_y
MOMDPs.initialstate_x
MOMDPs.initialstate_y
MOMDPs.statetype_x
MOMDPs.statetype_y
MOMDPs.ordered_states_x
MOMDPs.ordered_states_y
MOMDPs.is_y_prime_dependent_on_x_prime
MOMDPs.is_x_prime_dependent_on_y
MOMDPs.is_initial_distribution_independent
MOMDPs.beliefvec_y
MOMDPs.uniform_belief_y
```

## Extended Functions

### POMDPs.jl and POMDPTools.jl
```@docs
POMDPs.states(p::MOMDP)
POMDPs.stateindex(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}) where {X,Y,A,O}
POMDPs.transition(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}, a::A) where {X,Y,A,O}
POMDPs.initialstate(p::MOMDP{X,Y,A,O}) where {X,Y,A,O}
POMDPs.observations(p::POMDP_of_Discrete_MOMDP)
POMDPs.observation(p::POMDP_of_Discrete_MOMDP{X, Y, A, O}, a::A, s::Tuple{X,Y}) where {X,Y,A,O}
POMDPs.obsindex(p::POMDP_of_Discrete_MOMDP, o)
POMDPTools.Policies.value(p::MOMDPAlphaVectorPolicy, b)
POMDPTools.Policies.value(p::MOMDPAlphaVectorPolicy, b, x)
POMDPTools.Policies.action(p::MOMDPAlphaVectorPolicy, b)
POMDPTools.Policies.action(p::MOMDPAlphaVectorPolicy, b, x)
POMDPTools.Policies.actionvalues(p::MOMDPAlphaVectorPolicy, b, x)
POMDPTools.BeliefUpdaters.initialize_belief(bu::MOMDPDiscreteUpdater, dist::Any)
POMDPTools.BeliefUpdaters.update(bu::MOMDPDiscreteUpdater, b::DiscreteBelief, a, o, x, xp)
POMDPTools.BeliefUpdaters.update(bu::MOMDPDiscreteUpdater, b::Any, a, o, x, xp)
```

## Internal Functions
```@docs
MOMDPs.alphapairs
MOMDPs.alphavectors
```