# Discrete Helper Functions

The `MOMDPs.jl` package provides helper functions for discrete MOMDPs. These functions use the required functions to define a MOMDP to return the full state space, state index, transition distribution, and initial state distribution.

## States
The function [`states(p::MOMDP)`](@ref) returns the full state space for a discrete MOMDP where the states are tuples of the visible state and the hidden state.

## State Index
The function [`stateindex(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}) where {X,Y,A,O}`](@ref) returns the index of the Tuple{X,Y} state for a discrete MOMDP.

## Transition Distribution
The function [`transition(p::MOMDP{X,Y,A,O}, s::Tuple{X,Y}, a::A) where {X,Y,A,O}`](@ref) returns the full transition distribution for a discrete MOMDP. The states are Tuple{X,Y}. It uses `transition_x` and `transition_y` to construct the distribution.

## Initial State Distribution
The function [`initialstate(p::MOMDP)`](@ref) returns the initial state distribution for a discrete MOMDP. The states are Tuple{X,Y}. It uses `initialstate_x` and `initialstate_y` to construct the distribution.
