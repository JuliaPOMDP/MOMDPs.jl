"""
    MOMDP{X, Y, A, O} <: POMDP{Tuple{X,Y},A,O}

Abstract base type for a mixed observable Markov decision process.

    X: visible state type
    Y: hidden state type
    A: action type
    O: observation type
    
Notation matching Ong, Sylvie CW, et al. "POMDPs for robotic tasks with mixed observability." Robotics: Science and systems. Vol. 5. No. 4. 2009. [link](https://www.comp.nus.edu.sg/~leews/publications/rss09.pdf)
"""
abstract type MOMDP{X,Y,A,O} <: POMDP{Tuple{X,Y},A,O} end


"""
    transition_x(m::MOMDP{X,Y,A,O}, state::Tuple{X,Y}, action)
    
Return the transition distribution over the next visible state given the current state and action.

T_x(s, a, x′) = p(x′ | s, a) where s = (x,y)
"""
function transition_x end

"""
    transition_y(m::MOMDP{X,Y,A,O}, state::Tuple{X,Y}, action, statep_visible)
    
Return the transition distribution over the next hidden state given the current state, action, and next visible state.

T_y(s, a, x′, y′) = p(y′ | s, a, x′) where s = (x,y)
"""
function transition_y end

"""
    stateindex_x(problem::MOMDP, s)

Return the integer index of the visible state `x` where `s` is a tuple of the form `(x,y)`. Used for discrete models only.
"""
function stateindex_x end

"""
    stateindex_y(problem::MOMDP, s)

Return the integer index of the hidden state `y` where `s` is a tuple of the form `(x,y)`. Used for discrete models only.
"""
function stateindex_y end

"""
    states_x(problem::MOMDP)
    
Returns the complete visible state space of a MOMDP.
"""
function states_x end

"""
    states_y(problem::MOMDP)
    
Returns the complete hidden state space of a MOMDP.
"""
function states_y end

"""
    initialstate_x(problem::MOMDP)

Return the initial visible state distribution.
"""
function initialstate_x end

"""
    initialstate_y(problem::MOMDP, x)

Return the initial hidden state distribution conditioned on the visible state `x`.
"""
function initialstate_y end

"""
    statetype_x(t::Type)
    statetype_x(p::MOMDP)
    
Return the visible state type for a MOMDP (the `X` in `MOMDP{X,Y,A,O}`).

"""
statetype_x(t::Type) = statetype_x(supertype(t))
statetype_x(t::Type{MOMDP{X,Y,A,O}}) where {X,Y,A,O} = X
statetype_x(t::Type{Any}) = error("Attempted to extract the visible state type for $t. This is not a subtype of `MOMDP`. Did you declare your problem type as a subtype of `MOMDP{X,Y,A,O}`?")
statetype_x(p::MOMDP) = statetype_x(typeof(p))

"""
    statetype_y(t::Type)
    statetype_y(p::MOMDP)

Return the hidden state type for a MOMDP (the `Y` in `MOMDP{X,Y,A,O}`).
"""
statetype_y(t::Type) = statetype_y(supertype(t))
statetype_y(t::Type{MOMDP{X,Y,A,O}}) where {X,Y,A,O} = Y
statetype_y(t::Type{Any}) = error("Attempted to extract the hidden state type for $t. This is not a subtype of `MOMDP`. Did you declare your problem type as a subtype of `MOMDP{X,Y,A,O}`?")
statetype_y(p::MOMDP) = statetype_y(typeof(p))


"""
    ordered_states_x(momdp)
    
Return an `AbstractVector` of the visible states in a `MOMDP` ordered according to `stateindex_x(momdp, s)`.

`ordered_states_x(momdp)` will always return a `AbstractVector{X}` `v` containing all of the visible states in `states_x(momdp)` in the order such that `stateindex_x(momdp, v[i]) == i`. You may wish to override this for your problem for efficiency.
"""
ordered_states_x(momdp::MOMDP) = ordered_vector(statetype_x(typeof(momdp)), s->stateindex_x(momdp,s), states_x(momdp), "x")

"""
    ordered_states_y(momdp)

Return an `AbstractVector` of the hidden states in a `MOMDP` ordered according to `stateindex_y(momdp, s)`.

`ordered_states_y(momdp)` will always return a `AbstractVector{Y}` `v` containing all of the hidden states in `states_y(momdp)` in the order such that `stateindex_y(momdp, v[i]) == i`. You may wish to override this for your problem for efficiency.
"""
ordered_states_y(momdp::MOMDP) = ordered_vector(statetype_y(typeof(momdp)), s->stateindex_y(momdp,s), states_y(momdp), "y")

# `ordered_vector` is a function defined in POMDPTools.jl but not exported.
# TODO: Export ordered_vector from POMDPTools.jl and use that instead.
function ordered_vector(T::Type, index::Function, space, singular, plural=singular*"s")
    len = length(space)
    a = Array{T}(undef, len)
    gotten = falses(len)
    for x in space
        id = index(x)
        if id > len || id < 1
            error("""
                  stateindex_$(singular)(...) returned an index that was out of bounds for state_$(singular) $x.

                  index was $id.

                  n_states_$(singular)(...) was $len.
                  """) 
        end
        a[id] = x
        gotten[id] = true
    end
    if !all(gotten)
        missing = findall(.!gotten)
        @warn """
             Problem creating an ordered vector of state_$(singular)s in ordered_states_$(singular)(...). There is likely a mistake in stateindex_$(singular)(...) or n_states_$(singular)(...).

             n_states_$(singular)(...) was $len.

             state_$(singular)s corresponding to the following indices were missing from state_$(singular)(...): $missing
             """
    end
    return a
end

@POMDP_require ordered_states_x(momdp::MOMDP) begin
    P = typeof(momdp)
    @req stateindex_x(::P, ::statetype_x(P))
    @req states_x(::P)
    sxs = states_x(momdp)
    @req length(::typeof(sxs))
end

@POMDP_require ordered_states_y(momdp::MOMDP) begin
    P = typeof(momdp)
    @req stateindex_y(::P, ::statetype_y(P))
    @req states_y(::P)
    sys = states_y(momdp)
    @req length(::typeof(sys))
end

"""
    is_y_prime_dependent_on_x_prime(m::MOMDP)

Defines if the next hidden state `y′` depends on the next visible state `x′` given the current visible state `x`, hidden state `y`, and action `a`.

Return `false` if the conditional probability distribution satisfies:
`p(y′ | x, y, a, x′) = p(y′ | x, y, a)`.
"""
is_y_prime_dependent_on_x_prime(::MOMDP) = true

"""
    is_x_prime_dependent_on_y(m::MOMDP)

Defines if the next visible state `x′` depends on the current hidden state `y` given the current visible state `x` and action `a`.

Returns `false` if the conditional probability distribution satisfies:
`p(x′ | x, y, a) = p(x′ | x, a)`.
"""
is_x_prime_dependent_on_y(::MOMDP) = true

"""
    is_initial_distribution_independent(m::MOMDP)

Defines whether the initial distributions of the visible state `x` and hidden state `y` are independent.

Returns `true` if the joint probability distribution satisfies:
`p(x, y) = p(x)p(y)`, meaning `x` and `y` are independent in the initial distribution.
"""
is_initial_distribution_independent(::MOMDP) = true
