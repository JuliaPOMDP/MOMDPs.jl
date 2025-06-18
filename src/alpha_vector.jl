"""
    MOMDPAlphaVectorPolicy(momdp::MOMDP, alpha_vecs, action_map, vis_state_map)

Construct a policy from alpha vectors for a Mixed Observability Markov Decision Process (MOMDP).

# Arguments
- `momdp::MOMDP`: The MOMDP problem instance for which the policy is constructed.
- `alpha_vecs`: An abstract vector of alpha vectors, where each alpha vector is a vector of floats representing the value function for a particular belief state.
- `action_map`: A vector of actions corresponding to each alpha vector. Each action is associated with the alpha vector that prescribes it.
- `vis_state_map`: A vector mapping visible states to their corresponding alpha vectors, used to determine which alpha vector applies to a given state.

# Fields
- `momdp::MOMDP`: The MOMDP problem instance, necessary for mapping states to locations in the alpha vectors.
- `n_states_x::Int`: The number of visible states in the MOMDP.
- `n_states_y::Int`: The number of hidden states in the MOMDP.
- `alphas::Vector{Vector{Vector{Float64}}}`: A vector (of size ``|\\mathcal{X}|``) of vectors (number of alpha vectors) of vectors (number of alpha vectors) of alpha vectors (of size ``|\\mathcal{Y}|``). This structure holds the alpha vectors for each visible state.
- `action_map::Vector{Vector{A}}`: A vector (of size |X|) of vectors of actions corresponding to the alpha vectors. Each action is associated with a specific alpha vector for a visible state.

This structure represents a policy that uses alpha vectors to determine the best action to take given a belief state in a MOMDP.
"""
struct MOMDPAlphaVectorPolicy{M<:MOMDP,A} <: Policy
    momdp::M 
    n_states_x::Int
    n_states_y::Int
    alphas::Vector{Vector{Vector{Float64}}}
    action_map::Vector{Vector{A}}
end

function MOMDPAlphaVectorPolicy(m::MOMDP{X,Y,A,O}, alpha_vecs::AbstractVector, a_map, x_map) where {X,Y,A,O}
    n_states_x = length(states_x(m))
    n_states_y = length(states_y(m))

    alphas = Vector{Vector{Vector{Float64}}}(undef, n_states_x)
    action_map = Vector{Vector{A}}(undef, n_states_x)
    for (i, xi) in enumerate(states_x(m))
        n_x_alphas = count(x -> x == xi, x_map)
        n_x_idxs = findall(x -> x == xi, x_map)
        
        alphas[i] = Vector{Vector{Float64}}(undef, n_x_alphas)
        action_map[i] = Vector{A}(undef, n_x_alphas)
        for j = 1:n_x_alphas
            alphas[i][j] = vec(alpha_vecs[n_x_idxs[j]])
            action_map[i][j] = convert(actiontype(m), a_map[n_x_idxs[j]])
        end
    end
    MOMDPAlphaVectorPolicy(m, n_states_x, n_states_y, alphas, action_map)
end

# Assumes alpha vectors of size |Y| x (number of alpha vecs)
function MOMDPAlphaVectorPolicy(m::MOMDP, alpha_matrix::Matrix{Float64}, a_map, x_map)
    alpha_vecs = Vector{Vector{Float64}}(undef, size(alpha_matrix, 2))
    for i = 1:size(alpha_matrix, 2)
        alpha_vecs[i] = vec(alpha_matrix[:, i])
    end
    return MOMDPAlphaVectorPolicy(m, alpha_vecs, a_map, x_map)
end

POMDPTools.Policies.updater(p::MOMDPAlphaVectorPolicy) = DiscreteUpdater(p.momdp)

"""
Return an iterator of alpha vector-action pairs in the policy, given a visible state.
"""
function alphapairs(p::MOMDPAlphaVectorPolicy, x)
    yt = first(states_y(p.momdp))
    x_idx = stateindex_x(p.momdp, (x, yt))
    return (p.alphas[x_idx][i] => p.action_map[x_idx][i] for i in 1:length(p.alphas[x_idx]))
end

"""
Return the alpha vectors, given a visible state.
"""
function alphavectors(p::MOMDPAlphaVectorPolicy, x)
    yt = first(states_y(p.momdp))
    x_idx = stateindex_x(p.momdp, (x, yt))
    return p.alphas[x_idx]
end

"""
    value(p::MOMDPAlphaVectorPolicy, b)

Calculate the value of belief state `b` for a MOMDP using alpha vectors.

This function computes the value by:
1. Marginalizing the belief over the partially observable state to get ``b(x)`` for each fully observable state ``x``
2. Computing the conditional belief ``b(y \\mid x) = b(x,y)/b(x)`` for each ``x``
3. Finding the maximum dot product between ``b(y \\mid x)`` and the alpha vectors for each ``x``
4. Summing ``b(x) \\cdot V(x, b(y \\mid x))`` over all ``x`` to get the total value

# Arguments
- `p::MOMDPAlphaVectorPolicy`: The alpha vector policy
- `b`: The belief state over the joint state space (x,y)

# Returns
- The value of the belief state

# Notes
This is not the most efficient way to get a value if we are operating in a true MOMDP framework. However, this keeps the structure of the code similar to the POMDPs.jl framework.
"""
function POMDPTools.Policies.value(p::MOMDPAlphaVectorPolicy, b)
    sup = support(b)
    bx = zeros(p.n_states_x)
    for s in sup
        bx[stateindex_x(p.momdp, s)] += pdf(b, s)
    end

    V = 0.0
    for (x, bxi) in zip(states_x(p.momdp), bx)
        if bxi == 0
            continue
        end
        # We don't calculate b_{y|x} here because 
        # V(x, b_{y|x}) = 1/b_{x}(x) * max α ⋅ b(x, y)
        #   and 
        # V′(b) = ∑ b_{x}(x) * V(x, b_{y|x})
        Vyx = POMDPTools.Policies.value(p, b, x)
        V += Vyx
    end
    return V
end

"""
    value(p::MOMDPAlphaVectorPolicy, b, x)

Calculate the value for a specific visible state `x` given a belief `b` over the joint state space.

# Arguments
- `p::MOMDPAlphaVectorPolicy`: The alpha vector policy
- `b`: The belief distribution over the joint state space (x,y)
- `x`: The specific visible state to evaluate

# Returns
- The value for the given visible state and belief

# Description
This function computes the value by:
1. Extracting the marginal belief over hidden states `y` for the given visible state `x`
   by evaluating `pdf(b, (x,y))` for all `y`
2. Finding the maximum dot product between this marginal belief and all alpha vectors
   associated with visible state `x`

This is more efficient than the general `value(p, b)` function when the visible state
is known, as it only needs to consider alpha vectors for the specific `x` rather than
marginalizing over all visible states.
"""
function POMDPTools.Policies.value(p::MOMDPAlphaVectorPolicy, b, x)
    byx = zeros(p.n_states_y)
    for (j, y) in enumerate(states_y(p.momdp))
        byx[j] = pdf(b, (x, y))
    end
    
    Vyx = maximum(dot(byx, alpha) for alpha in alphavectors(p, x))
    return Vyx
end

"""
    action(p::MOMDPAlphaVectorPolicy, b)

Return the action prescribed by the MOMDP alpha-vector policy `p` for the belief `b`
over the joint state space (x, y).

# Heuristic
1. Find the visible state `x` with the largest probability mass in `b`.
2. Form the conditional distribution over `y` given that `x`.
3. Among the alpha-vectors in `p.alphas[x]`, pick the one with the largest dot product 
   with that conditional distribution.
4. Return the action associated with that alpha-vector.

# Notes

- **When ``b`` is not a pure distribution over a single ``x``:** Typically in a MOMDP, 
  we assume ``x`` (the "visible" state) is **fully** observed at runtime, so the 
  belief over ``(x, y)`` will place essentially all its probability mass on a 
  single ``x``. In that case, the above steps are effectively picking the single 
  ``x`` that we actually observed.
  
  If you are operating in a true MOMDP framework, you can implement a custom action function
  that takes in the visible state ``x`` and the conditional distribution over ``y`` given ``x``.
  While this would result in the same action as the heuristic above, it will be more
  efficient.
  E.g.: 
  ```julia
  # If ``x`` is known exactly at runtime and we have a distribution only over ``y``:
  function action(p::MOMDPAlphaVectorPolicy, x, by)
      x_idx = stateindex_x(p.momdp, x)
      # pick the alpha-vector among p.alphas[x_idx] that maximizes dot(alpha, by)
      ...
  end
  ```

- **In case your solver or simulator still gives a multi-modal distribution 
  over different ``x``** (which can happen in generic POMDP frameworks), the code 
  here picks the ``x`` that has the largest total probability mass in the belief. 
  While this heuristic might be sufficient for some problems, we recommend implementing
  a custom action function that performs a one-step lookahead using the `value` function.
"""
function POMDPTools.Policies.action(p::MOMDPAlphaVectorPolicy, b)
    # 1) Identify which x has the maximum mass in b
    sup_b = support(b)
    bx = zeros(p.n_states_x)
    for s in sup_b
        bx[stateindex_x(p.momdp, s)] += pdf(b, s)
    end
    
    best_x_idx = argmax(bx)
    best_x = states_x(p.momdp)[best_x_idx]
    
    # 2) Build the conditional distribution over y for that best_x
    # by[x_idx] = distribution over hidden states, given best_x
    by_vec = zeros(p.n_states_y)
    for s in sup_b
        if s[1] == best_x
            y_idx = stateindex_y(p.momdp, s)
            by_vec[y_idx] = pdf(b, s) / (bx[best_x_idx] + eps())
        end
    end

    # 3) Among alpha-vectors for best_x_idx, pick the one with max dot product with by_vec
    best_dot = -Inf
    best_dot_idx = 1
    for i in 1:length(p.alphas[best_x_idx])
        val = dot(p.alphas[best_x_idx][i], by_vec)
        if val > best_dot
            best_dot = val
            best_dot_idx = i
        end
    end

    # 4) Return the action associated with that alpha vector
    return p.action_map[best_x_idx][best_dot_idx]
end

"""
    action(p::MOMDPAlphaVectorPolicy, b, x)

Return the action prescribed by the MOMDP alpha-vector policy `p` for the belief `b`
over the hidden states, given the known visible state `x`.

# Arguments
- `p::MOMDPAlphaVectorPolicy`: The alpha vector policy
- `b`: The belief distribution over hidden states `y`
- `x`: The known visible state

# Returns
- The action for the given visible state and belief over hidden states

# Description
This function assumes the visible state `x` is fully observed and known. It:
1. Converts the belief `b` to a vector representation over hidden states
2. Among all alpha vectors associated with visible state `x`, finds the one with 
   the maximum dot product with the belief vector
3. Returns the action associated with that optimal alpha vector

This is more efficient than the version without explicit `x` when the visible state
is known at runtime, as it avoids the need to infer `x` from the belief distribution.
"""
function POMDPTools.Policies.action(p::MOMDPAlphaVectorPolicy, b, x)
    x_idx = stateindex_x(p.momdp, x)
    by_vec = beliefvec_y(p.momdp, p.n_states_y, b)

    best_dot = -Inf
    best_dot_idx = 1
    for i in 1:length(p.alphas[x_idx])
        val = dot(p.alphas[x_idx][i], by_vec)
        if val > best_dot
            best_dot = val
            best_dot_idx = i
        end
    end
    return p.action_map[x_idx][best_dot_idx]
end

"""
    actionvalues(p::MOMDPAlphaVectorPolicy, b, x)

Compute the action values (Q-values) for all actions given a belief `b` over hidden 
states and a known visible state `x`.

# Arguments
- `p::MOMDPAlphaVectorPolicy`: The alpha vector policy
- `b`: The belief distribution over hidden states `y`  
- `x`: The known visible state

# Returns
- A vector of action values where the ith element is the Q-value for action `i`

# Description
This function performs a one-step lookahead to compute action values by:
1. For each action `a` and each hidden state `y` in the belief support:
   - Computing the immediate reward `R(x,y,a)`
   - For each possible next visible state `x'` and hidden state `y'`:
     - For each possible observation `o`:
       - Updating the belief to get `b'`
       - Computing the value using the alpha vectors for `x'`
       - Accumulating the discounted expected future value
2. Summing over all transitions weighted by their probabilities

The resulting Q-values can be used for action selection or policy evaluation when
the visible state is known.
"""
function POMDPTools.Policies.actionvalues(p::MOMDPAlphaVectorPolicy, b, x)
    na = length(actions(p.momdp))
    γ = discount(p.momdp)
    qa = zeros(na)
    up = MOMDPDiscreteUpdater(p.momdp)
    for ai in 1:na
        for (yi, yprob) in weighted_iterator_y(b)
            rew_sum = reward(p.momdp, (x, yi), ai)
            for (xpi, xpprob) in weighted_iterator(transition_x(p.momdp, (x, yi), ai))
                xpprob == 0.0 && continue
                for (ypi, ypprob) in weighted_iterator_y(transition_y(p.momdp, (x, yi), ai, xpi))
                    ypprob == 0.0 && continue
                    for (oi, oprob) in weighted_iterator(observation(p.momdp, ai, (xpi, ypi)))
                        oprob == 0.0 && continue
                        bp = update(up, b, ai, oi, x, xpi)
                        v = maximum(dot(bp.b, alpha) for alpha in MOMDPs.alphavectors(p, xpi))
                        rew_sum += γ * xpprob * ypprob * oprob * v                        
                    end
                end
            end
            qa[ai] += rew_sum * yprob
        end
    end
    return qa
end 

"""
    beliefvec_y(m::MOMDP, n_states::Int, b)

Convert a belief distribution `b` over hidden states to a vector representation suitable 
for dot product operations with alpha vectors in a MOMDP.

# Arguments
- `m::MOMDP`: The MOMDP problem instance
- `n_states_y::Int`: The number of hidden states in the MOMDP
- `b`: The belief distribution over hidden states (supports various belief types)

# Returns
- A vector of length `n_states_y` where element `i` represents the probability of hidden state `i`

# Supported Belief Types
- `SparseCat`: Converts sparse categorical distribution to dense vector
- `AbstractVector`: Returns the vector directly (with length assertion)
- `DiscreteBelief`: Extracts the underlying probability vector
- `Deterministic`: Creates a one-hot vector for the deterministic state

This function is used internally by alpha vector policies to convert belief distributions
into the vector format required for computing dot products with alpha vectors.
"""
function beliefvec_y end

function beliefvec_y(m::MOMDP, n_states_y, b::SparseCat)
    b′ = zeros(n_states_y)
    for (yi, byi) in zip(b.vals, b.probs)
        y_idx = stateindex_y(m, yi)
        b′[y_idx] = byi
    end
    return b′
end
function beliefvec_y(m::MOMDP, n_states_y, b::AbstractVector)
    @assert length(b) == n_states_y
    return b
end
function beliefvec_y(m::MOMDP, n_states_y, b::DiscreteBelief)
    @assert length(b) == length(b.b)
    return b.b
end
function beliefvec_y(m::MOMDP, n_states_y, b::Deterministic)
    y_idx = stateindex_y(m, b.val)
    b′ = zeros(n_states_y)
    b′[y_idx] = 1.0
    return b′
end

weighted_iterator_y(d) = (x=>pdf_y(d, x) for x in support(d))
