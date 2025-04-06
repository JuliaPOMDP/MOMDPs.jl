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
- `alphas::Vector{Vector{Vector{Float64}}}`: A vector (of size |X|) of vectors (number of alpha vectors) of vectors (number of alpha vectors) of alpha vectors (of size |Y|). This structure holds the alpha vectors for each visible state.
- `action_map::Vector{Vector{A}}`: A vector (of size |X|) of vectors of actions corresponding to the alpha vectors. Each action is associated with a specific alpha vector for a visible state.

This structure represents a policy that uses alpha vectors to determine the best action to take given a belief state in a MOMDP.
"""
struct MOMDPAlphaVectorPolicy{M<:MOMDP,A} <: Policy
    momdp::M # needed for mapping states to locations in alpha vectors
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
    x_idx = stateindex_x(p.momdp, x)
    return (p.alphas[x_idx][i] => p.action_map[x_idx][i] for i in 1:length(p.alphas[x_idx]))
end

"""
Return the alpha vectors, given a visible state.
"""
function alphavectors(p::MOMDPAlphaVectorPolicy, x)
    x_idx = stateindex_x(p.momdp, x)
    return p.alphas[x_idx]
end



# We can execute a MOMDP like a POMDP. To do this, we first need to determine b(x) for each
# x by marginalizing over y. Then b(y | x) = b(x,y) / b(x). From that, we can then determine
# V(x, byy) = max_alpha dot(b(y | x), alpha) for alpha in alphavectors(p, x). Then, we can
# determine V(b) = sum b(x) * V(x, byx).
function POMDPTools.Policies.value(p::MOMDPAlphaVectorPolicy, b)
    sup = support(b)
    bx = zeros(p.n_states_x)
    for s in sup
        bx[stateindex_x(p.momdp, s[1])] += pdf(b, s)
    end

    V = 0.0
    statesY = states_y(p.momdp)
    for (x, bxi) in zip(states_x(p.momdp), bx)
        if bxi == 0
            continue
        end
        byx = zeros(p.n_states_y)

        for (j, y) in enumerate(statesY)
            byx[j] = pdf(b, (x, y))
        end

        Vyx = maximum(dot(byx, alpha) for alpha in alphavectors(p, x))
        V += Vyx
    end
    return V
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

- **When `b` is not a pure distribution over a single `x`:** Typically in a MOMDP, 
  we assume (x) (the "visible" state) is **fully** observed at runtime, so the 
  belief over (x, y) will place essentially all its probability mass on a 
  single (x). In that case, the above steps are effectively picking the single 
  (x) that we actually observed.
  
  If you are operating in a true MOMDP framework, you can implement a custom action function
  that takes in the visible state `x` and the conditional distribution over `y` given `x`.
  While this would result in the same action as the heuristic above, it will be more
  efficient.
  E.g.: 
  ```julia
  # If x is known exactly at runtime and we have a distribution only over y:
  function action(p::MOMDPAlphaVectorPolicy, x, by_over_y)
      x_idx = stateindex_x(p.momdp, x)
      # pick the alpha-vector among p.alphas[x_idx] that maximizes dot(alpha, by_over_y)
      ...
  end
  ```

- **In case your solver or simulator still gives a multi-modal distribution 
  over different `x`** (which can happen in generic POMDP frameworks), the code 
  here picks the `x` that has the largest total probability mass in the belief. 
  While this heuristic might be sufficient for some problems, we recommend implementing
  a custom action function that performs a one-step lookahead using the `value` function.
"""
function POMDPs.action(p::MOMDPAlphaVectorPolicy, b)
    # 1) Identify which x has the maximum mass in b
    sup_b = support(b)
    best_x = nothing
    best_x_prob = 0.0
    for s in sup_b
        # s is a tuple (x, y)
        # Probability mass for (x, y)
        prob_s = pdf(b, s)
        if prob_s > 0.0 && prob_s + 1e-12 > best_x_prob
            best_x_prob = prob_s
            best_x = s[1]
        end
    end

    # Edge case: if belief is all zero, throw an error
    if best_x === nothing
        error("Belief distribution was empty or near-zero.")
    end

    # 2) Build the conditional distribution over y for that best_x
    x_idx = stateindex_x(p.momdp, best_x)
    sum_x_prob = 0.0
    for s in sup_b
        if s[1] == best_x
            sum_x_prob += pdf(b, s)
        end
    end
    # by[x_idx] = distribution over hidden states, given best_x
    by_vec = zeros(p.n_states_y)
    for s in sup_b
        if s[1] == best_x
            y_idx = stateindex_y(p.momdp, s[2])
            by_vec[y_idx] = pdf(b, s) / (sum_x_prob + eps())
        end
    end

    # 3) Among alpha-vectors for x_idx, pick the one with max dot product with by_vec
    best_dot = -Inf
    best_dot_idx = 1
    for i in 1:length(p.alphas[x_idx])
        val = dot(p.alphas[x_idx][i], by_vec)
        if val > best_dot
            best_dot = val
            best_dot_idx = i
        end
    end

    # 4) Return the action associated with that alpha vector
    return p.action_map[x_idx][best_dot_idx]
end
