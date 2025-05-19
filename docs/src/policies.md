# Policies

The `MOMDP` type currently has only been developed to work with alpha vector policies. If other policy types are desired, please open an issue on the [GitHub repository](https://github.com/JuliaPOMDP/MOMDPs.jl/issues). Until the issue has been resolved, you can convert your MOMDP to a POMDP using the `POMDP_of_Discrete_MOMDP` type (or implement your own conversion).

## `MOMDPAlphaVectorPolicy`

The [`MOMDPAlphaVectorPolicy`](@ref) type is similar to the `AlphaVectorPolicy` from POMDPTools.jl. The main difference is how the alpha vectors are stored. `AlphaVectorPolicy` stores alpha vectors as a vector of alpha vectors, i.e. `Vector{Vector{Float64}}`. 

A MOMDP value function $V(x, b_y)$ can be represented as a collection of vector sets $\Gamma_{y}(x) \mid x \in \mathcal{X}$. Therefore, in `MOMDPAlphaVectorPolicy`, we have a vector of alpha vectors for each visible state and the size of the alpha vector is $|\mathcal{Y}|$ (the number of hidden states). Therefore, we have a vector of vectors of alpha vectors, i.e. `Vector{Vector{Vector{Float64}}}`.

For the action map (`action_map`), we also have a vector of size $|\mathcal{X}|$ (the number of visible states) that contains a vector of actions associated with each alpha vector.

## Value Function

With an alpha vector policy represented as `MOMDPAlphaVectorPolicy` (a collection of alpha vector sets), can use our visible state to determine the appropriate set and then and then find the maximum alpha vector in the set
$$V(x, b_{\mathcal{Y}}) = \max_{\alpha \in \Gamma_{y(x)}} \{\alpha \cdot b_{\mathcal{Y}} \}$$
[`value(p::MOMDPAlphaVectorPolicy, b, x)`](@ref) is provided to evaluate a `MOMDPAlphaVectorPolicy` with a known visible state `x`.

However, to maintain compatibility with simulation tools already existing within the POMDPs.jl ecosystem we also provide the ability to execute a computed MOMDP policy as an `MOMDPAlphaVectorPolicy` assuming a POMDP modle (allowing for uncertainty over $\mathcal{X}$ as well). We first calculate $b_{\mathcal{X}}(x) = \sum_{y \in \mathcal{Y}} b(x,y)$ and then 
$$V^\prime(b) = \sum_{x \in \mathcal{X}} b_{\mathcal{X}}(x) V(x, b_{\mathcal{Y} \mid x})$$
where $b_{\mathcal{Y} \mid x} = b(x,y) / b_{\mathcal{X}}(x)$. This value calcualtion is provided by [`value(p::MOMDPAlphaVectorPolicy, b)`](@ref).

## Action Function
The action function [`action(p::MOMDPAlphaVectorPolicy, b)`](@ref) implements a heuristic instead of a true one step lookahead when there is uncertainty over $\mathcal{X}$. If executing the MOMDP policy as a POMDP and `x` is not known, then we recommend implementing a custom action function that performs a one-step lookahead using the `value` function. 

As implemented, `action` finds the state `x` with the largest probability mass in `b`, forms a conditional distribution over `y` given that `x`, finds the alpha vector within the subset associated with `x` that maximizes the value given the conditional distirbution, and returns the actions associated with that alpha vector.