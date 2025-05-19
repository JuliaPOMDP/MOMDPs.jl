# Examples
The process of defining, solving, and evaluating a MOMDP closely mirrors the steps for a POMDP, differing primarily in the function definitions required.

## RockSample
We will use the classic Rock Sample problem to demonstrate forming a MOMDP, solving it with SARSOP, and evaluating the policy. Since RockSample is already defined as a POMDP in `RockSample.jl`, we will reuse existing definitions where possible and focus on the MOMDP-specific aspects.


### RockSampleMOMDP Type
We define the MOMDP type similarly to the existing POMDP and provide a constructor from the POMDP type.

```@example running_example
using POMDPs
using POMDPTools
using MOMDPs
using Printf 
using LinearAlgebra
using RockSample
using StaticArrays # for SVector

mutable struct RockSampleMOMDP{K} <: MOMDP{RSPos,SVector{K,Bool},Int,Int}
    map_size::Tuple{Int,Int}
    rocks_positions::SVector{K,RSPos}
    init_pos::RSPos
    sensor_efficiency::Float64
    bad_rock_penalty::Float64
    good_rock_reward::Float64
    step_penalty::Float64
    sensor_use_penalty::Float64
    exit_reward::Float64
    terminal_state::RSPos
    discount_factor::Float64
end

"""
    RockSampleMOMDP(rocksample_pomdp::RockSamplePOMDP)

Create a RockSampleMOMDP using the same parameters in a RockSamplePOMDP.
"""
function RockSampleMOMDP(rocksample_pomdp::RockSamplePOMDP)
    return RockSampleMOMDP(
        rocksample_pomdp.map_size,
        rocksample_pomdp.rocks_positions,
        rocksample_pomdp.init_pos,
        rocksample_pomdp.sensor_efficiency,
        rocksample_pomdp.bad_rock_penalty,
        rocksample_pomdp.good_rock_reward,
        rocksample_pomdp.step_penalty,
        rocksample_pomdp.sensor_use_penalty,
        rocksample_pomdp.exit_reward,
        rocksample_pomdp.terminal_state.pos,
        rocksample_pomdp.discount_factor
    )
end
```

### State Space
In RockSample, the robot knows its location but observes only rock states. Thus, grid locations form the visible state and rock states form the hidden state.


```@example running_example
# Visible states: All possible grid locations and a terminal state
function MOMDPs.states_x(problem::RockSampleMOMDP)
    map_states = vec([SVector{2,Int}((i, j)) for i in 1:problem.map_size[1], j in 1:problem.map_size[2]])
    push!(map_states, problem.terminal_state) # Add terminal state
    return map_states
end

# Hidden states: All possible K-length vector of booleans, where K is the number of rocks
function MOMDPs.states_y(problem::RockSampleMOMDP{K}) where {K}
    bool_options = [[true, false] for _ in 1:K]
    vec_bool_options = vec(collect(Iterators.product(bool_options...)))
    s_vec_bool_options = [SVector{K,Bool}(bool_vec) for bool_vec in vec_bool_options]
    return s_vec_bool_options
end
```
#### State Indexing
For certain solvers, we also need the `stateindex` function defined. For MOMDPs, we need to define it for both the visible and hidden states.

```@example running_example
function MOMDPs.stateindex_x(problem::RockSampleMOMDP, s::Tuple{RSPos, SVector{K,Bool}}) where {K}
    return stateindex_x(problem, s[1])
end
function MOMDPs.stateindex_x(problem::RockSampleMOMDP, x::RSPos)
    if isterminal(problem, (x, first(states_y(problem))))
        return length(states_x(problem))
    end
    return LinearIndices(problem.map_size)[x[1], x[2]]
end

function MOMDPs.stateindex_y(problem::RockSampleMOMDP, s::Tuple{RSPos, SVector{K,Bool}}) where {K}
    return stateindex_y(problem, s[2])
end
function MOMDPs.stateindex_y(problem::RockSampleMOMDP, y::SVector{K,Bool}) where {K}
    return findfirst(==(y), states_y(problem))
end
```

#### Initial State Distributions
Similarly, we need to define the initial distribution over both the visible and hidden states. We can start with any distribution over the visible states, but in RockSample, we start at the `init_pos` defined in the problem.

```@example running_example
function MOMDPs.initialstate_x(problem::RockSampleMOMDP)
    return Deterministic(problem.init_pos)
end
```

The distribution over hidden states is conditioned on the visible state. Therefore, [`initialstate_y`](@ref) has `x` as an input argument.

```@example running_example
function MOMDPs.initialstate_y(::RockSampleMOMDP{K}, x::RSPos) where K
    probs = normalize!(ones(2^K), 1)
    states = Vector{SVector{K,Bool}}(undef, 2^K)
    for (i,rocks) in enumerate(Iterators.product(ntuple(x->[false, true], K)...))
        states[i] = SVector(rocks)
    end
    return SparseCat(states, probs)
end
```

Notice that we didn't use `x` in the function `initialstate_y`. In RockSample, the initial distribution over the rock states is independent of the robot position. Therefore, we can set [`is_initial_distribution_independent`](@ref) to `true`.

```@example running_example
MOMDPs.is_initial_distribution_independent(::RockSampleMOMDP) = true
```

!!! note
    If we plan on using the POMDPs.jl ecosystem, we still need to define [`initialstate(p::MOMDP)`](@ref). However, since our problem is discrete, we can use the `initialstate(p::MOMDP)` function defined in `discrete_momdp_functions.jl` using `initialstate_x` and `initialstate_y`.

### Action Space
There is no change in our action space from the POMDP version.

```@example running_example
POMDPs.actions(::RockSampleMOMDP{K}) where {K} = 1:RockSample.N_BASIC_ACTIONS+K
POMDPs.actionindex(::RockSampleMOMDP, a::Int) = a
```


### Transition Funtions
For the transition function, we need to define both [`transition_x`](@ref) and [`transition_y`](@ref). As a reminder, `transition_x` returns the distribution over the next visible state given the current state and action where the current state is defined as the tuple `(x,y)`.

For RockSample, it is a deterministic transition based on the action selected and the current visible state. We will use a similar helper function `next_position` as in the POMDP version.

```@example running_example
function next_position(s::RSPos, a::Int)
    if a > RockSample.N_BASIC_ACTIONS || a == 1
        # robot check rocks or samples
        return s
    elseif a <= RockSample.N_BASIC_ACTIONS
        # the robot moves 
        return s + RockSample.ACTION_DIRS[a]
    end
end

function MOMDPs.transition_x(problem::RockSampleMOMDP, s::Tuple{RSPos,SVector{K,Bool}}, a::Int) where {K}
    x = s[1]
    if isterminal(problem, s)
        return Deterministic(problem.terminal_state)
    end
    new_pos = next_position(x, a)
    if new_pos[1] > problem.map_size[1]
        new_pos = problem.terminal_state
    else
        new_pos = RSPos(clamp(new_pos[1], 1, problem.map_size[1]),
            clamp(new_pos[2], 1, problem.map_size[2]))
    end
    return Deterministic(new_pos)
end
```

As we stated before defining the function, `x_prime` is only dependent on `x` and the action. Therefore, we can set [`is_x_prime_dependent_on_y`](@ref) to `false`.

```@example running_example
MOMDPs.is_x_prime_dependent_on_y(::RockSampleMOMDP) = false
```

[`transition_y`](@ref) returns the distribution over the next hidden state given the current state, action, and next visible state. In RockSample, this transition is also deterministic.

```@example running_example
function MOMDPs.transition_y(problem::RockSampleMOMDP, s::Tuple{RSPos,SVector{K,Bool}}, a::Int, x_prime::RSPos) where {K}
    if isterminal(problem, s)
        return Deterministic(s[2])
    end

    if a == RockSample.BASIC_ACTIONS_DICT[:sample] && in(s[1], problem.rocks_positions)
        rock_ind = findfirst(isequal(s[1]), problem.rocks_positions)
        new_rocks = MVector{K,Bool}(undef)
        for r = 1:K
            new_rocks[r] = r == rock_ind ? false : s[2][r]
        end
        new_rocks = SVector(new_rocks)
        
    else # We didn't sample, so states of rocks remain unchanged
        new_rocks = s[2]
    end
    
    return Deterministic(new_rocks)
end
```

Norice in our `transition_y` for RockSample that we did not use `x_prime` in the function. Therefore, we know the distritbuion of `y_prime` is conditionally independent of `x_prime` given the current state and action. Thus we can set [`is_y_prime_dependent_on_x_prime`](@ref) to `false`.

```@example running_example
MOMDPs.is_y_prime_dependent_on_x_prime(::RockSampleMOMDP) = false
```

### Observation Space
In RockSample, we started with a known initial position of our robot and then the robot transitions in the grid are deterministic. Therefore, the location is always known through belief updates without needing an observation. The only observations needed in the POMDP version and in the MOMDP version are the results on the sensor.

```@example running_example
POMDPs.observations(::RockSampleMOMDP) = 1:3
POMDPs.obsindex(::RockSampleMOMDP, o::Int) = o

function POMDPs.observation(problem::RockSampleMOMDP, a::Int, s::Tuple{RSPos,SVector{K,Bool}}) where {K}
    if a <= RockSample.N_BASIC_ACTIONS
        # no obs
        return SparseCat((1, 2, 3), (0.0, 0.0, 1.0))
    else
        rock_ind = a - RockSample.N_BASIC_ACTIONS
        rock_pos = problem.rocks_positions[rock_ind]
        dist = norm(rock_pos - s[1])
        efficiency = 0.5 * (1.0 + exp(-dist * log(2) / problem.sensor_efficiency))
        rock_state = s[2][rock_ind]
        if rock_state
            return SparseCat((1, 2, 3), (efficiency, 1.0 - efficiency, 0.0))
        else
            return SparseCat((1, 2, 3), (1.0 - efficiency, efficiency, 0.0))
        end
    end
end
```

If we wanted to start from any position (and thus the initial distribution would be uniform over the grid), then for the POMDP version we would need to increase our observation space to include the position of the robot. Therefore our observation space size would increase by a factor of $|\mathcal{X}| - 1$ (since we have a terminal state within $\mathcal{X}$). However, for a MOMDP, we do not need those observations, and our observation space would remain as defined.

### Other Functions
There are no other changes to defining a MOMDP vs the POMDP using the explicit interface. However, we still need to define the reward function, terminal function, and discount factor.

```@example running_example
function POMDPs.reward(problem::RockSampleMOMDP, s::Tuple{RSPos,SVector{K,Bool}}, a::Int) where {K}
    r = problem.step_penalty
    if next_position(s[1], a)[1] > problem.map_size[1]
        r += problem.exit_reward
        return r
    end

    if a == RockSample.BASIC_ACTIONS_DICT[:sample] && in(s[1], problem.rocks_positions) # sample 
        rock_ind = findfirst(isequal(s[1]), problem.rocks_positions) # slow ?
        r += s[2][rock_ind] ? problem.good_rock_reward : problem.bad_rock_penalty
    elseif a > RockSample.N_BASIC_ACTIONS # using senssor
        r += problem.sensor_use_penalty
    end
    return r
end

function POMDPs.isterminal(problem::RockSampleMOMDP, s::Tuple{RSPos,SVector{K,Bool}}) where {K}
    return s[1] == problem.terminal_state
end

POMDPs.discount(problem::RockSampleMOMDP) = problem.discount_factor
```

## Solving using SARSOP
Now that we have defined our MOMDP, we can solve it with SARSOP. We will create a POMDP RockSample problem and then a MOMDP RockSample problem from the POMDP since our constructor was defined with the POMDP type. Since we have the POMDP, we will also solve the POMDP using SARSOP so we can compare the policies.

!!! note
    SARSOP.jl and POMDPXFiles.jl have not been updated to work wtih MOMDPs.jl. We must include `test/pomdpxfiles.jl` and `test/sarsop.jl` until the packages are updated. This note will be removed and the examples will be updated when the packages are updated.
    
```@example running_example
using SARSOP
using POMDPXFiles
using ProgressMeter

include("../../test/sarsop.jl")
include("../../test/pomdpxfiles.jl")

```

```@example running_example
# Create a smaller RockSample problem
rocksample_pomdp = RockSample.RockSamplePOMDP(
    map_size=(3, 3),
    rocks_positions=[(2, 2), (3, 3), (1, 2)],
    init_pos=(1, 1),
    sensor_efficiency=0.5
)
rocksample_momdp = RockSampleMOMDP(rocksample_pomdp)

# Instantiate the solver
solver_pomdp = SARSOPSolver(; precision=1e-2, timeout=30, 
    pomdp_filename="test_rocksample_pomdp.pomdpx", verbose=false)
solver_momdp = SARSOPSolver(; precision=1e-2, timeout=30, 
    pomdp_filename="test_rocksample_momdp.pomdpx", verbose=false)
  
# Solve the POMDP and the MOMDP
policy_pomdp = solve(solver_pomdp, rocksample_pomdp)
policy_momdp = solve(solver_momdp, rocksample_momdp)

# Evaluate the policies at the initial belief
b0_pomdp = initialstate(rocksample_pomdp)
b0_momdp = initialstate(rocksample_momdp)

val_pomdp_b0 = value(policy_pomdp, b0_pomdp)
val_momdp_b0 = value(policy_momdp, b0_momdp)

@printf("Value of POMDP policy: %.4f\n", val_pomdp_b0)
@printf("Value of MOMDP policy: %.4f\n", val_momdp_b0)

# What is the action of the policies at the initial belief?
a_pomdp_b0 = action(policy_pomdp, b0_pomdp)
a_momdp_b0 = action(policy_momdp, b0_momdp)

@printf("Action of POMDP policy: %d\n", a_pomdp_b0)
@printf("Action of MOMDP policy: %d\n", a_momdp_b0)
```

## Converting to a POMDP
If you have a problem defined as a MOMDP, you can convert it to an equivalent POMDP. If your problem is discrete, you can use the [`POMDP_of_Discrete_MOMDP`](@ref) type. 

```@example running_example
rocksample_pomdp_from_momdp = POMDP_of_Discrete_MOMDP(rocksample_momdp)

solver_pomdp_from_momdp = SARSOPSolver(; precision=1e-2, timeout=30, 
    pomdp_filename="test_rocksample_momdp.pomdpx", verbose=false)

policy_pomdp_from_momdp = solve(solver_pomdp_from_momdp, rocksample_pomdp_from_momdp)

b0_pomdp_from_momdp = initialstate(rocksample_pomdp_from_momdp)
val_pomdp_from_momdp_b0 = value(policy_pomdp_from_momdp, b0_pomdp_from_momdp)

@printf("Value of POMDP generated from MOMDP: %.4f\n", val_pomdp_from_momdp_b0)
```