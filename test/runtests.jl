using POMDPs
using POMDPTools
using MOMDPs

using Random
using RockSample
using LinearAlgebra
using StaticArrays
using SARSOP
using POMDPXFiles

using Test
using Aqua

#! These will be removed when SARSOP and POMDPXFiles are updated
include("sarsop.jl")
using ProgressMeter
include("pomdpxfiles.jl")

include("rocksample_momdp.jl")

# Default to running full tests unless explicitly disabled
const SKIP_LONG_TESTS = get(ENV, "SKIP_LONG_TESTS", "false") == "true"


# Create a simple RockSample problem for testing
function create_test_rocksample(problem_type::Symbol=:five_by_five; map_size=3, n_rocks=3)
    if problem_type == :five_by_five
        # Create a small RockSample problem for quick testing
        rocksample_pomdp = RockSample.RockSamplePOMDP(
            map_size=(5, 5),
            rocks_positions=[(2, 2), (1, 3), (3, 1)],
            init_pos=(1, 1),
            sensor_efficiency=1.0
        )
    elseif problem_type == :three_by_three
        # Create a smaller RockSample problem for quick testing
        rocksample_pomdp = RockSample.RockSamplePOMDP(
            map_size=(3, 3),
            rocks_positions=[(2, 2), (3, 3), (1, 2)],
            init_pos=(1, 1),
            sensor_efficiency=0.5
        )
    elseif problem_type == :random
        # Create a random RockSample problem for testing
        rocksample_pomdp = RockSample.RockSamplePOMDP(map_size, n_rocks)
    else
        error("Unsupported problem size: $problem_size")
    end
    
    rocksample_momdp = RockSampleMOMDP(rocksample_pomdp)
    pomdp_of_momdp = POMDP_of_Discrete_MOMDP(rocksample_momdp)
    
    return rocksample_pomdp, rocksample_momdp, pomdp_of_momdp
end

@testset "MOMDPs.jl" begin

    @testset "Types" begin
        mutable struct X <: MOMDP{Float64, Int, Int, Bool} end
        abstract type Z <: MOMDP{Float64, Int, Int, Bool} end
        mutable struct Y <: Z end
        mutable struct W <: POMDP{Float64, Int, Bool} end
    
        @test_throws ErrorException statetype(Int)
        @test_throws ErrorException actiontype(Int)
        @test_throws ErrorException obstype(Int)
    
        @test statetype(X) == Tuple{Float64, Int}
        @test statetype(Y) == Tuple{Float64, Int}
        @test actiontype(X) == Int
        @test actiontype(Y) == Int
        @test obstype(X) == Bool
        @test obstype(Y) == Bool
        
        @test statetype_x(X) == Float64
        @test statetype_x(Y) == Float64
        @test statetype_y(X) == Int
        @test statetype_y(Y) == Int
        
        x = X()
        y = Y()
        @test statetype_x(x) == Float64
        @test statetype_y(x) == Int
        @test statetype_x(y) == Float64
        @test statetype_y(y) == Int
        
        @test_throws ErrorException statetype_x(W)
        @test_throws ErrorException statetype_y(W)
    end
    
    @testset "Interface Tests" begin
        # Use RockSampleMOMDP to test interface implementation
        _, momdp, _ = create_test_rocksample()
        
        @test isa(momdp, MOMDP)
        
        # Test state space accessors
        @test !isempty(states_x(momdp))
        @test !isempty(states_y(momdp))
        @test length(ordered_states_x(momdp)) == length(states_x(momdp))
        @test length(ordered_states_y(momdp)) == length(states_y(momdp))
        
        # Test state indexing
        @test stateindex_x(momdp, SVector{2,Int}((1, 1))) == 1
        @test stateindex_x(momdp, SVector{2,Int}((5, 5))) == 25
        @test stateindex_y(momdp, SVector{3,Bool}([true, true, true])) == 1
        @test stateindex_y(momdp, SVector{3,Bool}([false, false, false])) == 8
        
        # Test initial state distributions
        @test isa(initialstate_x(momdp), Deterministic)
        @test isa(initialstate_y(momdp, first(states_x(momdp))), SparseCat)
        @test isa(initialstate(momdp), SparseCat)
        
        # Test transition functions
        s = (SVector{2,Int}((1, 1)), SVector{3,Bool}([true, true, true]))
        a = 2 # Move east
        next_x = SVector{2,Int}((2, 1))
        
        @test isa(transition_x(momdp, s, a), Deterministic)
        @test isa(transition_y(momdp, s, a, next_x), Deterministic)
        
        # Test POMDPs.jl interface
        @test isa(actions(momdp), AbstractVector)
        @test isa(discount(momdp), Number)
        @test isa(observations(momdp), AbstractVector)
        
        # Test default conditional independence assumptions
        mutable struct X <: MOMDP{Float64, Int, Int, Bool} end
        x = X()
        @test is_y_prime_dependent_on_x_prime(x) == true
        @test is_x_prime_dependent_on_y(x) == true
        @test is_initial_distribution_independent(x) == true
    end

    @testset "RockSample MOMDP Implementation" begin
        _, momdp, _ = create_test_rocksample()
        
        # Test MOMDP specific functions
        @test !is_y_prime_dependent_on_x_prime(momdp)
        @test !is_x_prime_dependent_on_y(momdp)
        @test is_initial_distribution_independent(momdp)
        
        # Test specific transitions
        s = (SVector{2,Int}((1, 1)), SVector{3,Bool}([true, true, true]))
        a_east = 3 # Move east
        
        # Test moving east from (1,1)
        next_x_dist = transition_x(momdp, s, a_east)
        @test pdf(next_x_dist, SVector{2,Int}((2, 1))) ≈ 1.0
        
        # Test rock sampling
        a_sample = 1
        rock_pos = SVector{2,Int}((1, 3))
        s_at_rock = (rock_pos, SVector{3,Bool}([true, true, true]))
        next_y_dist = transition_y(momdp, s_at_rock, a_sample, rock_pos)
        
        # After sampling at rock position (1,3), the second rock should be false
        @test pdf(next_y_dist, SVector{3,Bool}([true, false, true])) ≈ 1.0
        
        # Test observations
        # Check no observation when moving
        obs_dist = observation(momdp, a_east, s)
        @test pdf(obs_dist, 3) ≈ 1.0
        
        # Test sensor
        a_sense_rock1 = 6 # Sense first rock
        sense_dist = observation(momdp, a_sense_rock1, s)
        @test pdf(sense_dist, 1) > 0.0  # Good detection (some probability)
        @test pdf(sense_dist, 2) > 0.0  # Bad detection (some probability)
        @test pdf(sense_dist, 3) ≈ 0.0  # No observation
        
        # Test rewards
        r_move = reward(momdp, s, a_east)
        @test r_move == momdp.step_penalty  # Should be negative due to step penalty
        
        # Sample at rock position
        r_sample_good = reward(momdp, s_at_rock, a_sample)
        @test r_sample_good == momdp.good_rock_reward  # Should be positive for good rock
        
        # Sample at non-rock position
        r_sample_none = reward(momdp, s, a_sample)
        @test r_sample_none == 0  # Just step penalty, no sampling effect
    end

    @testset "MOMDP to POMDP Conversion" begin
        _, momdp, pomdp_of_momdp = create_test_rocksample()
        
        # Test that state spaces are the same
        @test length(states(pomdp_of_momdp)) == length(states_x(momdp)) * length(states_y(momdp))
        
        # Test that actions are the same
        @test actions(pomdp_of_momdp) == actions(momdp)
        
        # Test transitions
        s = (SVector{2,Int}((1, 1)), SVector{3,Bool}([true, true, true]))
        a = 3 # Move east

        # Get transitions from both
        trans_momdp_x = transition_x(momdp, s, a)
        trans_momdp_y = transition_y(momdp, s, a, SVector{2,Int}((2, 1)))
        trans_pomdp = transition(pomdp_of_momdp, s, a)
        
        # Check that the resulting distributions assign same probabilities
        for sp in support(trans_pomdp)
            p_x = pdf(trans_momdp_x, sp[1])
            p_y = pdf(trans_momdp_y, sp[2])
            p_joint = pdf(trans_pomdp, sp)
            
            @test p_joint ≈ p_x * p_y
        end
        
        sp = rand(trans_pomdp)
        @test sp == (SVector{2,Int}((2, 1)), SVector{3,Bool}([true, true, true]))
        
        # Test observations
        # The observations in pomdp_of_momdp should be tuples of (x, o),
        # where o comes from the observation model of the MOMDP
        obs_momdp = observation(momdp, a, sp)
        obs_pomdp = observation(pomdp_of_momdp, a, sp)
        
        for o_pomdp in support(obs_pomdp)
            # Extract MOMDP observation from tuple
            o_momdp_x = o_pomdp[1]
            o_momdp_y = o_pomdp[2]
            
            # Since x is fully observable, x should match next_x
            next_x = SVector{2,Int}((2, 1))  # After moving east
            @test o_pomdp[1] == next_x
            
            # Check probability matches
            @test pdf(obs_pomdp, o_pomdp) ≈ pdf(obs_momdp, o_momdp_y)
        end
        
        # Test obsindex function for pomdp_of_momdp
        s = (SVector{2,Int}((1, 1)), SVector{3,Bool}([true, true, true]))
        o = 1
        @test obsindex(pomdp_of_momdp, (s, o)) == 1
        
        s = (SVector{2,Int}((1, 1)), SVector{3,Bool}([true, true, true]))
        o = 2
        @test obsindex(pomdp_of_momdp, (s, o)) == 1 + length(states_x(pomdp_of_momdp.momdp))
        
        s = (SVector{2,Int}((2, 1)), SVector{3,Bool}([true, true, true]))
        o = 2
        @test obsindex(pomdp_of_momdp, (s, o)) == 2 + length(states_x(pomdp_of_momdp.momdp))
    end

    @testset "SARSOP Policy Value Comparisons" begin
        rocksample_pomdp, rocksample_momdp, pomdp_of_momdp = create_test_rocksample(:three_by_three)
            
        # Solve both problems with SARSOP
        # Use short timeout for testing
        solver_pomdp = SARSOPSolver(; precision=1e-2, timeout=30, 
            pomdp_filename="test_rocksample_pomdp.pomdpx", verbose=false)
        solver_momdp = SARSOPSolver(; precision=1e-2, timeout=30, 
            pomdp_filename="test_rocksample_momdp.pomdpx", verbose=false)
        solver_pomdp_of_momdp = SARSOPSolver(; precision=1e-2, timeout=30, 
            pomdp_filename="test_rocksample_pomdp_of_momdp.pomdpx", verbose=false)
        
        policy_pomdp = solve(solver_pomdp, rocksample_pomdp)
        policy_momdp = solve(solver_momdp, rocksample_momdp)
        policy_pomdp_of_momdp = solve(solver_pomdp_of_momdp, pomdp_of_momdp)
        
        # Get initial beliefs
        b0_pomdp = initialstate(rocksample_pomdp)
        b0_momdp = initialstate(rocksample_momdp)
        b0_pomdp_of_momdp = initialstate(pomdp_of_momdp)
        
        # Test value of initial beliefs are the same
        val_pomdp_b0 = value(policy_pomdp, b0_pomdp)
        val_momdp_b0 = value(policy_momdp, b0_momdp)
        val_pomdp_of_momdp_b0 = value(policy_pomdp_of_momdp, b0_pomdp_of_momdp)
        @test isapprox(val_pomdp_b0, val_momdp_b0, atol=1e-5)
        @test isapprox(val_pomdp_b0, val_pomdp_of_momdp_b0, atol=1e-5)
        
        # Create other beliefs and test the values are the same
        
        # Create a few random problems and test the values are the same
        for _ in 1:10
            rocksample_pomdp, rocksample_momdp, pomdp_of_momdp = create_test_rocksample(:random)
            policy_pomdp = solve(solver_pomdp, rocksample_pomdp)
            policy_momdp = solve(solver_momdp, rocksample_momdp)
            policy_pomdp_of_momdp = solve(solver_pomdp_of_momdp, pomdp_of_momdp)
            
            # Get initial beliefs
            b0_pomdp = initialstate(rocksample_pomdp)
            b0_momdp = initialstate(rocksample_momdp)
            b0_pomdp_of_momdp = initialstate(pomdp_of_momdp)
            
            val_pomdp_b0 = value(policy_pomdp, b0_pomdp)
            val_momdp_b0 = value(policy_momdp, b0_momdp)
            val_pomdp_of_momdp_b0 = value(policy_pomdp_of_momdp, b0_pomdp_of_momdp)
            @test isapprox(val_pomdp_b0, val_momdp_b0, atol=1e-4)
            @test isapprox(val_pomdp_b0, val_pomdp_of_momdp_b0, atol=1e-4)
        end
    
        # Cleanup files
        isfile("test_rocksample_pomdp.pomdpx") && rm("test_rocksample_pomdp.pomdpx")
        isfile("test_rocksample_momdp.pomdpx") && rm("test_rocksample_momdp.pomdpx")
        isfile("test_rocksample_pomdp_of_momdp.pomdpx") && rm("test_rocksample_pomdp_of_momdp.pomdpx")
        isfile("policy.out") && rm("policy.out")
    end
    
    @testset "Policy Comparison Tests" begin
        # Skip long tests if requested
        if !SKIP_LONG_TESTS
            # Run for various seeds and a certain number of steps
            seeds = 1:100
            steps = 1:10
        
            rocksample_pomdp, rocksample_momdp, pomdp_of_momdp = create_test_rocksample(:three_by_three)
            
            # Solve both problems with SARSOP
            # Use short timeout for testing
            solver_pomdp = SARSOPSolver(; precision=1e-2, timeout=30, 
                pomdp_filename="test_rocksample_pomdp.pomdpx", verbose=false)
            solver_momdp = SARSOPSolver(; precision=1e-2, timeout=30, 
                pomdp_filename="test_rocksample_momdp.pomdpx", verbose=false)
            
            policy_pomdp = solve(solver_pomdp, rocksample_pomdp)
            policy_momdp = solve(solver_momdp, rocksample_momdp)
            
            # Get initial beliefs
            b0_pomdp = initialstate(rocksample_pomdp)
            b0_momdp = initialstate(rocksample_momdp)
            
            # Test value of initial beliefs are the same
            val_pomdp_b0 = value(policy_pomdp, b0_pomdp)
            val_momdp_b0 = value(policy_momdp, b0_momdp)
            @test isapprox(val_pomdp_b0, val_momdp_b0, atol=1e-2)
            
            # Test actions are the same for initial beliefs
            @test action(policy_pomdp, b0_pomdp) == action(policy_momdp, b0_momdp)
            
            rewards_pomdp = zeros(length(seeds))
            rewards_momdp = zeros(length(seeds))
            
            for seed_i in seeds
                # Set seed for reproducibility
                rng_pomdp = MersenneTwister(seed_i)
                
                # Sample initial states
                s_pomdp = rand(rng_pomdp, initialstate(rocksample_pomdp))
                s_momdp = (s_pomdp.pos, s_pomdp.rocks)
                
                # Create updaters
                up_pomdp = updater(policy_pomdp)
                up_momdp = updater(policy_momdp)
                
                b_pomdp = deepcopy(b0_pomdp)
                b_momdp = deepcopy(b0_momdp)
                
                # Run a few steps and check actions match
                r_pomdp = 0.0
                r_momdp = 0.0
                γ = discount(rocksample_pomdp)
                for i in steps
                    # Test if values at beliefs are the same
                    v_pomdp = value(policy_pomdp, b_pomdp)
                    v_momdp = value(policy_momdp, b_momdp)
                    @test isapprox(v_pomdp, v_momdp, atol=1e-5)
                    
                    a_pomdp = action(policy_pomdp, b_pomdp)
                    a_momdp = action(policy_momdp, b_momdp)
                    
                    r_pomdp += γ^(i-1) * reward(rocksample_pomdp, s_pomdp, a_pomdp)
                    r_momdp += γ^(i-1) * reward(rocksample_momdp, s_momdp, a_momdp)
                    
                    # Transition to next states
                    sp_pomdp = rand(rng_pomdp, transition(rocksample_pomdp, s_pomdp, a_pomdp))
                    sp_momdp = rand(rng_pomdp, transition(rocksample_momdp, s_momdp, a_momdp))
                    
                    # Observations
                    o_pomdp = rand(rng_pomdp, observation(rocksample_pomdp, a_pomdp, s_pomdp))
                    o_momdp = rand(rng_pomdp, observation(rocksample_momdp, a_momdp, s_momdp))
                    
                    if POMDPs.isterminal(rocksample_pomdp, s_pomdp)
                        break
                    end
                    
                    # Update beliefs
                    b_pomdp = update(up_pomdp, b_pomdp, a_pomdp, o_pomdp)
                    b_momdp = update(up_momdp, b_momdp, a_momdp, o_momdp)
                    
                    # Update states for next iteration
                    s_pomdp = sp_pomdp
                    s_momdp = sp_momdp
                end
                rewards_pomdp[seed_i] = r_pomdp
                rewards_momdp[seed_i] = r_momdp
            end
            
            ave_rewards_pomdp = mean(rewards_pomdp)
            ave_rewards_momdp = mean(rewards_momdp)
            
            # Default sensor value is large, so rewards should be the same
            @test isapprox(ave_rewards_pomdp, ave_rewards_momdp, atol=1e-3)
            
            # Clean up files
            isfile("test_rocksample_pomdp.pomdpx") && rm("test_rocksample_pomdp.pomdpx")
            isfile("test_rocksample_momdp.pomdpx") && rm("test_rocksample_momdp.pomdpx")
            isfile("policy.out") && rm("policy.out")
        else
            @info "Skipping policy comparison tests (set ENV[\"SKIP_LONG_TESTS\"] = false to run)"
            @test true
        end
    end

    @testset "Alpha Vector Policy Tests" begin
        _, momdp, _ = create_test_rocksample()
        
        # Create a simple alpha vector policy manually for testing
        n_x = length(states_x(momdp))
        n_y = length(states_y(momdp))
        n_actions = length(actions(momdp))
        
        # Create simple alphas - one alpha vector per visible state
        alphas = Vector{Vector{Vector{Float64}}}(undef, n_x)
        action_map = Vector{Vector{Int}}(undef, n_x)
        
        for x_idx in 1:n_x
            # For simplicity, create 1 alpha vector per visible state
            alphas[x_idx] = [zeros(n_y)]
            action_map[x_idx] = [1]  # Always choose action 1
            
            # Simple values - prefer first action, values decrease for later hidden states
            for y_idx in 1:n_y
                alphas[x_idx][1][y_idx] = n_y - y_idx + 1
            end
        end
        
        # Create policy
        policy = MOMDPAlphaVectorPolicy(momdp, n_x, n_y, alphas, action_map)
        
        # Test action selection on the policy
        s_x = first(states_x(momdp))
        
        # Create belief with perfect knowledge of hidden state
        b_y = zeros(n_y)
        b_y[1] = 1.0  # Certainty about first hidden state
        
        # Create mixed-observable belief
        beliefs = Dict{Tuple{typeof(s_x), SVector{3,Bool}}, Float64}()
        for (y_idx, y) in enumerate(ordered_states_y(momdp))
            beliefs[(s_x, y)] = b_y[y_idx]
        end
        b = SparseCat(collect(keys(beliefs)), collect(values(beliefs)))
        
        # Test action selection
        a = action(policy, b)
        @test a == 1
        
        # Test value function
        v = value(policy, b)
        @test v > 0
        
        # Test action selection based on known visible state
        b = SparseCat([first(states_y(momdp))], [1.0])
        a = action(policy, b, s_x)
        @test a == 1
    end

    @testset "MOMDPDiscreteUpdater Tests" begin
        _, momdp, _ = create_test_rocksample()
        
        # Test updater creation
        @testset "Updater Creation" begin
            updater = MOMDPDiscreteUpdater(momdp)
            @test updater.momdp === momdp
            @test isa(updater, MOMDPDiscreteUpdater)
        end
        
        # Test uniform belief creation
        @testset "Uniform Belief Creation" begin
            updater = MOMDPDiscreteUpdater(momdp)
            
            # Test with updater
            b_uniform = uniform_belief_y(updater)
            @test isa(b_uniform, DiscreteBelief)
            @test length(b_uniform.b) == length(states_y(momdp))
            @test all(b_uniform.b .≈ 1.0 / length(states_y(momdp)))
            @test sum(b_uniform.b) ≈ 1.0
            
            # Test with momdp directly
            b_uniform2 = uniform_belief_y(momdp)
            @test b_uniform.b ≈ b_uniform2.b
        end
        
        # Test belief initialization
        @testset "Belief Initialization" begin
            updater = MOMDPDiscreteUpdater(momdp)
            
            # Test with SparseCat distribution
            states_y_list = ordered_states_y(momdp)
            sparse_dist = SparseCat([states_y_list[1], states_y_list[3]], [0.7, 0.3])
            b_sparse = initialize_belief(updater, sparse_dist)
            
            @test isa(b_sparse, DiscreteBelief)
            @test sum(b_sparse.b) ≈ 1.0
            @test b_sparse.b[1] ≈ 0.7  # First hidden state
            @test b_sparse.b[2] ≈ 0.0  # Second hidden state (not in sparse support)
            @test b_sparse.b[3] ≈ 0.3  # Third hidden state
            
            # Test with uniform distribution
            uniform_dist = uniform_belief_y(momdp)
            b_from_uniform = initialize_belief(updater, uniform_dist)
            @test b_from_uniform.b ≈ uniform_dist.b
        end
        
        # Test belief updates
        @testset "Belief Updates" begin
            updater = MOMDPDiscreteUpdater(momdp)
            
            # Create initial belief
            b_initial = uniform_belief_y(updater)
            
            # Test parameters for update
            x = SVector{2,Int}((1, 1))  # Initial visible state
            xp = SVector{2,Int}((2, 1)) # Next visible state (moved east)
            a = 3  # Move east action
            o = 3  # No observation (movement)
            
            # Perform update
            b_updated = update(updater, b_initial, a, o, x, xp)
            
            @test isa(b_updated, DiscreteBelief)
            @test sum(b_updated.b) ≈ 1.0
            @test length(b_updated.b) == length(b_initial.b)
            
            # Test that belief changes appropriately
            # For movement without sampling, hidden states should remain unchanged
            # (since rocks don't change when moving)
            @test b_updated.b ≈ b_initial.b
            
            # Test update with rock sampling
            x_at_rock = SVector{2,Int}((1, 3))  # Position of second rock
            a_sample = 1  # Sample action
            o_sample = 3  # No observation from sampling
            
            # Create belief with some structure
            b_temp = zeros(length(states_y(momdp)))
            b_temp[1] = 0.1
            b_temp[2] = 0.6
            b_temp[3] = 0.3
            b_structured = DiscreteBelief(momdp, ordered_states_y(momdp), b_temp)
            b_after_sample = update(updater, b_structured, a_sample, o_sample, x_at_rock, x_at_rock)
            
            @test isa(b_after_sample, DiscreteBelief)
            @test sum(b_after_sample.b) ≈ 1.0
            # After sampling at rock 2, the belief should reflect that rock 2 is now bad
        end
        
        # Test update with observation
        @testset "Update with Sensor Observations" begin
            updater = MOMDPDiscreteUpdater(momdp)
            
            # Create initial belief
            b_initial = uniform_belief_y(updater)
            
            # Test sensing first rock
            x = SVector{2,Int}((1, 1))  # Initial position
            a_sense = 6  # Sense first rock action
            o_good = 1   # Good rock observation
            o_bad = 2    # Bad rock observation
            
            # Update with good observation
            b_good = update(updater, b_initial, a_sense, o_good, x, x)
            @test isa(b_good, DiscreteBelief)
            @test sum(b_good.b) ≈ 1.0
            
            # Update with bad observation
            b_bad = update(updater, b_initial, a_sense, o_bad, x, x)
            @test isa(b_bad, DiscreteBelief)
            @test sum(b_bad.b) ≈ 1.0
            
            # The beliefs should be different
            @test !(b_good.b ≈ b_bad.b)
        end
        
        # Test convenience update method
        @testset "Convenience Update Method" begin
            updater = MOMDPDiscreteUpdater(momdp)
            
            # Create a SparseCat belief
            states_y_list = collect(states_y(momdp))
            sparse_belief = SparseCat([states_y_list[1]], [1.0])
            
            x = SVector{2,Int}((1, 1))
            xp = SVector{2,Int}((2, 1))
            a = 3  # Move east
            o = 3  # No observation
            
            # This should work with the convenience method
            b_updated = update(updater, sparse_belief, a, o, x, xp)
            @test isa(b_updated, DiscreteBelief)
            @test sum(b_updated.b) ≈ 1.0
            
            # Should be equivalent to explicit conversion
            b_converted = initialize_belief(updater, sparse_belief)
            b_updated_explicit = update(updater, b_converted, a, o, x, xp)
            @test b_updated.b ≈ b_updated_explicit.b
        end
        
        # Test error conditions
        @testset "Error Conditions" begin
            updater = MOMDPDiscreteUpdater(momdp)
            
            # Create a belief that will lead to impossible observation
            states_y_list = collect(states_y(momdp))
            # Create a deterministic belief about hidden state
            impossible_belief = DiscreteBelief(momdp, states_y_list, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            x = SVector{2,Int}((1, 1))
            a = 6  # Sense first rock
            o_impossible = 1  # This might be impossible given certain rock configurations
            
            # Note: This test might pass if the observation is actually possible
            # The exact test depends on the specific MOMDP dynamics
            # We'll test that the function doesn't crash rather than testing specific error
            try
                result = update(updater, impossible_belief, a, o_impossible, x, x)
                @test isa(result, DiscreteBelief)
            catch e
                # If it throws an error, it should be the specific error we expect
                @test occursin("Failed discrete belief update", string(e))
            end
        end
        
        # Test belief consistency properties
        @testset "Belief Consistency" begin
            updater = MOMDPDiscreteUpdater(momdp)
            b_initial = uniform_belief_y(updater)
            
            # Test multiple update steps maintain probability mass
            x = SVector{2,Int}((1, 1))
            b_current = b_initial
            
            for step in 1:5
                xp = SVector{2,Int}((min(step + 1, 5), 1))  # Move east but stay in bounds
                a = 3  # Move east
                o = 3  # No observation
                
                b_current = update(updater, b_current, a, o, x, xp)
                
                @test isa(b_current, DiscreteBelief)
                @test sum(b_current.b) ≈ 1.0
                @test all(b_current.b .≥ 0.0)  # All probabilities non-negative
                
                x = xp  # Update for next iteration
            end
        end
    end
    
    @testset "beliefvec_y Tests" begin
        _, momdp, _ = create_test_rocksample()
        n_y = length(states_y(momdp))
        states_y_list = ordered_states_y(momdp)
        
        @testset "SparseCat Conversion" begin
            # Test with sparse categorical distribution
            sparse_states = [states_y_list[1], states_y_list[3]]
            sparse_probs = [0.7, 0.3]
            sparse_belief = SparseCat(sparse_states, sparse_probs)
            
            b_vec = MOMDPs.beliefvec_y(momdp, n_y, sparse_belief)
            
            @test length(b_vec) == n_y
            @test b_vec[1] ≈ 0.7  # First state
            @test b_vec[2] ≈ 0.0  # Not in sparse support
            @test b_vec[3] ≈ 0.3  # Third state
            @test sum(b_vec) ≈ 1.0
            @test all(b_vec .≥ 0.0)
        end
        
        @testset "AbstractVector Conversion" begin
            # Test with vector input
            input_vec = [0.2, 0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
            @test length(input_vec) == n_y  # Ensure correct length
            
            b_vec = MOMDPs.beliefvec_y(momdp, n_y, input_vec)
            
            @test b_vec === input_vec  # Should return same vector
            @test length(b_vec) == n_y
            
            # Test assertion for wrong length
            wrong_length_vec = [0.5, 0.5]
            @test_throws AssertionError MOMDPs.beliefvec_y(momdp, n_y, wrong_length_vec)
        end
        
        @testset "DiscreteBelief Conversion" begin
            # Test with DiscreteBelief
            prob_vec = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]
            discrete_belief = DiscreteBelief(momdp, states_y_list, prob_vec)
            
            b_vec = MOMDPs.beliefvec_y(momdp, n_y, discrete_belief)
            
            @test b_vec === discrete_belief.b  # Should return underlying vector
            @test length(b_vec) == n_y
            @test b_vec ≈ prob_vec
        end
        
        @testset "Deterministic Conversion" begin
            # Test with deterministic belief
            det_state = states_y_list[2]  # Second hidden state
            det_belief = Deterministic(det_state)
            
            b_vec = MOMDPs.beliefvec_y(momdp, n_y, det_belief)
            
            @test length(b_vec) == n_y
            @test sum(b_vec) ≈ 1.0
            @test count(x -> x > 0, b_vec) == 1  # Only one non-zero element
            @test b_vec[2] ≈ 1.0  # Second state should have probability 1
            @test all(b_vec[i] ≈ 0.0 for i in [1, 3, 4, 5, 6, 7, 8])  # Others zero
        end
        
        @testset "Edge Cases" begin
            # Test with uniform sparse belief
            uniform_sparse = SparseCat(states_y_list, fill(1.0/n_y, n_y))
            b_vec_uniform = MOMDPs.beliefvec_y(momdp, n_y, uniform_sparse)
            
            @test length(b_vec_uniform) == n_y
            @test all(b_vec_uniform .≈ 1.0/n_y)
            @test sum(b_vec_uniform) ≈ 1.0
            
            # Test with single state sparse belief
            single_sparse = SparseCat([states_y_list[1]], [1.0])
            b_vec_single = MOMDPs.beliefvec_y(momdp, n_y, single_sparse)
            
            @test b_vec_single[1] ≈ 1.0
            @test all(b_vec_single[2:end] .≈ 0.0)
        end
    end
    
    @testset "actionvalues Tests" begin
        _, momdp, _ = create_test_rocksample()
        
        # Create a simple alpha vector policy for testing
        n_x = length(states_x(momdp))
        n_y = length(states_y(momdp))
        n_actions = length(actions(momdp))
        
        # Create simple alphas with different values for different actions
        alphas = Vector{Vector{Vector{Float64}}}(undef, n_x)
        action_map = Vector{Vector{Int}}(undef, n_x)
        
        for x_idx in 1:n_x
            # Create multiple alpha vectors for different actions
            n_alphas = min(3, n_actions)  # Limit for testing
            alphas[x_idx] = Vector{Vector{Float64}}(undef, n_alphas)
            action_map[x_idx] = Vector{Int}(undef, n_alphas)
            
            for alpha_idx in 1:n_alphas
                alphas[x_idx][alpha_idx] = zeros(n_y)
                action_map[x_idx][alpha_idx] = alpha_idx
                
                # Create different value patterns for different actions
                for y_idx in 1:n_y
                    alphas[x_idx][alpha_idx][y_idx] = alpha_idx * (n_y - y_idx + 1)
                end
            end
        end
        
        policy = MOMDPAlphaVectorPolicy(momdp, n_x, n_y, alphas, action_map)
        
        @testset "Basic actionvalues Computation" begin
            # Test with uniform belief over hidden states
            b_uniform = uniform_belief_y(momdp)
            x = first(states_x(momdp))
            
            qa = MOMDPs.actionvalues(policy, b_uniform, x)
            
            @test isa(qa, Vector{Float64})
            @test length(qa) == n_actions
            @test all(isfinite.(qa))  # All values should be finite
            
            # Action values should be real numbers (can be negative due to step penalties)
            @test all(isa(q, Real) for q in qa)
        end
        
        @testset "Different Beliefs" begin
            x = first(states_x(momdp))
            
            # Test with deterministic belief
            det_state = first(ordered_states_y(momdp))
            b_det = SparseCat([det_state], [1.0])
            qa_det = MOMDPs.actionvalues(policy, b_det, x)
            
            @test length(qa_det) == n_actions
            @test all(isfinite.(qa_det))
            
            # Test with structured belief
            states_y_list = ordered_states_y(momdp)
            structured_probs = [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
            b_structured = DiscreteBelief(momdp, states_y_list, structured_probs)
            qa_structured = MOMDPs.actionvalues(policy, b_structured, x)
            
            @test length(qa_structured) == n_actions
            @test all(isfinite.(qa_structured))
        end
        
        @testset "Different Visible States" begin
            b = uniform_belief_y(momdp)
            states_x_list = collect(states_x(momdp))
            
            # Test with different visible states
            x1 = states_x_list[1]
            x2 = states_x_list[min(2, length(states_x_list))]
            
            qa1 = MOMDPs.actionvalues(policy, b, x1)
            qa2 = MOMDPs.actionvalues(policy, b, x2)
            
            @test length(qa1) == n_actions
            @test length(qa2) == n_actions
            @test all(isfinite.(qa1))
            @test all(isfinite.(qa2))
        end
    end
    
    @testset "File I/O Tests" begin
        MOMDPs.is_y_prime_dependent_on_x_prime(::RockSampleMOMDP) = false
        MOMDPs.is_x_prime_dependent_on_y(::RockSampleMOMDP) = false
        MOMDPs.is_initial_distribution_independent(::RockSampleMOMDP) = true
        
        _, momdp, _ = create_test_rocksample()
        
        # Create a POMDPXFile
        pomdpx = POMDPXFile("test_momdp.pomdpx"; description="Test MOMDP")
        
        # Write MOMDP to POMDPX file
        write(momdp, pomdpx)
        
        # Number of lines in the POMDPX file
        n_lines_momdp = countlines("test_momdp.pomdpx")
        
        # Test file exists
        @test isfile("test_momdp.pomdpx")
        
        isfile("test_momdp.pomdpx") && rm("test_momdp.pomdpx")
        
        # Chnage the dependencies in the MOMDP and check that the new POMDPX file is larger
        MOMDPs.is_y_prime_dependent_on_x_prime(::RockSampleMOMDP) = true
        
        pomdpx = POMDPXFile("test_momdp.pomdpx")
        write(momdp, pomdpx)
        n_lines_tft = countlines("test_momdp.pomdpx")
        @test n_lines_tft > n_lines_momdp
        
        MOMDPs.is_x_prime_dependent_on_y(::RockSampleMOMDP) = true
        write(momdp, pomdpx)
        n_lines_ttt = countlines("test_momdp.pomdpx")
        @test n_lines_ttt > n_lines_tft
        
        MOMDPs.is_initial_distribution_independent(::RockSampleMOMDP) = false
        write(momdp, pomdpx)
        n_lines_ttf = countlines("test_momdp.pomdpx")
        @test n_lines_ttf > n_lines_ttt
        
        # Clean up
        isfile("test_momdp.pomdpx") && rm("test_momdp.pomdpx")
    end
    
    Aqua.test_all(MOMDPs)
    
end
