"""
    write(momdp::MOMDP, pomdpx::AbstractPOMDPXFile)

Write a MOMDP to a POMDPXFile. This file is very similar to 
Base.write(::POMDP, ::AbstractPOMDPXFile) as defined in POMDPXFiles.jl with the only 
difference being the formatting and counting of lines to be written.
"""
function Base.write(momdp::MOMDP, pomdpx::AbstractPOMDPXFile)
    file_name = pomdpx.file_name
    discount_factor = discount(momdp)

    # Open file to write to
    out_file = open("$file_name", "w")

    n_xs = length(ordered_states_x(momdp))
    n_ys = length(ordered_states_y(momdp))
    n_as = length(ordered_actions(momdp))
    n_os = length(ordered_observations(momdp))
    
    if is_x_prime_dependent_on_y(momdp)
        n_belief_xml = n_xs + n_xs * n_ys
    else
        n_belief_xml = n_xs + n_ys
    end
    if is_x_prime_dependent_on_y(momdp)
        n_trans_xml_x = n_xs * n_ys * n_as * n_xs
    else
        n_trans_xml_x = n_xs * n_as * n_xs
    end
    
    if is_y_prime_dependent_on_x_prime(momdp)
        n_trans_xml_y = n_xs * n_ys * n_as * n_xs * n_ys
    else
        n_trans_xml_y = n_xs * n_ys * n_as * n_ys 
    end
    n_obs_xml = n_xs * n_ys * n_as * n_os
    n_reward_xml = n_xs * n_ys * n_as
    
    # Not true number, but the large majority for progress awareness
    n_xml_lines = n_belief_xml + n_trans_xml_x + n_trans_xml_y + n_obs_xml + n_reward_xml
    p1 = Progress(n_xml_lines, dt=0.1)

    # Header for xml
    write(out_file, "<?xml version='1.0' encoding='ISO-8859-1'?>\n\n\n")
    write(out_file, "<pomdpx version='0.1' id='test' ")
    write(out_file, "xmlns:='http://www.w3.org/2001/XMLSchema-instance' ")
    write(out_file, "xsi:noNamespaceSchemaLocation='pomdpx.xsd'>\n\n\n")
    write(out_file, "\t<Description> $(pomdpx.description)</Description>\n\n\n")
    
    write(out_file, "\t<Discount>$(discount_factor)</Discount>\n\n\n")
    
    write(out_file, "\t<Variable>\n")
    write(out_file, state_xml(momdp, pomdpx))
    write(out_file, POMDPXFiles.action_xml(momdp, pomdpx))
    write(out_file, POMDPXFiles.obs_var_xml(momdp, pomdpx))
    write(out_file, POMDPXFiles.reward_var_xml(momdp, pomdpx))
    write(out_file, "\t</Variable>\n\n\n")
    
    belief_xml(momdp, pomdpx, out_file, p1)
    trans_xml(momdp, pomdpx, out_file, p1)
    obs_xml(momdp, pomdpx, out_file, p1)
    reward_xml(momdp, pomdpx, out_file, p1)
    
    finish!(p1)

    write(out_file, "</pomdpx>")
    close(out_file)
    
    println("POMDPX file written successfully.")
end

function state_xml(momdp::MOMDP, pomdpx::POMDPXFile)
    n_xs = length(states_x(momdp))
    n_ys = length(states_y(momdp))
    xname = pomdpx.state_name * "x"
    yname = pomdpx.state_name * "y"
    
    str = ""
    # visible states
    str *= "\t\t<StateVar vnamePrev=\"$(xname)0\" vnameCurr=\"$(xname)1\" fullyObs=\"true\">\n"
    str *= "\t\t\t<NumValues>$(n_xs)</NumValues>\n"
    str *= "\t\t</StateVar>\n\n"
    
    # hidden states
    str *= "\t\t<StateVar vnamePrev=\"$(yname)0\" vnameCurr=\"$(yname)1\" fullyObs=\"false\">\n"
    str *= "\t\t\t<NumValues>$(n_ys)</NumValues>\n"
    str *= "\t\t</StateVar>\n\n"
    
    return str
end

function belief_xml(momdp::MOMDP, pomdpx::POMDPXFile, out_file::IOStream, p1)
    xname = pomdpx.state_name * "x"
    yname = pomdpx.state_name * "y"
    
    # Initial belief distribution for visible states
    write(out_file, "\t\t<InitialStateBelief>\n")
    str = "\t\t\t<CondProb>\n"
    str *= "\t\t\t\t<Var>$(xname)0</Var>\n"
    str *= "\t\t\t\t<Parent>null</Parent>\n"
    str *= "\t\t\t\t<Parameter type = \"TBL\">\n"
    
    dx = initialstate_x(momdp)
    for (i, xi) in enumerate(ordered_states_x(momdp))
        p_xi = pdf(dx, xi)
        if p_xi > 0.0
            str *= "\t\t\t\t\t<Entry>\n"
            str *= "\t\t\t\t\t\t<Instance>s$(i-1)</Instance>\n"
            str *= "\t\t\t\t\t\t<ProbTable>$(p_xi)</ProbTable>\n"
            str *= "\t\t\t\t\t</Entry>\n"
        end
        next!(p1)
    end
    str *= "\t\t\t\t</Parameter>\n"
    str *= "\t\t\t</CondProb>\n"
    write(out_file, str)
    
    # Initial belief distribution for hidden states
    str = "\t\t\t<CondProb>\n"
    str *= "\t\t\t\t<Var>$(yname)0</Var>\n"
    if !is_initial_distribution_independent(momdp)
        str *= "\t\t\t\t<Parent>$(xname)0</Parent>\n"
    else
        str *= "\t\t\t\t<Parent>null</Parent>\n"
    end
    str *= "\t\t\t\t<Parameter type = \"TBL\">\n"
    
    if is_initial_distribution_independent(momdp)
        dy = initialstate_y(momdp, first(ordered_states_x(momdp)))
        for (j, yi) in enumerate(ordered_states_y(momdp))
            p_yi = pdf(dy, yi)
            if p_yi > 0.0
                str *= "\t\t\t\t\t<Entry>\n"
                str *= "\t\t\t\t\t\t<Instance>s$(j-1)</Instance>\n"
                str *= "\t\t\t\t\t\t<ProbTable>$(p_yi)</ProbTable>\n"
                str *= "\t\t\t\t\t</Entry>\n"
            end
            next!(p1)
        end
    else
        for (i, xi) in enumerate(ordered_states_x(momdp))
            dy = initialstate_y(momdp, xi)
            p_xi = pdf(dx, xi)
            if p_xi > 0.0
                for (j, yi) in enumerate(ordered_states_y(momdp))
                    p_yi = pdf(dy, yi)
                    if p_yi > 0.0
                        str *= "\t\t\t\t\t<Entry>\n"
                        str *= "\t\t\t\t\t\t<Instance>s$(i-1) s$(j-1)</Instance>\n"
                        str *= "\t\t\t\t\t\t<ProbTable>$(p_yi)</ProbTable>\n"
                        str *= "\t\t\t\t\t</Entry>\n"
                    end
                    next!(p1)
                end
            else
                # Initial state has probability 0, so these probabilites don't matter.
                # However, we need to write something to the file as the C++ SARSOP code checks
                # the make sure appropriate probabilties are written for all states.
                pt = 1 / length(ordered_states_y(momdp))
                str *= "\t\t\t\t\t<Entry>\n"
                str *= "\t\t\t\t\t\t<Instance>s$(i-1) *</Instance>\n"
                str *= "\t\t\t\t\t\t<ProbTable>$(pt)</ProbTable>\n"
                str *= "\t\t\t\t\t</Entry>\n"
                for _ in 1:length(ordered_states_y(momdp))
                    next!(p1)
                end
            end
        end
    end 
        
    str *= "\t\t\t\t</Parameter>\n"
    str *= "\t\t\t</CondProb>\n"
    write(out_file, str)
    
    write(out_file, "\t\t</InitialStateBelief>\n\n\n")
end

function trans_xml(momdp::MOMDP, pomdpx::POMDPXFile, out_file::IOStream, p1)
    xname = pomdpx.state_name * "x"
    yname = pomdpx.state_name * "y"
    aname = pomdpx.action_name
    
    xs = ordered_states_x(momdp)
    ys = ordered_states_y(momdp)
    acts = ordered_actions(momdp)

    write(out_file, "\t<StateTransitionFunction>\n")
    
    # Transition probability table for visible states
    str = "\t\t<CondProb>\n"
    str *= "\t\t\t<Var>$(xname)1</Var>\n"
    if is_x_prime_dependent_on_y(momdp)
        str *= "\t\t\t<Parent>$(aname) $(xname)0 $(yname)0</Parent>\n"
    else
        str *= "\t\t\t<Parent>$(aname) $(xname)0</Parent>\n"
    end
    str *= "\t\t\t<Parameter type = \"TBL\">\n"
    write(out_file, str)
    
    if is_x_prime_dependent_on_y(momdp)
        for (i, xi) in enumerate(xs)
            for (j, yi) in enumerate(ys)
                if isterminal(momdp, (xi, yi))
                    str = "\t\t\t\t<Entry>\n"
                    str *= "\t\t\t\t\t<Instance>* s$(i-1) s$(j-1) s$(i-1)</Instance>\n"
                    str *= "\t\t\t\t\t<ProbTable>1.0</ProbTable>\n"
                    str *= "\t\t\t\t</Entry>\n"
                    write(out_file, str)
                    for _ in 1:(length(acts) * length(xs))
                        next!(p1)
                    end
                else
                    str = ""
                    for (k, ai) in enumerate(acts)
                        dx = transition_x(momdp, (xi, yi), ai)
                        for (l, xip) in enumerate(xs)
                            p = pdf(dx, xip)
                            if p > 0.0
                                str *= "\t\t\t\t<Entry>\n"
                                str *= "\t\t\t\t\t<Instance>a$(k-1) s$(i-1) s$(j-1) s$(l-1)</Instance>\n"
                                str *= "\t\t\t\t\t<ProbTable>$(p)</ProbTable>\n"
                                str *= "\t\t\t\t</Entry>\n"
                            end
                            next!(p1)
                        end
                    end
                    write(out_file, str)
                end
            end
        end
    else
        for (i, xi) in enumerate(xs)
            if isterminal(momdp, (xi, first(ys)))
                str = "\t\t\t\t<Entry>\n"
                str *= "\t\t\t\t\t<Instance>* s$(i-1) s$(i-1)</Instance>\n"
                str *= "\t\t\t\t\t<ProbTable>1.0</ProbTable>\n"
                str *= "\t\t\t\t</Entry>\n"
                write(out_file, str)
                for _ in 1:length(acts)
                    next!(p1)
                end
            else
                str = ""
                for (k, ai) in enumerate(acts)
                    dx = transition_x(momdp, (xi, first(ys)), ai)
                    for (l, xip) in enumerate(xs)
                        p = pdf(dx, xip)
                        if p > 0.0
                            str *= "\t\t\t\t<Entry>\n"
                            str *= "\t\t\t\t\t<Instance>a$(k-1) s$(i-1) s$(l-1)</Instance>\n"
                            str *= "\t\t\t\t\t<ProbTable>$(p)</ProbTable>\n"
                            str *= "\t\t\t\t</Entry>\n"
                        end
                        next!(p1)
                    end
                end
                write(out_file, str)
            end
        end
    end
    str = "\t\t\t</Parameter>\n"
    str *= "\t\t</CondProb>\n"
    write(out_file, str)
    
    # Transition probability table for hidden states
    str = "\t\t<CondProb>\n"
    str *= "\t\t\t<Var>$(yname)1</Var>\n"
    if is_y_prime_dependent_on_x_prime(momdp)
        str *= "\t\t\t<Parent>$(aname) $(xname)0 $(yname)0 $(xname)1</Parent>\n"
    else
        str *= "\t\t\t<Parent>$(aname) $(xname)0 $(yname)0</Parent>\n"
    end
    str *= "\t\t\t<Parameter type = \"TBL\">\n"
    write(out_file, str)
    
    if is_y_prime_dependent_on_x_prime(momdp)
        for (i, xi) in enumerate(xs)
            for (j, yi) in enumerate(ys)
                if isterminal(momdp, (xi, yi))
                    str = "\t\t\t\t<Entry>\n"
                    str *= "\t\t\t\t\t<Instance>* s$(i-1) s$(j-1) * s$(j-1)</Instance>\n"
                    str *= "\t\t\t\t\t<ProbTable>1.0</ProbTable>\n"
                    str *= "\t\t\t\t</Entry>\n"
                    write(out_file, str)
                    for _ in 1:(length(acts) * length(xs) * length(ys))
                        next!(p1)
                    end
                else
                    str = ""
                    for (k, ai) in enumerate(acts)
                        dx = transition_x(momdp, (xi, yi), ai)
                        for (l, xip) in enumerate(xs)
                            p_xip = pdf(dx, xip)
                            if p_xip > 0.0
                                dy = transition_y(momdp, (xi, yi), ai, xip)
                                for (m, yip) in enumerate(ys)
                                    p_yip = pdf(dy, yip)
                                    if p_yip > 0.0
                                        str *= "\t\t\t\t<Entry>\n"
                                        str *= "\t\t\t\t\t<Instance>a$(k-1) s$(i-1) s$(j-1) s$(l-1) s$(m-1)</Instance>\n"
                                        str *= "\t\t\t\t\t<ProbTable>$(p_yip)</ProbTable>\n"
                                        str *= "\t\t\t\t</Entry>\n"
                                    end
                                    next!(p1)
                                end
                            else
                                # These probabilities don't matter since p_xip = 0, but required
                                # by the C++ SARSOP code.
                                pt = 1 / length(ys)
                                str *= "\t\t\t\t<Entry>\n"
                                str *= "\t\t\t\t\t<Instance>a$(k-1) s$(i-1) s$(j-1) s$(l-1) *</Instance>\n"
                                str *= "\t\t\t\t\t<ProbTable>$(pt)</ProbTable>\n"
                                str *= "\t\t\t\t</Entry>\n"
                                for _ in 1:length(ys)
                                    next!(p1)
                                end
                            end
                        end    
                    end
                    write(out_file, str)
                end
            end
        end
    else
        for (i, xi) in enumerate(xs)
            for (j, yi) in enumerate(ys)
                if isterminal(momdp, (xi, yi))
                    str = "\t\t\t\t<Entry>\n"
                    str *= "\t\t\t\t\t<Instance>* s$(i-1) s$(j-1) s$(j-1)</Instance>\n"
                    str *= "\t\t\t\t\t<ProbTable>1.0</ProbTable>\n"
                    str *= "\t\t\t\t</Entry>\n"
                    write(out_file, str)
                    for _ in 1:(length(acts) * length(xs) * length(ys))
                        next!(p1)
                    end
                else
                    str = ""
                    for (k, ai) in enumerate(acts)    
                        dy = transition_y(momdp, (xi, yi), ai, first(xs))
                        for (m, yip) in enumerate(ys)
                            p_yip = pdf(dy, yip)
                            if p_yip > 0.0
                                str *= "\t\t\t\t<Entry>\n"
                                str *= "\t\t\t\t\t<Instance>a$(k-1) s$(i-1) s$(j-1) s$(m-1)</Instance>\n"
                                str *= "\t\t\t\t\t<ProbTable>$(p_yip)</ProbTable>\n"
                                str *= "\t\t\t\t</Entry>\n"
                            end
                            next!(p1)
                        end
                    end
                    write(out_file, str)
                end
            end
        end
    end
        
    str = "\t\t\t</Parameter>\n"
    str *= "\t\t</CondProb>\n"
    write(out_file, str)
    write(out_file, "\t</StateTransitionFunction>\n\n\n")
end

function obs_xml(momdp::MOMDP, pomdpx::POMDPXFile, out_file::IOStream, p1)
    xname = pomdpx.state_name * "x"
    yname = pomdpx.state_name * "y"
    aname = pomdpx.action_name
    oname = pomdpx.obs_name

    xs = ordered_states_x(momdp)
    ys = ordered_states_y(momdp)
    acts = ordered_actions(momdp)
    obs = ordered_observations(momdp)
    
    write(out_file, "\t<ObsFunction>\n")
    str = "\t\t<CondProb>\n"
    str *= "\t\t\t<Var>$(oname)</Var>\n"
    str *= "\t\t\t<Parent>$(aname) $(xname)1 $(yname)1</Parent>\n"
    str *= "\t\t\t<Parameter type = \"TBL\">\n"
    write(out_file, str)
    
    try observation(momdp, first(acts), (first(xs), first(ys)))
    catch ex
        if ex isa MethodError
            @warn("""POMDPXFiles only supports observation distributions conditioned on a and sp.

                  Check that there is an `observation(::M, ::A, ::S)` method available (or an (::A, ::S) method of the observation function for a QuickPOMDP).
                  
                  This warning is designed to give a helpful hint to fix errors, but may not always be relevant.
                  """, M=typeof(pomdp), S=typeof(first(pomdp_states)), A=typeof(first(acts)))
        end
        rethrow(ex)
    end
    
    for (i, xi) in enumerate(xs)
        for (j, yi) in enumerate(ys)
            str = ""
            for (k, ai) in enumerate(acts)
                d_o = observation(momdp, ai, (xi, yi))
                for (l, oi) in enumerate(obs)
                    p = pdf(d_o, oi)
                    if p > 0.0
                        str *= "\t\t\t\t<Entry>\n"
                        str *= "\t\t\t\t\t<Instance>a$(k-1) s$(i-1) s$(j-1) o$(l-1)</Instance>\n"
                        str *= "\t\t\t\t\t<ProbTable>$(p)</ProbTable>\n"
                        str *= "\t\t\t\t</Entry>\n"
                    end
                    next!(p1)
                end
            end
            write(out_file, str)
        end
    end
    str = "\t\t\t</Parameter>\n"
    str *= "\t\t</CondProb>\n"
    write(out_file, str)
    write(out_file, "\t</ObsFunction>\n\n\n")
end

function reward_xml(momdp::MOMDP, pomdpx::POMDPXFile, out_file::IOStream, p1)
    xname = pomdpx.state_name * "x"
    yname = pomdpx.state_name * "y"
    aname = pomdpx.action_name
    rname = pomdpx.reward_name
    
    xs = ordered_states_x(momdp)
    ys = ordered_states_y(momdp)
    acts = ordered_actions(momdp)
    
    write(out_file, "\t<RewardFunction>\n")
    str = "\t\t<Func>\n"
    str *= "\t\t\t<Var>$(rname)</Var>\n"
    str *= "\t\t\t<Parent>$(aname) $(xname)0 $(yname)0</Parent>\n"
    str *= "\t\t\t<Parameter type = \"TBL\">\n"
    write(out_file, str)

    for (i, xi) in enumerate(xs)
        for (j, yi) in enumerate(ys)
            str = ""
            for (k, ai) in enumerate(acts)
                r = reward(momdp, (xi, yi), ai)
                str *= "\t\t\t\t<Entry>\n"
                str *= "\t\t\t\t\t<Instance>a$(k-1) s$(i-1) s$(j-1)</Instance>\n"
                str *= "\t\t\t\t\t<ValueTable>$(r)</ValueTable>\n"
                str *= "\t\t\t\t</Entry>\n"
                next!(p1)
            end
            write(out_file, str)
        end
    end
    str = "\t\t\t</Parameter>\n"
    str *= "\t\t</Func>\n"
    write(out_file, str)
    write(out_file, "\t</RewardFunction>\n\n\n")
end

"""
    MOMDPAlphas

A structure representing alpha vectors for a MOMDP.

# Fields
- `alpha_vectors::Matrix{Float64}`: Matrix of alpha vectors, where each column represents an alpha vector
- `alpha_actions::Vector{Int64}`: Vector of actions associated with each alpha vector (0-indexed)
- `alpha_visible_states::Vector{Int64}`: Vector of visible states associated with each alpha vector

# Constructors
- `MOMDPAlphas(av::Matrix{Float64}, aa::Vector{Int64}, avs::Vector{Int64})`: Create with explicit alpha vectors, actions, and visible states
- `MOMDPAlphas(av::Matrix{Float64})`: Create with alpha vectors only, automatically generating 0-indexed actions
- `MOMDPAlphas(filename::AbstractString)`: Create by reading policy from a file
- `MOMDPAlphas()`: Create an empty structure with zero-sized arrays

# Notes
- The actions are 0-indexed to match SARSOP output format
- Each alpha vector is associated with a specific action and a specific visible state
"""
mutable struct MOMDPAlphas <: Alphas
    alpha_vectors::Matrix{Float64}
    alpha_actions::Vector{Int64}
    alpha_visible_states::Vector{Int64}

    MOMDPAlphas(av::Matrix{Float64}, aa::Vector{Int64}, avs::Vector{Int64}) = new(av, aa, avs)

    # Constructor if no action list is given
    # Here, we 0-index actions, to match sarsop output
    function MOMDPAlphas(av::Matrix{Float64})
        numActions = size(av, 1)
        alist = [0:(numActions-1)]
        return new(av, alist, Int64[])
    end

    # Constructor reading policy from file
    function MOMDPAlphas(filename::AbstractString)
        alpha_vectors, alpha_actions, alpha_visible_states = read_momdp(filename)
        return new(alpha_vectors, alpha_actions, alpha_visible_states)
    end

    # Default constructor
    function MOMDPAlphas()
        return new(zeros(0,0), zeros(Int64,0), zeros(Int64,0))
    end
end

"""
    read_momdp(filename::String)

Read a MOMDP policy from a POMDPX file containing alpha vectors.

This function parses a POMDPX file that contains a policy represented as alpha vectors. The file should have a structure with `Policy` as the root tag, containing `AlphaVector` tags with associated `Vector` elements.

# Arguments
- `filename::String`: Path to the POMDPX file containing the policy

# Returns
- `alpha_vectors::Matrix{Float64}`: Matrix of alpha vectors, where each column represents an alpha vector
- `action_indices::Vector{Int64}`: Vector of action indices associated with each alpha vector
- `observable_indices::Vector{Int64}`: Vector of observable state indices associated with each alpha vector

# Notes
- The function expects the POMDPX file to have a specific structure with `Policy`, `AlphaVector`, and `Vector` tags
- Action and observable state values are converted from strings to integers
- The alpha vectors are stored as columns in the returned matrix
- The number of vectors and their length are determined from the XML attributes `numVectors` and `vectorLength`
"""
function read_momdp(filename::String)
    xdoc = POMDPXFiles.parse_file(filename)

    # Get the root of the document (the Policy tag in this case)
    policy_tag = POMDPXFiles.root(xdoc)

    # Determine expected number of vectors and their length
    alphavector_tag = POMDPXFiles.get_elements_by_tagname(policy_tag, "AlphaVector")[1]
    num_vectors = POMDPXFiles.parse(Int64, POMDPXFiles.attribute(alphavector_tag, "numVectors"))
    vector_length = POMDPXFiles.parse(Int64, POMDPXFiles.attribute(alphavector_tag, "vectorLength"))

    # Arrays with vector tags
    vector_tags = POMDPXFiles.get_elements_by_tagname(alphavector_tag, "Vector")

    # Initialize the gamma matrix. This is basically a matrix with the alpha
    #   vectors as columns.
    alpha_vectors = Array{Float64}(undef, vector_length, num_vectors)
    alpha_actions = Array{String}(undef, num_vectors)
    observable_states = Array{String}(undef, num_vectors)
    gammarow = 1

    # Fill in gamma
    for vector in vector_tags
        alpha = parse.(Float64, split(POMDPXFiles.content(vector)))
        alpha_vectors[:,gammarow] = alpha
        alpha_actions[gammarow] = POMDPXFiles.attribute(vector, "action")
        observable_states[gammarow] = POMDPXFiles.attribute(vector, "obsValue")
        gammarow += 1
    end
    
    action_indices = [parse(Int64,a) for a in alpha_actions]
    observable_indices = [parse(Int64,s) for s in observable_states]
    
    return alpha_vectors, action_indices, observable_indices
end
