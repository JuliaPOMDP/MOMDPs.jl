module MOMDPs

using POMDPs
using POMDPTools
using POMDPLinter
using LinearAlgebra
using POMDPXFiles
using ProgressMeter
using SARSOP

# Needed for reading POMDPX files
import LightXML: parse_file, root, get_elements_by_tagname, attribute, content

"""
    Notation matching Ong, Sylvie CW, et al. "POMDPs for robotic tasks with mixed 
    observability." Robotics: Science and systems. Vol. 5. No. 4. 2009. 
    https://www.comp.nus.edu.sg/~leews/publications/rss09.pdf
"""

export 
    MOMDP,
    
    states_x,
    states_y,
    
    transition_x,
    transition_y,
    
    stateindex_x,
    stateindex_y,
    
    statetype_x,
    statetype_y,
    
    ordered_states_x,
    ordered_states_y,
    
    initialstate_x,
    initialstate_y,
    
    is_y_prime_dependent_on_x_prime,
    is_x_prime_dependent_on_y,
    is_initial_distribution_independent,
    
    POMDP_of_Discrete_MOMDP,
    
    MOMDPAlphas,
    read_momdp,
    MOMDPAlphaVectorPolicy
    
    
include("momdp.jl")
include("discrete_momdp_functions.jl")
include("pomdp_of_momdp.jl")
include("pomdpxfiles.jl")
include("alpha_vector.jl")
include("sarsop.jl")

end # module
