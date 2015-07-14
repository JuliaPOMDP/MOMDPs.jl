module MOMDPs

include("reexport.jl")
@reexport using POMDPs

export 
    MOMDP,
    create_fully_obs_transition,
    create_partially_obs_transition,
    fully_obs_space,
    part_obs_space


abstract MOMDP <: POMDP

create_fully_obs_transition(pomdp::MOMDP) = error("$(typeof(pomdp)) does not implement create_fully_obs_transition") 
create_partially_obs_transition(pomdp::MOMDP) = error("$(typeof(pomdp)) does not implement create_part_obs_transition")   

fully_obs_space(pomdp::POMDP) = error("$(typeof(pomdp)) does not implement fully_obs_space")   
part_obs_space(pomdp::POMDP) = error("$(typeof(pomdp)) does not implement part_obs_space")   

end # module
