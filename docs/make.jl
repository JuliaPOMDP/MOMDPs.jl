using Documenter, POMDPs, MOMDPs

page_order = [
    "MOMDPs.jl" => "index.md",
    "Defining a MOMDP" => "defining_momdp.md",
    "Policies" => "policies.md",
    "Discrete Helper Functions" => "discrete_momdp_functions.md",
    "Examples" => "examples.md",
    "API" => "api.md"
]

makedocs(
    sitename = "MOMDPs.jl",
    authors = "Dylan Asmar",
    modules = [MOMDPs],
    format = Documenter.HTML(),
    warnonly = [:missing_docs],
    pages = page_order
)

deploydocs(
    repo = "github.com/JuliaPOMDP/MOMDPs.jl.git",
    push_preview = true
)
