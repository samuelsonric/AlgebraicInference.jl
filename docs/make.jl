using AlgebraicInference
using Documenter
using Literate

Literate.markdown(
    joinpath(@__DIR__, "literate", "regression.jl"),
    joinpath(@__DIR__, "src", "generated");
    credit = false,
)

makedocs(
    modules = [AlgebraicInference],
    sitename = "AlgebraicInference.jl",
    pages = [
        "AlgebraicInference.jl" => "index.md",
        "Examples" => "generated/regression.md",
        "Library Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/samuelsonric/AlgebraicInference.jl.git",
)
