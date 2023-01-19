using AlgebraicInference
using Documenter
using Literate

Literate.markdown(
    joinpath(@__DIR__, "literate", "kalman.jl"),
    joinpath(@__DIR__, "src", "generated");
    credit = false,
)

makedocs(
    modules = [AlgebraicInference],
    sitename = "AlgebraicInference.jl",
    pages = [
        "AlgebraicInference.jl" => "index.md",
        "Examples" => "generated/kalman.md",
        "Library Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/samuelsonric/AlgebraicInference.jl.git",
)
