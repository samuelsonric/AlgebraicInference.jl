using AlgebraicInference
using BayesNets
using Catlab, Catlab.WiringDiagrams
using Documenter
using Literate
using Random

for file in readdir(joinpath(@__DIR__, "literate"))
    Literate.markdown(
        joinpath(@__DIR__, "literate", file),
        joinpath(@__DIR__, "src", "generated");
        credit = false)
end

makedocs(
    modules = [AlgebraicInference],
    sitename = "AlgebraicInference.jl",
    pages = [
        "AlgebraicInference.jl" => "index.md",
        "Examples" => [
            "generated/regression.md",
            "generated/kalman.md",
            "generated/pixel.md"
        ],
        "Library Reference" => "api.md",
    ])

deploydocs(repo="github.com/samuelsonric/AlgebraicInference.jl.git")
