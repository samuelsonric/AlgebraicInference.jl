using AlgebraicInference
using Documenter
using Literate

for file in readdir(joinpath(@__DIR__, "literate"))
    Literate.markdown(
        joinpath(@__DIR__, "literate", file),
        joinpath(@__DIR__, "src", "generated");
        credit = false,
    )
end

makedocs(
    modules = [AlgebraicInference],
    sitename = "AlgebraicInference.jl",
    pages = [
        "AlgebraicInference.jl" => "index.md",
        "Examples" => [
            "generated/inference.md",
            "generated/filter_undirected.md",
            "generated/filter_directed.md",
            "generated/regression.md",
],
        "Library Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/samuelsonric/AlgebraicInference.jl.git",
)
