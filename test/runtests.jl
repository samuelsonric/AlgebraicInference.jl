using AbstractTrees
using AlgebraicInference
using BayesNets
using Catlab.Programs
using Distributions
using FillArrays
using Graphs
using Test


@testset "Construction" begin
    Σ = normal([3, 1], [1 1; 1 1])
    @test Σ.P ≈ [1/4  1/4;  1/4  1/4]
    @test Σ.S ≈ [1/2 -1/2; -1/2  1/2]
    @test Σ.p ≈ [1,  1]
    @test Σ.s ≈ [1, -1]
    @test Σ.σ ≈ 2

    Σ = normal([3, 1], Eye(2))
    @test Σ.P == [1 0; 0 1]
    @test Σ.S == [0 0; 0 0]
    @test Σ.p == [3, 1]
    @test Σ.s == [0, 0]
    @test Σ.σ == 0

    Σ = normal([3, 1], Zeros(2, 2))
    @test Σ.P == [0 0; 0 0]
    @test Σ.S == [1 0; 0 1]
    @test Σ.p == [0, 0]
    @test Σ.s == [3, 1]
    @test Σ.σ == 10

    Σ = normal(1, 1/2)
    @test Σ.P == [4;;]
    @test Σ.S == [0;;]
    @test Σ.p == [4]
    @test Σ.s == [0]
    @test Σ.σ == 0

    Σ = kernel([1 0; 0 1], [3, 1], [1 1; 1 1])
    @test Σ.P ≈ [1/4  1/4 -1/4 -1/4;  1/4  1/4 -1/4 -1/4; -1/4 -1/4  1/4  1/4; -1/4 -1/4  1/4  1/4]
    @test Σ.S ≈ [1/2 -1/2 -1/2  1/2; -1/2  1/2  1/2 -1/2; -1/2  1/2  1/2 -1/2;  1/2 -1/2 -1/2  1/2]
    @test Σ.p ≈ [-1, -1,  1,  1]
    @test Σ.s ≈ [-1,  1,  1, -1]
    @test Σ.σ ≈ 2

    Σ = kernel([1], 1, 1/2)
    @test Σ.P == [4 -4; -4  4]
    @test Σ.S == [0  0;  0  0]
    @test Σ.p == [-4,  4]
    @test Σ.s == [ 0,  0]
    @test Σ.σ == 0
end


@testset "Elimination" begin
    graph = Graph(8)

    add_edge!(graph, 1, 7)
    add_edge!(graph, 2, 3)
    add_edge!(graph, 2, 4)
    add_edge!(graph, 2, 6)
    add_edge!(graph, 3, 4)
    add_edge!(graph, 4, 5)
    add_edge!(graph, 4, 7)
    add_edge!(graph, 4, 8)
    add_edge!(graph, 5, 6)
    add_edge!(graph, 5, 7);

    order₁ = AlgebraicInference.EliminationOrder(graph, MinFill())
    @test order₁ == [1, 8, 7, 3, 6, 4, 5, 2]

    order₂ = AlgebraicInference.EliminationOrder(graph, MinDegree())
    @test order₂ == [1, 8, 7, 6, 5, 4, 3, 2]

    tree = AlgebraicInference.EliminationTree(graph, order₁)
    @test rootindex(tree) == 2

    @test parentindex(tree, 1) == 7
    @test parentindex(tree, 2) == nothing
    @test parentindex(tree, 3) == 4
    @test parentindex(tree, 4) == 5
    @test parentindex(tree, 5) == 2
    @test parentindex(tree, 6) == 5
    @test parentindex(tree, 7) == 4
    @test parentindex(tree, 8) == 4

    @test issetequal(childindices(tree, 1), [])
    @test issetequal(childindices(tree, 2), [5])
    @test issetequal(childindices(tree, 3), [])
    @test issetequal(childindices(tree, 4), [3, 7, 8])
    @test issetequal(childindices(tree, 5), [4, 6])
    @test issetequal(childindices(tree, 6), [])
    @test issetequal(childindices(tree, 7), [1])
    @test issetequal(childindices(tree, 8), [])

    @test issetequal(tree[1], [7])
    @test issetequal(tree[2], [])
    @test issetequal(tree[3], [2, 4])
    @test issetequal(tree[4], [2, 5])
    @test issetequal(tree[5], [2])
    @test issetequal(tree[6], [2, 5])
    @test issetequal(tree[7], [4, 5])
    @test issetequal(tree[8], [4])
end


# Example 9
# https://www.kalmanfilter.net/multiExamples.html
@testset "Kalman Filter" begin
    F =  [
        1   1   1/2 0   0   0
        0   1   1   0   0   0
        0   0   1   0   0   0
        0   0   0   1   1   1/2
        0   0   0   0   1   1
        0   0   0   0   0   1
    ]

    Q =  [
        1/4 1/2 1/2 0   0   0
        1/2 1   1   0   0   0
        1/2 1   1   0   0   0
        0   0   0   1/4 1/2 1/2
        0   0   0   1/2 1   1
        0   0   0   1/2 1   1
    ] * 1/25

    H =  [
        1   0   0   0   0   0
        0   0   0   1   0   0
    ]

    R =  [
        9   0
        0   9
    ]

    P = [
        500 0   0   0   0   0
        0   500 0   0   0   0
        0   0   500 0   0   0
        0   0   0   500 0   0
        0   0   0   0   500 0
        0   0   0   0   0   500
    ]

    z₁ = [
       -393.66
        300.40
    ]

    z₂ = [
       -375.93
        301.78
    ]

    true_cov = [
        8.92    11.33   5.13    0       0       0
        11.33   61.1    75.4    0       0       0
        5.13    75.4    126.5   0       0       0
        0       0       0       8.92    11.33   5.13
        0       0       0       11.33   61.1    75.4
        0       0       0       5.13    75.4    126.5]

    true_mean = [
       -378.9
        53.8
        94.5
        303.9
       -22.3
       -63.6
    ]


    uwd = @relation (x₂,) where (x₀::X, x₁::X, x₂::X, z₁::Z, z₂::Z) begin
        state(x₀)
        predict(x₀, x₁)
        predict(x₁, x₂)
        measure(x₁, z₁)
        measure(x₂, z₂)
        observe₁(z₁)
        observe₂(z₂)
    end

    hom_map = Dict{Symbol, DenseGaussianSystem{Float64}}(
        :state => normal(Zeros(6), P),
        :predict => kernel(F, Zeros(6), Q),
        :measure => kernel(H, Zeros(2), R),
        :observe₁ => normal(z₁, Zeros(2, 2)),
        :observe₂ => normal(z₂, Zeros(2, 2)))

    ob_map = Dict(
        :X => 6,
        :Z => 2)

    Σ = oapply(uwd, hom_map, ob_map; ob_attr=:junction_type)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    ip = InferenceProblem(uwd, hom_map, ob_map; ob_attr=:junction_type)
    @test ip.query == [3]

    is = init(ip, MinFill())
    Σ = solve!(is)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    ip.query = []
    is = init(ip, MinDegree()); is.query = [3]
    Σ = solve!(is)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    is.query = [1, 2, 3, 4, 5]
    @test_throws ErrorException("Query not covered by join tree.") solve!(is)
end

# Example 8
# https://www.kalmanfilter.net/kalman1d_pn.html
@testset "BayesNets" begin
    x₀ = 10
    p₀ = 10000
    q = 0.15
    r = 0.01

    true_var = 0.0094
    true_mean = 50.934

    bn = BayesNet()
    push!(bn, StaticCPD(:x₀, Normal(x₀, √p₀)))
    push!(bn, LinearGaussianCPD(:x₁, [:x₀], [1], 0, √q))
    push!(bn, LinearGaussianCPD(:x₂, [:x₁], [1], 0, √q))
    push!(bn, LinearGaussianCPD(:z₁, [:x₁], [1], 0, √r))
    push!(bn, LinearGaussianCPD(:z₂, [:x₂], [1], 0, √r))

    query = [:x₂]
    evidence = Dict(:z₁ => 50.486, :z₂ => 50.963)

    ip = InferenceProblem(bn, query, evidence)
    is = init(ip, MinFill())

    Σ = solve!(is)
    @test isapprox(true_var, only(var(Σ)); atol=0.001)
    @test isapprox(true_mean, only(mean(Σ)); atol=0.001)
end
