using AbstractTrees
using AlgebraicInference
using BayesNets
using Catlab.Programs
using Distributions
using FillArrays
using Graphs
using LinearAlgebra
using Random
using Test


@testset "Construction" begin
    @testset "GaussianSystem" begin
        d = MvNormalCanon([3, 1], [3 1; 1 2])
        Σ = GaussianSystem(d)        

        @test Σ.P == [
            3    1
            1    2
        ]

        @test Σ.p == [3, 1]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)

        d = NormalCanon(1, 2)
        Σ = GaussianSystem(d)

        @test Σ.P == [2;;]
        @test Σ.p == [1]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)

        d = MvNormal([3, 1], [3 1; 1 2])
        Σ = GaussianSystem(d)

        @test Σ.P ≈ [
            2/5 -1/5
           -1/5  3/5
        ]

        @test Σ.p ≈ [1, 0]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)

        d = Normal(1, √2)
        Σ = GaussianSystem(d)

        @test Σ.P ≈ [1/2;;]
        @test Σ.p ≈ [1/2]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)

        cpd = LinearGaussianCPD(:z, [:x, :y], [1, 2], 1, √2)
        Σ = GaussianSystem(cpd)

        @test Σ.P ≈ [
            1/2  1   -1/2
            1    2   -1
           -1/2 -1    1/2
        ]

        @test Σ.p ≈ [-1/2, -1, 1/2]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)

        cpd = StaticCPD(:x, Normal(1, √2))
        Σ = GaussianSystem(cpd)

        @test Σ.P ≈ [1/2;;]
        @test Σ.p ≈ [1/2]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)
    end

    @testset "normal" begin
        Σ = normal([3, 1], [1 1; 1 1])

        @test Σ.P ≈ [
            1/4  1/4
            1/4  1/4
        ]

        @test Σ.S ≈ [
            1/2 -1/2
           -1/2  1/2
        ]

        @test Σ.p ≈ [1,  1]
        @test Σ.s ≈ [1, -1]
        @test Σ.σ ≈ 2

        Σ = normal([3, 1], Eye(2))

        @test Σ.P == [
            1    0
            0    1
        ]

        @test Σ.p == [3, 1]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)

        Σ = normal([3, 1], Zeros(2, 2))
     
        @test Σ.S == [
            1    0
            0    1
        ]

        @test Σ.s == [3, 1]
        @test Σ.σ == 10

        @test iszero(Σ.P)
        @test iszero(Σ.p)

        Σ = normal(1, 1/2)

        @test Σ.P == [4;;]
        @test Σ.p == [4]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)
    end

    @testset "kernel" begin
        Σ = kernel([1 0; 0 1], [3, 1], [1 1; 1 1])

        @test Σ.P ≈ [
            1/4  1/4 -1/4 -1/4
            1/4  1/4 -1/4 -1/4
           -1/4 -1/4  1/4  1/4
           -1/4 -1/4  1/4  1/4
        ]

        @test Σ.S ≈ [
            1/2 -1/2 -1/2  1/2
           -1/2  1/2  1/2 -1/2
           -1/2  1/2  1/2 -1/2
            1/2 -1/2 -1/2  1/2
        ]

        @test Σ.p ≈ [-1, -1,  1,  1]
        @test Σ.s ≈ [-1,  1,  1, -1]
        @test Σ.σ ≈ 2

        Σ = kernel([1], 1, 1/2)

        @test Σ.P == [
            4    -4
           -4     4
        ]

        @test Σ.p == [-4,  4]

        @test iszero(Σ.S)
        @test iszero(Σ.s)
        @test iszero(Σ.σ)
    end
end

@testset "Algebra" begin
    P = [
        2 1
        1 2
    ]

    S = [
        0 0
        0 1
    ]

    p = [1, 1]
    s = [1, 1]
    σ = 1

    Σ = GaussianSystem(P, S, p, s, σ)

    @test Σ == Σ + zero(Σ) == zero(Σ) + Σ
    @test typeof(Σ) == typeof(zero(Σ))

    M = [
        1 2
        3 4
    ]

    N = [
        5 6
        7 8
    ]

    @test (Σ * M) * N == Σ * (M * N)
    @test zero(Σ) == zero(Σ) * M
end


@testset "Sampling" begin
    P = Symmetric([
        1 1 1
        1 2 2
        1 2 3
    ])

    S = Symmetric([
        0 0 0
        0 0 0
        0 0 1
    ])

    p = [1, 1, 1]
    s = [0, 0, 1]
    σ = 1

    Σ = GaussianSystem(P, S, p, s, σ)
    spl = sampler(Σ)

    @test mean(Σ) ≈ mean(spl)
    @test cov(Σ) ≈ cov(spl)
    @test var(Σ) ≈ var(spl)
    @test spl == sampler(GaussianSystem(spl))

    n = 100000
    rng = MersenneTwister(42)
    samples = Matrix{Float64}(undef, 3, n)

    for i in 1:n
        samples[:, i] = rand(rng, spl)
    end

    @test isapprox(mean(samples; dims=2), mean(Σ); atol=0.1)
    @test isapprox(cov(samples; dims=2), cov(Σ); atol=0.1) 
    @test isapprox(var(samples; dims=2), var(Σ); atol=0.1)
end


# Example 3.18
# Pouly and Kohlas, *Generic Inference*
# A 1
# B 2
# D 3
# E 4
# L 5
# S 6
# T 7
# X 8
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
    add_edge!(graph, 5, 7)

    order = Order(graph, CuthillMcKeeJL_RCM())
    order = Order(graph, AMDJL_AMD())
    order = Order(graph, MetisJL_ND())

    order = Order(graph, MinDegree())
    @test order == [1, 8, 7, 6, 5, 4, 3, 2]

    order = Order(graph, MinFill())
    @test order == [1, 8, 7, 3, 6, 4, 5, 2]

    ordered_graph = OrderedGraph(order, graph)
    @test vertices(ordered_graph) == 1:8
    @test nv(ordered_graph) == 8

    @test issetequal(inneighbors(ordered_graph, 1), [])
    @test issetequal(inneighbors(ordered_graph, 2), [3, 4, 6])
    @test issetequal(inneighbors(ordered_graph, 3), [])
    @test issetequal(inneighbors(ordered_graph, 4), [3, 7, 8])
    @test issetequal(inneighbors(ordered_graph, 5), [4, 6, 7])
    @test issetequal(inneighbors(ordered_graph, 6), [])
    @test issetequal(inneighbors(ordered_graph, 7), [1])
    @test issetequal(inneighbors(ordered_graph, 8), [])
 
    @test issetequal(outneighbors(ordered_graph, 1), [7])
    @test issetequal(outneighbors(ordered_graph, 2), [])
    @test issetequal(outneighbors(ordered_graph, 3), [2, 4])
    @test issetequal(outneighbors(ordered_graph, 4), [2, 5])
    @test issetequal(outneighbors(ordered_graph, 5), [])
    @test issetequal(outneighbors(ordered_graph, 6), [2, 5])
    @test issetequal(outneighbors(ordered_graph, 7), [4, 5])
    @test issetequal(outneighbors(ordered_graph, 8), [4])

    elimination_tree = EliminationTree(ordered_graph, Val(false))
    @test rootindex(elimination_tree) == 2

    @test parentindex(elimination_tree, 1) == 7
    @test parentindex(elimination_tree, 2) == nothing
    @test parentindex(elimination_tree, 3) == 4
    @test parentindex(elimination_tree, 4) == 5
    @test parentindex(elimination_tree, 5) == 2
    @test parentindex(elimination_tree, 6) == 5
    @test parentindex(elimination_tree, 7) == 4
    @test parentindex(elimination_tree, 8) == 4

    @test issetequal(childindices(elimination_tree, 1), [])
    @test issetequal(childindices(elimination_tree, 2), [5])
    @test issetequal(childindices(elimination_tree, 3), [])
    @test issetequal(childindices(elimination_tree, 4), [3, 7, 8])
    @test issetequal(childindices(elimination_tree, 5), [4, 6])
    @test issetequal(childindices(elimination_tree, 6), [])
    @test issetequal(childindices(elimination_tree, 7), [1])
    @test issetequal(childindices(elimination_tree, 8), [])

    @test issetequal(nodevalue(elimination_tree, 1), [7])
    @test issetequal(nodevalue(elimination_tree, 2), [])
    @test issetequal(nodevalue(elimination_tree, 3), [2, 4])
    @test issetequal(nodevalue(elimination_tree, 4), [2, 5])
    @test issetequal(nodevalue(elimination_tree, 5), [2])
    @test issetequal(nodevalue(elimination_tree, 6), [2, 5])
    @test issetequal(nodevalue(elimination_tree, 7), [4, 5])
    @test issetequal(nodevalue(elimination_tree, 8), [4])

    join_tree = JoinTree(elimination_tree, Node())
    @test rootindex(join_tree) == 2

    @test parentindex(join_tree, 1) == 7
    @test parentindex(join_tree, 2) == nothing
    @test parentindex(join_tree, 3) == 4
    @test parentindex(join_tree, 4) == 5
    @test parentindex(join_tree, 5) == 2
    @test parentindex(join_tree, 6) == 5
    @test parentindex(join_tree, 7) == 4
    @test parentindex(join_tree, 8) == 4

    @test issetequal(childindices(join_tree, 1), [])
    @test issetequal(childindices(join_tree, 2), [5])
    @test issetequal(childindices(join_tree, 3), [])
    @test issetequal(childindices(join_tree, 4), [3, 7, 8])
    @test issetequal(childindices(join_tree, 5), [4, 6])
    @test issetequal(childindices(join_tree, 6), [])
    @test issetequal(childindices(join_tree, 7), [1])
    @test issetequal(childindices(join_tree, 8), [])

    @test issetequal(first(nodevalue(join_tree, 1)), [7])
    @test issetequal(first(nodevalue(join_tree, 2)), [])
    @test issetequal(first(nodevalue(join_tree, 3)), [2, 4])
    @test issetequal(first(nodevalue(join_tree, 4)), [2, 5])
    @test issetequal(first(nodevalue(join_tree, 5)), [2])
    @test issetequal(first(nodevalue(join_tree, 6)), [2, 5])
    @test issetequal(first(nodevalue(join_tree, 7)), [4, 5])
    @test issetequal(first(nodevalue(join_tree, 8)), [4])

    @test issetequal(last(nodevalue(join_tree, 1)), [1])
    @test issetequal(last(nodevalue(join_tree, 2)), [2])
    @test issetequal(last(nodevalue(join_tree, 3)), [3])
    @test issetequal(last(nodevalue(join_tree, 4)), [4])
    @test issetequal(last(nodevalue(join_tree, 5)), [5])
    @test issetequal(last(nodevalue(join_tree, 6)), [6])
    @test issetequal(last(nodevalue(join_tree, 7)), [7])
    @test issetequal(last(nodevalue(join_tree, 8)), [8])

    join_tree = JoinTree(elimination_tree, MaximalSupernode())
    @test rootindex(join_tree) == 6

    @test parentindex(join_tree, 1) == 3
    @test parentindex(join_tree, 2) == 6
    @test parentindex(join_tree, 3) == 6
    @test parentindex(join_tree, 4) == 6
    @test parentindex(join_tree, 5) == 6
    @test parentindex(join_tree, 6) == nothing

    @test issetequal(childindices(join_tree, 1), [])
    @test issetequal(childindices(join_tree, 2), [])
    @test issetequal(childindices(join_tree, 3), [1])
    @test issetequal(childindices(join_tree, 4), [])
    @test issetequal(childindices(join_tree, 5), [])
    @test issetequal(childindices(join_tree, 6), [2, 3, 4, 5])

    @test issetequal(first(nodevalue(join_tree, 1)), [7])
    @test issetequal(first(nodevalue(join_tree, 2)), [4])
    @test issetequal(first(nodevalue(join_tree, 3)), [4, 5])
    @test issetequal(first(nodevalue(join_tree, 4)), [2, 4])
    @test issetequal(first(nodevalue(join_tree, 5)), [2, 5])
    @test issetequal(first(nodevalue(join_tree, 6)), [])

    @test issetequal(last(nodevalue(join_tree, 1)), [1])
    @test issetequal(last(nodevalue(join_tree, 2)), [8])
    @test issetequal(last(nodevalue(join_tree, 3)), [7])
    @test issetequal(last(nodevalue(join_tree, 4)), [3])
    @test issetequal(last(nodevalue(join_tree, 5)), [6])
    @test issetequal(last(nodevalue(join_tree, 6)), [2, 4, 5])
end


@testset "Chordal Graphs" begin
    graph = Graph(8)

    add_edge!(graph, 1, 7)
    add_edge!(graph, 2, 3)
    add_edge!(graph, 2, 4)
    add_edge!(graph, 2, 5)
    add_edge!(graph, 2, 6)
    add_edge!(graph, 3, 4)
    add_edge!(graph, 4, 5)
    add_edge!(graph, 4, 7)
    add_edge!(graph, 4, 8)
    add_edge!(graph, 5, 6)
    add_edge!(graph, 5, 7)
   
    order = Order(graph, MaxCardinality())
    @test order == [1, 3, 6, 2, 5, 7, 4, 8]

    ordered_graph = OrderedGraph(order, graph)
    elimination_tree = EliminationTree(ordered_graph, Val(true))
    @test rootindex(elimination_tree) == 8

    @test parentindex(elimination_tree, 1) == 7
    @test parentindex(elimination_tree, 2) == 5
    @test parentindex(elimination_tree, 3) == 2
    @test parentindex(elimination_tree, 4) == 8
    @test parentindex(elimination_tree, 5) == 7
    @test parentindex(elimination_tree, 6) == 2
    @test parentindex(elimination_tree, 7) == 4
    @test parentindex(elimination_tree, 8) == nothing

    @test issetequal(childindices(elimination_tree, 1), [])
    @test issetequal(childindices(elimination_tree, 2), [3, 6])
    @test issetequal(childindices(elimination_tree, 3), [])
    @test issetequal(childindices(elimination_tree, 4), [7])
    @test issetequal(childindices(elimination_tree, 5), [2])
    @test issetequal(childindices(elimination_tree, 6), [])
    @test issetequal(childindices(elimination_tree, 7), [1, 5])
    @test issetequal(childindices(elimination_tree, 8), [4])

    @test issetequal(nodevalue(elimination_tree, 1), [7])
    @test issetequal(nodevalue(elimination_tree, 2), [4, 5])
    @test issetequal(nodevalue(elimination_tree, 3), [2, 4])
    @test issetequal(nodevalue(elimination_tree, 4), [8])
    @test issetequal(nodevalue(elimination_tree, 5), [4, 7])
    @test issetequal(nodevalue(elimination_tree, 6), [2, 5])
    @test issetequal(nodevalue(elimination_tree, 7), [4])
    @test issetequal(nodevalue(elimination_tree, 8), [])
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

    diagram = @relation (x₂,) where (x₀::X, x₁::X, x₂::X, z₁::Z, z₂::Z) begin
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

    Σ = oapply(diagram, hom_map, ob_map; ob_attr=:junction_type)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    problem = InferenceProblem(diagram, hom_map, ob_map)
    @test problem.query == [:x₂]

    solver = init(problem, MinFill(), Node(), ShenoyShafer())
    @test solver.query == [:x₂]

    Σ = solve!(solver)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    solver.query = [:x₀, :x₁, :x₂, :z₁, :z₂]
    @test_throws ErrorException("Query not covered by join tree.") solve!(solver)

    solver = init(problem, MinDegree(), MaximalSupernode(), LauritzenSpiegelhalter())
    @test solver.query == [:x₂]

    Σ = solve!(solver)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    solver = init(problem, CuthillMcKeeJL_RCM(), Node(), HUGIN())
    @test solver.query == [:x₂]

    Σ = solve!(solver)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    solver = init(problem, AMDJL_AMD(), MaximalSupernode(), AncestralSampler())
    @test solver.query == [:x₂]

    Σ = solve!(solver)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    n = 100000
    rng = MersenneTwister(42)
    samples = Matrix{Float64}(undef, 6, n)

    for i in 1:n
        samples[:, i] = rand(rng, Σ)
    end

    @test isapprox(true_mean, mean(samples; dims=2); atol=0.3)
    @test isapprox(true_cov, cov(samples; dims=2); atol=1) 
end


@testset "ASIA Network" begin
    asia = [
        0.01
        0.99
    ]

    tuberculosis = [
        0.05 0.01
        0.95 0.99
    ]

    smoker = [
        0.50
        0.50
    ]

    lung = [
        0.10 0.01
        0.90 0.99
    ] 

    bronchitis = [
        0.60 0.30
        0.40 0.70
    ]

    either = [
        1.00 1.00
        0.00 0.00;;;
        1.00 0.00
        0.00 1.00;;;
    ]

    xray = [
        0.98 0.05
        0.02 0.95
    ]

    dyspnoea = [
        0.90 0.80
        0.10 0.20;;;
        0.70 0.10
        0.30 0.90;;;
    ]

    posterior = [
        0.05
        0.95
    ]

    diagram = @relation (a,) where (a::n, t::n, s::n, l::n, b::n, e::n, x::n, d::n) begin
        asia(a)
        tuberculosis(t, a)
        smoker(s)
        lung(l, s)
        bronchitis(b, s)
        either(e, t, l)
        xray(x, e)
        dyspnoea(d, e, b)
    end

    hom_map = Dict(
        :asia => asia,
        :tuberculosis => tuberculosis,
        :smoker => smoker,
        :lung => lung,
        :bronchitis => bronchitis,
        :either => either,
        :xray => xray,
        :dyspnoea => dyspnoea)

    ob_map = Dict(:n => 2)

    context = Dict(:t => 1, :s => 2, :b => 1)

    problem = InferenceProblem(diagram, hom_map, ob_map)
    problem = reduce_to_context(problem, context)

    A = solve(problem, MinFill(), Node(), ShenoyShafer())
    @test isapprox(A / sum(A), posterior; atol=0.02)

    A = solve(problem, MinFill(), Node(), LauritzenSpiegelhalter())
    @test isapprox(A / sum(A), posterior; atol=0.02)

    A = solve(problem, MinFill(), Node(), HUGIN())
    @test isapprox(A / sum(A), posterior; atol=0.02)
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

    network = BayesNet()
    push!(network, StaticCPD(:x₀, Normal(x₀, √p₀)))
    push!(network, LinearGaussianCPD(:x₁, [:x₀], [1], 0, √q))
    push!(network, LinearGaussianCPD(:x₂, [:x₁], [1], 0, √q))
    push!(network, LinearGaussianCPD(:z₁, [:x₁], [1], 0, √r))
    push!(network, LinearGaussianCPD(:z₂, [:x₂], [1], 0, √r))

    query = [:x₂]
    context = Dict(:z₁ => 50.486, :z₂ => 50.963)

    problem = InferenceProblem(network, query, context)
    @test problem.query == [:x₂]
    @test problem.context == Dict(:z₁ => [50.486], :z₂ => [50.963])

    Σ = solve(problem)    
    @test isapprox(true_var, only(var(Σ)); atol=0.001)
    @test isapprox(true_mean, only(mean(Σ)); atol=0.001)
end
