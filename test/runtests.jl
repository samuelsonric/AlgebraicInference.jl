using AlgebraicInference
using Catlab.ACSetInterface, Catlab.CategoricalAlgebra, Catlab.Graphs, Catlab.Programs,
      Catlab.Theories
using FillArrays
using LinearAlgebra
using Test

using Catlab.WiringDiagrams.MonoidalUndirectedWiringDiagrams: UntypedHypergraphDiagram

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

    P₀ = [
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

    wd = @relation (x21, x22, x23, x24, x25, x26) begin
        initial_state(x01, x02, x03, x04, x05, x06)
        predict(x01, x02, x03, x04, x05, x06, x11, x12, x13, x14, x15, x16)
        predict(x11, x12, x13, x14, x15, x16, x21, x22, x23, x24, x25, x26)
        measure(x11, x12, x13, x14, x15, x16, z11, z12)
        measure(x21, x22, x23, x24, x25, x26, z21, z22)
        observe₁(z11, z12)
        observe₂(z21, z22)
    end

    bm = Dict(
        :initial_state => normal(P₀, Zeros(6)),
        :predict => kernel(Q, Zeros(6), F),
        :measure => kernel(R, Zeros(2), H),
        :observe₁ => normal(Zeros(2, 2), z₁),
        :observe₂ => normal(Zeros(2, 2), z₂))

    Σ = oapply(wd, bm)
    @test isapprox(true_cov, cov(Σ); atol=0.2)
    @test isapprox(true_mean, mean(Σ); atol=0.2)

    T = DenseGaussianSystem{Float64}
    ip = UWDProblem{T}(wd, bm)
    @test ip.query == [:x21, :x22, :x23, :x24, :x25, :x26]

    is = init(ip, MinFill())
    Σ = solve(is)
    @test isapprox(true_cov, cov(Σ); atol=0.2)
    @test isapprox(true_mean, mean(Σ); atol=0.2)

    Σ = solve!(is)
    @test isapprox(true_cov, cov(Σ); atol=0.2)
    @test isapprox(true_mean, mean(Σ); atol=0.2)

    ip.query = []
    is = init(ip, MinWidth())
    is.query = [:x21, :x22, :x23, :x24, :x25, :x26]
    Σ = solve(is)
    @test isapprox(true_cov, cov(Σ); atol=0.2)
    @test isapprox(true_mean, mean(Σ); atol=0.2)

    Σ = solve!(is)
    @test isapprox(true_cov, cov(Σ); atol=0.2)
    @test isapprox(true_mean, mean(Σ); atol=0.2)

    is.query = [:x31]
    @test_throws ErrorException("Query not covered by join tree.") solve(is)
    @test_throws ErrorException("Query not covered by join tree.") solve!(is)

    _wd = wd; wd = UntypedHypergraphDiagram{Symbol}(); copy_parts!(wd, _wd)
    Σ = solve(UWDProblem{T}(wd, bm), MinFill()) 
    @test isapprox(true_cov, cov(Σ); atol=0.2)
    @test isapprox(true_mean, mean(Σ); atol=0.2)
end

@testset "UWDBox" begin
    _, OpenGraph = OpenCSetTypes(Graph, :V)

    g = @acset Graph begin
        V = 2
        E = 1
        src = [1]
        tgt = [2]
    end

    f = OpenGraph(g, FinFunction([1], 2), FinFunction([2], 2))
    ϕ₁ = UWDBox{OpenGraph, Symbol}(f, [:x, :y])
    ϕ₂ = UWDBox{OpenGraph, Symbol}(f, [:y, :z])

    wd = @relation (x, y, z) begin
        f(x, y)
        g(y, z)
    end

    @test combine(ϕ₁, ϕ₂).box == oapply(wd, [f, f])

    wd = @relation (x,) begin
        f(x, y)
    end

    @test project(ϕ₁, [:x]).box == oapply(wd, [f])

    wd = @relation (x,) begin
        f(x, x)
    end

    UWDBox{OpenGraph, Symbol}(f, [:x, :x], false).box == oapply(wd, [f])
end
