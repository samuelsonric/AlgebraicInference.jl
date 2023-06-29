using AlgebraicInference
using Catlab.CategoricalAlgebra, Catlab.Graphs, Catlab.Programs, Catlab.Theories
using FillArrays
using LinearAlgebra
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

    wd = @relation (x₂,) where (x₀::X, x₁::X, x₂::X, z₁::Z, z₂::Z) begin
        initial_state(x₀)
        predict(x₀, x₁)
        predict(x₁, x₂)
        measure(x₁, z₁)
        measure(x₂, z₂)
        observe₁(z₁)
        observe₂(z₂)
    end

    hom_map = Dict(
        :initial_state => normal(Zeros(6), P),
        :predict => kernel(F, Zeros(6), Q),
        :measure => kernel(H, Zeros(2), R),
        :observe₁ => normal(z₁, Zeros(2, 2)),
        :observe₂ => normal(z₂, Zeros(2, 2)))

    ob_map = Dict(
        :X => 6,
        :Z => 2)

    Σ = oapply(wd, hom_map, ob_map; ob_attr=:junction_type)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    T = DenseGaussianSystem{Float64}
    ip = InferenceProblem{T, Int}(wd, hom_map, ob_map; ob_attr=:junction_type)
    @test ip.query == [3]

    is = init(ip, MinFill())
    Σ = solve(is)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    Σ = solve!(is)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    ip.query = []
    is = init(ip, MinDegree()); is.query = [3]
    Σ = solve(is)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    Σ = solve!(is)
    @test isapprox(true_cov, cov(Σ); atol=0.3)
    @test isapprox(true_mean, mean(Σ); atol=0.3)

    is.query = [-1]
    @test_throws ErrorException("Query not covered by join tree.") solve(is)
    @test_throws ErrorException("Query not covered by join tree.") solve!(is)
end

@testset "Open Graph" begin
    OpenGraphOb, OpenGraph = OpenCSetTypes(Graph, :V)
    @test one(Valuation{OpenGraph}).morphism == id(munit(OpenGraphOb))

    g = @acset Graph begin
        V = 2
        E = 1
        src = [1]
        tgt = [2]
    end

    objects = [
        FinSet(1),
        FinSet(1),
        FinSet(1),
        FinSet(1),
    ]

    wd = @relation (x,) begin
        f(x, x)
    end

    f = OpenGraph(g, FinFunction([1], 2), FinFunction([2], 2))
    #ϕ = Valuation{OpenGraph}(f, [1, 1])
    #@test expand(ϕ, [1]) == oapply(wd, [f])

    wd = @relation (x, x, y, y) begin
        f(x, y)
    end

    ϕ = Valuation{OpenGraph}(f, [1, 2])
    @test expand(ϕ, [1, 1, 2, 2], objects) == oapply(wd, [f])

    wd = @relation (x,) begin
        f(x, y)
    end

    @test project(ϕ, [1], objects).morphism == oapply(wd, [f])

    wd = @relation (w, x, y, z) begin
        f(w, x)
    end

    @test extend(ϕ, [1, 2, 3, 4], objects).morphism == oapply(wd, [f], objects)

    wd = @relation (x, y, z) begin
        f₁(x, y)
        f₂(y, z)
    end
 
    ϕ₁ = ϕ
    ϕ₂ = Valuation{OpenGraph}(f, [2, 3])
    @test combine(ϕ₁, ϕ₂, objects).morphism == oapply(wd, [f, f])

    wd = @relation (w, x, z) begin
        f₁(x, y)
        f₂(y, y)
    end
    
    ip = InferenceProblem{OpenGraph, FinSet}(wd, [f, f], objects)
    @test_broken solve(ip, MinFill()) == oapply(wd, [f, f], objects)
end
