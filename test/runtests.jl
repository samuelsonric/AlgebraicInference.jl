using AlgebraicInference
using Catlab, Catlab.Graphs, Catlab.Programs, Catlab.Theories
using LinearAlgebra
using OrderedCollections
using Test


@testset "Gaussian Systems" begin
    # Example 9
    # https://www.kalmanfilter.net/multiExamples.html
    @testset "Kalman Filter" begin
        kalman_filter = @relation (x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆) begin
            initial_state(x₀₁, x₀₂, x₀₃, x₀₄, x₀₅, x₀₆)
            predict(x₀₁, x₀₂, x₀₃, x₀₄, x₀₅, x₀₆, x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆)
            predict(x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆, x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆)
            measure(x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆, z₁₁, z₁₂)
            measure(x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆, z₂₁, z₂₂)
            observe₁(z₁₁, z₁₂)
            observe₂(z₂₁, z₂₂)
        end

        F =  [ 1   1   1/2 0   0   0
               0   1   1   0   0   0
               0   0   1   0   0   0
               0   0   0   1   1   1/2
               0   0   0   0   1   1
               0   0   0   0   0   1 ]

        Q =  [ 1/4 1/2 1/2 0   0   0
               1/2 1   1   0   0   0
               1/2 1   1   0   0   0
               0   0   0   1/4 1/2 1/2
               0   0   0   1/2 1   1
               0   0   0   1/2 1   1 ] * 1/25

        H =  [ 1   0   0   0   0   0
               0   0   0   1   0   0 ]

        R =  [ 9   0
               0   9 ]

        P₀ = [ 500 0   0   0   0   0
               0   500 0   0   0   0
               0   0   500 0   0   0
               0   0   0   500 0   0
               0   0   0   0   500 0
               0   0   0   0   0   500 ]

        z₁ = [-393.66
               300.40 ]

        z₂ = [-375.93
               301.78 ]

        hom_map = Dict( :initial_state  => ClassicalSystem(√P₀),
                        :predict        => System([-F I], ClassicalSystem(√Q)),
                        :measure        => System([-H I], ClassicalSystem(√R)),
                        :observe₁       => ClassicalSystem(z₁),
                        :observe₂       => ClassicalSystem(z₂),                 )
        Σ = oapply(kalman_filter, hom_map)
        
        COV = [ 8.92    11.33   5.13    0       0       0
                11.33   61.1    75.4    0       0       0
                5.13    75.4    126.5   0       0       0
                0       0       0       8.92    11.33   5.13
                0       0       0       11.33   61.1    75.4
                0       0       0       5.13    75.4    126.5 ]
        @test isapprox(COV, cov(Σ); rtol=1e-3)

        MEAN = [-378.9
                 53.8
                 94.5
                 303.9
                 -22.3
                 -63.6 ]
        @test isapprox(MEAN, mean(Σ); rtol=1e-3)
    end
end

@testset "Valuations" begin
    @testset "Projection" begin
        L = [ 1 0 0
              0 2 0
              0 0 3 ]

        μ = [ 1
              2 
              3 ]

        ϕ = LabeledBox(ClassicalSystem(L, μ), OrderedSet([:a, :b, :c]))
        ψ = ϕ ↓ Set([:a, :c])
        M = [ i == j
              for i in [:a, :c],
                  j in ψ.labels  ]

        @test d(ψ) == Set([:a, :c])

        COV = [ 1 0 
                0 9 ]
        @test isapprox(COV, cov(M * ψ.box); rtol=1e-3)

        MEAN = [ 1
                 3 ]
        @test isapprox(MEAN, mean(M * ψ.box); rtol=1e-3)
    end

    @testset "Combination" begin
        L₁ = [ 1 0
               0 1 ]

        L₂ = [ 2 0
               0 2 ]

        μ₁ = [ 1
               0 ]

        μ₂ = [ 0
               1 ]

        ϕ₁ = LabeledBox(ClassicalSystem(L₁, μ₁), OrderedSet([:a, :b]))
        ϕ₂ = LabeledBox(ClassicalSystem(L₂, μ₂), OrderedSet([:b, :c]))
        ψ  = ϕ₁ ⊗ ϕ₂
        M = [ i == j
              for i in [:a, :b, :c],
                  j in ψ.labels     ]

        @test d(ψ) == Set([:a, :b, :c])

        COV = [ 1  0   0
                0  4/5 0
                0  0   4 ]
        @test isapprox(COV, cov(M * ψ.box); rtol=1e-3)

        MEAN = [ 1
                 0
                 1 ]
        @test isapprox(MEAN, mean(M * ψ.box); rtol=1e-3)
    end

    # Example 3.2
    # *Generic Inference. A Unified Theory for Automated Reasoning*
    @testset "Join Tree Construction" begin
        domains = Set([ Set([:A])
                        Set([:A, :T])
                        Set([:L, :S])
                        Set([:B, :S])
                        Set([:E, :L, :T])
                        Set([:E, :X])
                        Set([:B, :D, :E])
                        Set([:S])         ])

        elimination_sequence = OrderedSet([ :X
                                            :S
                                            :L
                                            :T
                                            :E  ])

        G, λ = join_tree_construction(domains, elimination_sequence)
        V = vertices(G)
        E = Set([ Set([src(G, i), tgt(G, i)])
                  for i in edges(G)           ]) 

        @test filter(i -> λ[i] == Set([:E, :X]),         V) |> length == 1
        @test filter(i -> λ[i] == Set([:B, :L, :S]),     V) |> length == 1
        @test filter(i -> λ[i] == Set([:B, :E, :L, :T]), V) |> length == 1
        @test filter(i -> λ[i] == Set([:A, :B, :E, :T]), V) |> length == 1
        @test filter(i -> λ[i] == Set([:A, :B, :D, :E]), V) |> length == 1
        @test filter(i -> λ[i] == Set([:A, :B, :D]),     V) |> length == 1

        i₁ = filter(i -> λ[i] == Set([:E, :X]),         V) |> first
        i₂ = filter(i -> λ[i] == Set([:B, :L, :S]),     V) |> first
        i₃ = filter(i -> λ[i] == Set([:B, :E, :L, :T]), V) |> first
        i₄ = filter(i -> λ[i] == Set([:A, :B, :E, :T]), V) |> first
        i₅ = filter(i -> λ[i] == Set([:A, :B, :D, :E]), V) |> first
        i₆ = filter(i -> λ[i] == Set([:A, :B, :D]),     V) |> first

        @test E == Set([ Set([i₂, i₃])
                         Set([i₃, i₄])
                         Set([i₄, i₅])
                         Set([i₁, i₅])
                         Set([i₅, i₆]) ])
    end
end
