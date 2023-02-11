using AlgebraicInference
using Catlab, Catlab.ACSetInterface, Catlab.Programs, Catlab.Theories
using LinearAlgebra: norm
using Test

import Base: ≈

const FC = FreeAbelianBicategoryRelations

function ≈(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution)
    Q₁, a₁, B₁, b₁ = params(d₁)
    Q₂, a₂, B₂, b₂ = params(d₂)

    atol = 1e-3
    return (
        Q₁ ≈ Q₂
        && isapprox(a₁, a₂; atol)
        && isapprox(B₁, B₂; atol)
        && isapprox(b₁, zero(b₁); atol) == isapprox(b₂, zero(b₂); atol)
    )
end

@testset "KalmanFilter" begin
    @testset "Directed" begin
        X = Ob(FC, :X)
        F = Hom(:F, X, X)
        Q = Hom(:Q, mzero(FC.Ob), X)
        kalman_predict = (F ⊕ Q) ⋅ plus(X)
       
        Z = Ob(FC, :Z)
        H = Hom(:H, X, Z)
        R = Hom(:R, mzero(FC.Ob), Z)
        kalman_observe = Δ(X) ⋅ (id(X) ⊕ H ⊕ R) ⋅ (id(X) ⊕ plus(Z))

        P₀ = Hom(:P_0, mzero(FC.Ob), X)
        z₁ = Hom(:z_1, mzero(FC.Ob), Z)
        z₂ = Hom(:z_2, mzero(FC.Ob), Z)
        kalman_filter = P₀
        kalman_filter = kalman_filter ⋅ kalman_predict ⋅ kalman_observe ⋅ (id(X) ⊕ dagger(z₁))
        kalman_filter = kalman_filter ⋅ kalman_predict ⋅ kalman_observe ⋅ (id(X) ⊕ dagger(z₂))

        types = (GaussDom, OpenGaussianDistribution)

        generators = Dict(
            X => GaussDom(6),
            Z => GaussDom(2),
            F => OpenGaussianDistribution([
                1   1   1/2 0   0   0
                0   1   1   0   0   0
                0   0   1   0   0   0
                0   0   0   1   1   1/2
                0   0   0   0   1   1
                0   0   0   0   0   1
            ]),
            H => OpenGaussianDistribution([
                1   0   0   0   0   0
                0   0   0   1   0   0
            ]),
            Q => OpenGaussianDistribution(GaussianDistribution([
                1/4 1/2 1/2 0   0   0
                1/2 1   1   0   0   0
                1/2 1   1   0   0   0
                0   0   0   1/4 1/2 1/2
                0   0   0   1/2 1   1
                0   0   0   1/2 1   1
            ] * 1/25)),
            R => OpenGaussianDistribution(GaussianDistribution([
                9   0
                0   9
            ])),
            P₀ => OpenGaussianDistribution(GaussianDistribution([
                500 0   0   0   0   0
                0   500 0   0   0   0
                0   0   500 0   0   0
                0   0   0   500 0   0
                0   0   0   0   500 0
                0   0   0   0   0   500
            ])),
            z₁ => OpenGaussianDistribution(GaussianDistribution([-393.66, 300.40])),
            z₂ => OpenGaussianDistribution(GaussianDistribution([-375.93, 301.78])),
        )

        d = functor(types, kalman_filter; generators)
 
        Σ = [
            8.92    11.33   5.13    0       0       0
            11.33   61.1    75.4    0       0       0
            5.13    75.4    126.5   0       0       0
            0       0       0       8.92    11.33   5.13
            0       0       0       11.33   61.1    75.4
            0       0       0       5.13    75.4    126.5
        ]

        μ = [-378.9, 53.8, 94.5, 303.9, -22.3, -63.6]
       
        rtol = 1e-3
        @test isapprox(Σ, cov(d); rtol)
        @test isapprox(μ, mean(d); rtol)
    end

    @testset "Undirected" begin
        kalman_filter = @relation (x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆) begin
            P₀(x₀₁, x₀₂, x₀₃, x₀₄, x₀₅, x₀₆)
            F(x₀₁, x₀₂, x₀₃, x₀₄, x₀₅, x₀₆, x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆)
            F(x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆, x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆)
            H(x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆, z₁₁, z₁₂)
            H(x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆, z₂₁, z₂₂)
            z₁(z₁₁, z₁₂)
            z₂(z₂₁, z₂₂)
        end

        scheduled_kalman_filter = ScheduledUntypedHypergraphDiagram()
        copy_parts!(scheduled_kalman_filter, kalman_filter)
        add_parts!(scheduled_kalman_filter, :Composite, 6; parent=[4, 4, 6, 5, 6, 6])
        set_subpart!(scheduled_kalman_filter, :box_parent, [1, 1, 5, 2, 3, 2, 3])
 
        hom_map = Dict(
            :F => OpenGaussianDistribution([
                    1   1   1/2 0   0   0
                    0   1   1   0   0   0
                    0   0   1   0   0   0
                    0   0   0   1   1   1/2
                    0   0   0   0   1   1
                    0   0   0   0   0   1
                ],
                GaussianDistribution([
                    1/4 1/2 1/2 0   0   0
                    1/2 1   1   0   0   0
                    1/2 1   1   0   0   0
                    0   0   0   1/4 1/2 1/2
                    0   0   0   1/2 1   1
                    0   0   0   1/2 1   1
                ] * 1/25),
            ),

            :H => OpenGaussianDistribution([
                    1   0   0   0   0   0
                    0   0   0   1   0   0
                ],
                GaussianDistribution([
                    9   0
                    0   9
                ]),
            ),

            :P₀ => OpenGaussianDistribution(GaussianDistribution([
                500 0   0   0   0   0
                0   500 0   0   0   0
                0   0   500 0   0   0
                0   0   0   500 0   0
                0   0   0   0   500 0
                0   0   0   0   0   500
            ])),

            :z₁ => OpenGaussianDistribution(GaussianDistribution([-393.66, 300.40])),
            :z₂ => OpenGaussianDistribution(GaussianDistribution([-375.93, 301.78])),
        )

        d₁ = oapply(kalman_filter, hom_map)
        d₂ = eval_schedule(scheduled_kalman_filter, hom_map)
       
        Σ = [
            8.92    11.33   5.13    0       0       0
            11.33   61.1    75.4    0       0       0
            5.13    75.4    126.5   0       0       0
            0       0       0       8.92    11.33   5.13
            0       0       0       11.33   61.1    75.4
            0       0       0       5.13    75.4    126.5
        ]

        μ = [-378.9, 53.8, 94.5, 303.9, -22.3, -63.6]

        rtol = 1e-3
        @test isapprox(Σ, cov(d₁); rtol)
        @test isapprox(Σ, cov(d₂); rtol)
        @test isapprox(μ, mean(d₁); rtol)
        @test isapprox(μ, mean(d₂); rtol)
    end
end

@testset "Structure" begin
    A = GaussDom(1)
    B = GaussDom(2)

    #################################################################

    @test (plus(A) ⊕ id(A)) ⋅ plus(A) ≈ (id(A) ⊕ plus(A)) ⋅ plus(A) # ⚪-as
    @test plus(A) ≈ swap(A, A) ⋅ plus(A)                            # ⚪-co
    @test (zero(A) ⊕ id(A)) ⋅ plus(A) ≈ id(A)                       # ⚪-unl
    @test Δ(A) ⋅ (Δ(A) ⊕ id(A)) ≈ Δ(A) ⋅ (id(A) ⊕ Δ(A))             # ⚫-coas
    @test Δ(A) ≈ Δ(A) ⋅ swap(A, A)                                  # ⚫-coco
    @test Δ(A) ⋅ (◊(A) ⊕ id(A)) ≈ id(A)                             # ⚫-counl
    @test (∇(A) ⊕ id(A)) ⋅ ∇(A) ≈ (id(A) ⊕ ∇(A)) ⋅ ∇(A)             # ⚫-as
    @test ∇(A) ≈ swap(A, A) ⋅ ∇(A)                                  # ⚫-co 
    @test (□(A) ⊕ id(A)) ⋅ ∇(A) ≈ id(A)                             # ⚫-unl

    #################################################################

    @test plus(A) ⋅ Δ(A) ≈ (Δ(A) ⊕ Δ(A)) ⋅ (id(A) ⊕ swap(A, A) ⊕ id(A)) ⋅ (plus(A) ⊕ plus(A))   # ⚪⚫-bi
    @test zero(A) ⋅ Δ(A) ≈ zero(A) ⊕ zero(A)                                                    # ⚪⚫-biun
    @test plus(A) ⋅ ◊(A) ≈ ◊(A) ⊕ ◊(A)                                                          # ⚫⚪-biun
    @test zero(A) ⋅ ◊(A) ≈ id(mzero(GaussDom))                                                  # ⚪⚫-bo

    #################################################################

    @test (id(A) ⊕ Δ(A)) ⋅ (∇(A) ⊕ id(A)) ≈ ∇(A) ⋅ Δ(A) # ⚫-fr1
    @test (Δ(A) ⊕ id(A)) ⋅ (id(A) ⊕ ∇(A)) ≈ ∇(A) ⋅ Δ(A) # ⚫-fr2
    @test Δ(A) ⋅ ∇(A) ≈ id(A)                           # ⚫-sp
    @test □(A) ⋅ ◊(A) ≈ id(mzero(GaussDom))             # ⚫-bo

    #################################################################

    @test (id(A) ⊕ coplus(A)) ⋅ (plus(A) ⊕ id(A)) ≈ plus(A) ⋅ coplus(A) # ⚪-fr1
    @test (coplus(A) ⊕ id(A)) ⋅ (id(A) ⊕ plus(A)) ≈ plus(A) ⋅ coplus(A) # ⚪-fr2
    @test coplus(A) ⋅ plus(A) ≈ id(A)                                   # ⚪-sp
    @test zero(A) ⋅ cozero(A) ≈ id(mzero(GaussDom))                     # ⚪-bo
end
