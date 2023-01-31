using AlgebraicInference
using Catlab, Catlab.Theories
using LinearAlgebra: norm
using Test

import Base: ≈

const FC = FreeAbelianBicategoryRelations

function ≈(F₁::QuadraticBifunction, F₂::QuadraticBifunction)
    Q₁, a₁, α₁, B₁, b₁ = conjugate(F₁)
    Q₂, a₂, α₂, B₂, b₂ = conjugate(F₂)

    atol = 1e-3
    return (
        Q₁ ≈ Q₂
        && isapprox(a₁, a₂; atol)
        && isapprox(α₁, α₂; atol)
        && isapprox(B₁, B₂; atol)
        && isapprox(b₁, zero(b₁); atol) == isapprox(b₂, zero(b₂); atol)
    )
end

@testset "KalmanFilter" begin
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

    types = (GaussRelDom, GaussianRelation)

    generators = Dict(
        X => GaussRelDom(6),
        Z => GaussRelDom(2),
        F => GaussianRelation([
            1   1   1/2 0   0   0
            0   1   1   0   0   0
            0   0   1   0   0   0
            0   0   0   1   1   1/2
            0   0   0   0   1   1
            0   0   0   0   0   1
        ]),
        H => GaussianRelation([
            1   0   0   0   0   0
            0   0   0   1   0   0
        ]),
        Q => GaussianRelation(GaussianDistribution([
            1/4 1/2 1/2 0   0   0
            1/2 1   1   0   0   0
            1/2 1   1   0   0   0
            0   0   0   1/4 1/2 1/2
            0   0   0   1/2 1   1
            0   0   0   1/2 1   1
        ] * 1/25)),
        R => GaussianRelation(GaussianDistribution([
            9   0
            0   9
        ])),
        P₀ => GaussianRelation(GaussianDistribution([
            500 0   0   0   0   0
            0   500 0   0   0   0
            0   0   500 0   0   0
            0   0   0   500 0   0
            0   0   0   0   500 0
            0   0   0   0   0   500
        ])),
        z₁ => GaussianRelation(GaussianDistribution([-393.66, 300.40])),
        z₂ => GaussianRelation(GaussianDistribution([-375.93, 301.78])),
    )

    d = functor(types, kalman_filter; generators)
    rtol = 1e-3
    
    @test isapprox([
        8.92    11.33   5.13    0       0       0
        11.33   61.1    75.4    0       0       0
        5.13    75.4    126.5   0       0       0
        0       0       0       8.92    11.33   5.13
        0       0       0       11.33   61.1    75.4
        0       0       0       5.13    75.4    126.5
    ], cov(d); rtol)

    @test isapprox([-378.9, 53.8, 94.5, 303.9, -22.3, -63.6], mean(d); rtol)
end

@testset "Structure" begin
    A = QuadDom(1)
    B = QuadDom(2)

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
    @test zero(A) ⋅ ◊(A) ≈ id(mzero(QuadDom))                                                   # ⚪⚫-bo

    #################################################################

    @test (id(A) ⊕ Δ(A)) ⋅ (∇(A) ⊕ id(A)) ≈ ∇(A) ⋅ Δ(A) # ⚫-fr1
    @test (Δ(A) ⊕ id(A)) ⋅ (id(A) ⊕ ∇(A)) ≈ ∇(A) ⋅ Δ(A) # ⚫-fr2
    @test Δ(A) ⋅ ∇(A) ≈ id(A)                           # ⚫-sp
    @test □(A) ⋅ ◊(A) ≈ id(mzero(QuadDom))              # ⚫-bo

    #################################################################

    @test (id(A) ⊕ coplus(A)) ⋅ (plus(A) ⊕ id(A)) ≈ plus(A) ⋅ coplus(A) # ⚪-fr1
    @test (coplus(A) ⊕ id(A)) ⋅ (id(A) ⊕ plus(A)) ≈ plus(A) ⋅ coplus(A) # ⚪-fr2
    @test coplus(A) ⋅ plus(A) ≈ id(A)                                   # ⚪-sp
    @test zero(A) ⋅ cozero(A) ≈ id(mzero(QuadDom))                      # ⚪-bo
end
