using AlgebraicInference
using Catlab, Catlab.Theories
using Test

@testset "KalmanFilter" begin
    O = mzero(FreeAbelianBicategoryRelations.Ob)
    X = Ob(FreeAbelianBicategoryRelations, :X)
    Q = Hom(:Q, O, X)
    F = Hom(:F, X, X)
    kalman_predict = (F ⊕ Q) ⋅ plus(X)
   

    Z = Ob(FreeAbelianBicategoryRelations, :Z)
    R = Hom(:R, O, Z)
    H = Hom(:H, X, Z)
    kalman_observe = Δ(X) ⋅ (id(X) ⊕ H ⊕ R) ⋅ (id(X) ⊕ plus(Z))

    P₀ = Hom(:P₀, O, X)
    z₁ = Hom(:z_1, O, Z)
    z₂ = Hom(:z_2, O, Z)
    
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

    P₂, x₂, _... = params(functor(types, kalman_filter; generators))
    rtol = 1e-3
    
    @test isapprox([
        8.92    11.33   5.13    0       0       0
        11.33   61.1    75.4    0       0       0
        5.13    75.4    126.5   0       0       0
        0       0       0       8.92    11.33   5.13
        0       0       0       11.33   61.1    75.4
        0       0       0       5.13    75.4    126.5
    ], P₂; rtol)

    @test isapprox([-378.9, 53.8, 94.5, 303.9, -22.3, -63.6], x₂; rtol)
end

