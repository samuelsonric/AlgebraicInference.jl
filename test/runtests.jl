using AlgebraicInference
using Catlab, Catlab.Programs
using LinearAlgebra
using Test

# Example 9
# https://www.kalmanfilter.net/multiExamples.html
@testset "KalmanFilter" begin
    kalman_filter = @relation (x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆) begin
        initial_state(x₀₁, x₀₂, x₀₃, x₀₄, x₀₅, x₀₆)
        predict(x₀₁, x₀₂, x₀₃, x₀₄, x₀₅, x₀₆, x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆)
        predict(x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆, x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆)
        measure(x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆, z₁₁, z₁₂)
        measure(x₂₁, x₂₂, x₂₃, x₂₄, x₂₅, x₂₆, z₂₁, z₂₂)
        observe₁(z₁₁, z₁₂)
        observe₂(z₂₁, z₂₂)
    end

    F = [
        1   1   1/2 0   0   0
        0   1   1   0   0   0
        0   0   1   0   0   0
        0   0   0   1   1   1/2
        0   0   0   0   1   1
        0   0   0   0   0   1
    ]

    Q = 1/25 * [
        1/4 1/2 1/2 0   0   0
        1/2 1   1   0   0   0
        1/2 1   1   0   0   0
        0   0   0   1/4 1/2 1/2
        0   0   0   1/2 1   1
        0   0   0   1/2 1   1
    ]

    H = [
        1   0   0   0   0   0
        0   0   0   1   0   0
    ]

    R = [
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

    z₁ = [-393.66, 300.40]
    z₂ = [-375.93, 301.78]

    hom_map = Dict(
        :initial_state  => ClassicalSystem(√P₀),
        :predict        => System([-F I], ClassicalSystem(√Q)),
        :measure        => System([-H I], ClassicalSystem(√R)),
        :observe₁       => ClassicalSystem(z₁),
        :observe₂       => ClassicalSystem(z₂),
    )

    Σ = oapply(kalman_filter, hom_map)
  
    Γ = [
        8.92    11.33   5.13    0       0       0
        11.33   61.1    75.4    0       0       0
        5.13    75.4    126.5   0       0       0
        0       0       0       8.92    11.33   5.13
        0       0       0       11.33   61.1    75.4
        0       0       0       5.13    75.4    126.5
    ]

    μ = [-378.9, 53.8, 94.5, 303.9, -22.3, -63.6]

    rtol = 1e-3
    @test isapprox(Γ, cov(Σ); rtol)
    @test isapprox(μ, mean(Σ); rtol)
end
