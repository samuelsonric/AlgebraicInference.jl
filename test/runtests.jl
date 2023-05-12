using AlgebraicInference
using Catlab, Catlab.Programs, Catlab.Theories
using LinearAlgebra
using Test

# Example 9
# https://www.kalmanfilter.net/multiExamples.html
@testset "Kalman Filter" begin
    F =  [1   1   1/2 0   0   0
          0   1   1   0   0   0
          0   0   1   0   0   0
          0   0   0   1   1   1/2
          0   0   0   0   1   1
          0   0   0   0   0   1]

    Q =  [1/4 1/2 1/2 0   0   0
          1/2 1   1   0   0   0
          1/2 1   1   0   0   0
          0   0   0   1/4 1/2 1/2
          0   0   0   1/2 1   1
          0   0   0   1/2 1   1] * 1/25

    H =  [1   0   0   0   0   0
          0   0   0   1   0   0]

    R =  [9   0
          0   9]

    P₀ = [500 0   0   0   0   0
          0   500 0   0   0   0
          0   0   500 0   0   0
          0   0   0   500 0   0
          0   0   0   0   500 0
          0   0   0   0   0   500]

    z₁ = [-393.66
           300.40]

    z₂ = [-375.93
           301.78]

    true_cov = [8.92    11.33   5.13    0       0       0
                11.33   61.1    75.4    0       0       0
                5.13    75.4    126.5   0       0       0
                0       0       0       8.92    11.33   5.13
                0       0       0       11.33   61.1    75.4
                0       0       0       5.13    75.4    126.5]

    true_mean = [-378.9
                  53.8
                  94.5
                  303.9
                 -22.3
                 -63.6]

    composite = @relation (x21, x22, x23, x24, x25, x26) begin
        initial_state(x01, x02, x03, x04, x05, x06)
        predict(x01, x02, x03, x04, x05, x06, x11, x12, x13, x14, x15, x16)
        predict(x11, x12, x13, x14, x15, x16, x21, x22, x23, x24, x25, x26)
        measure(x11, x12, x13, x14, x15, x16, z11, z12)
        measure(x21, x22, x23, x24, x25, x26, z21, z22)
        observe₁(z11, z12)
        observe₂(z21, z22)
    end
    box_map = Dict(:initial_state => ClassicalSystem(P₀),
                   :predict => System([-F I], ClassicalSystem(Q)),
                   :measure => System([-H I], ClassicalSystem(R)),
                   :observe₁ => ClassicalSystem(z₁),
                   :observe₂ => ClassicalSystem(z₂))
    Σ = oapply(composite, box_map)
    @test isapprox(true_cov, cov(Σ); rtol=1e-3)
    @test isapprox(true_mean, mean(Σ); rtol=1e-3)

    knowledge_base, query = construct_inference_problem(AbstractSystem, composite, box_map) 
    domains = Set(domain(ϕ) for ϕ in knowledge_base)
    elimination_sequence = construct_elimination_sequence(domains, query)
    ϕ = fusion_algorithm(knowledge_base, elimination_sequence)
    M = [i == j.id
         for i in 1:6,
             j in ϕ.labels]
    @test Set(X.id for X in domain(ϕ)) == Set(1:6)
    @test isapprox(true_cov, cov(M * ϕ.box); rtol=1e-3)
    @test isapprox(true_mean, mean(M * ϕ.box); rtol=1e-3)

    #=
    join_tree = construct_join_tree(domains, elimination_sequence)
    assignment_map = [findfirst(join_tree.labels) do x; domain(ϕ) ⊆ x end
                      for ϕ in knowledge_base]
    join_tree_factors = construct_join_tree_factors(knowledge_base,
                                                    join_tree,
                                                    assignment_map)
    ϕ = collect_algorithm(join_tree_factors, join_tree, query)
    M = [i == j
         for i in 1:6,
             j in ϕ.labels]
    @test Set(X.value for X in domain(ϕ)) == Set(1:6)
    @test isapprox(true_cov, cov(M * ϕ.box); rtol=1e-3)
    @test isapprox(true_mean, mean(M * ϕ.box); rtol=1e-3)
    =#
end
