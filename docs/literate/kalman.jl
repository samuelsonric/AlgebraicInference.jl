# # Kalman Filter
using AlgebraicInference
using BenchmarkTools
using Catlab.Graphics, Catlab.Programs, Catlab.WiringDiagrams
using Distributions
using FillArrays
using LinearAlgebra
using Random
# A Kalman filter with ``n`` steps is a probability distribution over states
# ``(s_1, \dots, s_n)`` and measurements ``(z_1, \dots, z_n)`` determined by the equations
# ```math
#     s_{i+1} \mid s_i \sim \mathcal{N}(As_i, P)
# ```
# and
# ```math
#     z_i \mid s_i \sim \mathcal{N}(Bs_i, Q).
# ```
θ = π / 15

A = [
    cos(θ) -sin(θ)
    sin(θ) cos(θ)
]

B = [
    1.3 0.0
    0.0 0.7
]

P = [
    0.05 0.0
    0.0 0.05
]

Q = [
    10.0 0.0
    0.0 10.0
]

function generate_data(n; seed=42)
    Random.seed!(seed)
    x = zeros(2)
    data = Vector{Float64}[]
    
    for i in 1:n
        x = rand(MvNormal(A * x, P))
        push!(data, rand(MvNormal(B * x, Q)))
    end

    data
end;
# The *filtering problem* involves predicting the value of the state ``s_n`` given
# observations of ``(z_1, \dots, z_n)``. The function `kalman` constructs a wiring diagram
# that represents the filtering problem.
function kalman_step(i)
    kf = HypergraphDiagram{String, String}(["X"])
    add_box!(kf, ["X"]; name="state")
    add_box!(kf, ["X", "X"]; name="predict")
    add_box!(kf, ["X", "Z"]; name="measure")
    add_box!(kf, ["Z"]; name="z$i")
    
    add_wires!(kf, [
        (0, 1) => (2, 2),
        (1, 1) => (2, 1),
        (1, 1) => (3, 1),
        (3, 2) => (4, 1)])
    
    kf
end

function kalman(n)
    reduce((kf, i) -> ocompose(kalman_step(i), 1, kf), 2:n; init=kalman_step(1))
end

to_graphviz(kalman(5), box_labels=:name; implicit_junctions=true)
# We generate ``100`` points of data and solve the filtering problem. 
n = 100; kf = kalman(n); data = generate_data(n)

evidence = Dict("z$i" => normal(data[i], Zeros(2, 2)) for i in 1:n)

hom_map = Dict(
    evidence...,
    "state" => normal(Zeros(2), 100I(2)),
    "predict" => kernel(A, Zeros(2), P),
    "measure" => kernel(B, Zeros(2), Q))

ob_map = Dict(
    "X" => 2,
    "Z" => 2)

ob_attr = :junction_type

mean(oapply(kf, hom_map, ob_map; ob_attr))
#
@benchmark oapply(kf, hom_map, ob_map; ob_attr)
# Since the filtering problem is large, we may wish to solve it using belief propagation.
T₁ = DenseGaussianSystem{Float64}
T₂ = Int
T₃ = Float64

ip = InferenceProblem{T₁, T₂, T₃}(kf, hom_map, ob_map; ob_attr)
is = init(ip, MinFill())

mean(solve(is))
#
@benchmark solve(is)
