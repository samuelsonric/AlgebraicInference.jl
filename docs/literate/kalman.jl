# # Kalman Filter
using AlgebraicInference
using Catlab.Graphics, Catlab.Programs, Catlab.WiringDiagrams
using Distributions
using FillArrays
using LinearAlgebra
using Random
# A Kalman filter with ``n`` steps is a probability distribution over states
# ``(x_1, \dots, x_n)`` and measurements ``(y_1, \dots, y_n)`` determined by the equations
# ```math
#     p(x_{i+1} \mid x_i) = \mathcal{N}(Ax_i, P)
# ```
# and
# ```math
#     p(y_i \mid x_i) = \mathcal{N}(Bx_i, Q).
# ```
θ = π / 15

A = [
    cos(θ) -sin(θ)
    sin(θ)  cos(θ)
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


function generate_data(n::Integer; seed::Integer=42)
    Random.seed!(seed)
    data = Matrix{Float64}(undef, 2, n)

    N₁ = MvNormal(P)
    N₂ = MvNormal(Q)

    x = Zeros(2)

    for i in 1:n
        x = rand(N₁) + A * x
        data[:, i] .= rand(N₂) + B * x
    end

    data
end;
# The *prediction problem* involves finding the posterior mean and covariance of the state
# ``x_{n + 1}`` given observations of ``(y_1, \dots, y_n)``.
function make_diagram(n::Integer)
    outer_ports = ["X"]

    uwd = TypedRelationDiagram{String, String, Tuple{Int, Int}}(outer_ports)

    x = add_junction!(uwd, "X"; variable=(1, 1))
    y = add_junction!(uwd, "Y"; variable=(2, 1))

    state   = add_box!(uwd, ["X"];      name="state")
    predict = add_box!(uwd, ["X", "X"]; name="predict")
    measure = add_box!(uwd, ["X", "Y"]; name="measure")
    context = add_box!(uwd, ["Y"];      name="y1")

    set_junction!(uwd, (state,   1), x)
    set_junction!(uwd, (predict, 1), x)
    set_junction!(uwd, (measure, 1), x)
    set_junction!(uwd, (measure, 2), y)
    set_junction!(uwd, (context, 1), y)

    for i in 2:n
        x = add_junction!(uwd, "X"; variable=(1, i))
        y = add_junction!(uwd, "Y"; variable=(2, i))

        set_junction!(uwd, (predict, 2), x)

        predict = add_box!(uwd, ["X", "X"]; name="predict")
        measure = add_box!(uwd, ["X", "Y"]; name="measure")
        context = add_box!(uwd, ["Y"];      name="y$i")

        set_junction!(uwd, (predict, 1), x)
        set_junction!(uwd, (measure, 1), x)
        set_junction!(uwd, (measure, 2), y)
        set_junction!(uwd, (context, 1), y)
    end

    i = n + 1
    x = add_junction!(uwd, "X"; variable=(1, i))

    set_junction!(uwd, (0,       1), x)
    set_junction!(uwd, (predict, 2), x)

    uwd
end

to_graphviz(make_diagram(5), box_labels=:name; junction_labels=:variable)
# We generate ``100`` points of data and solve the prediction problem.
n = 100
diagram = make_diagram(n)
data = generate_data(n)

hom_map = Dict{String, DenseGaussianSystem{Float64}}(
    "state" => normal(Zeros(2), 100I(2)),
    "predict" => kernel(A, Zeros(2), P),
    "measure" => kernel(B, Zeros(2), Q))

ob_map = Dict(
    "X" => 2,
    "Y" => 2)

for i in 1:n
    hom_map["y$i"] = normal(data[:, i], Zeros(2, 2))
end

problem = InferenceProblem(diagram, hom_map, ob_map)

Σ = solve(problem)

round.(mean(Σ); digits=4)
#
round.(cov(Σ); digits=4)
