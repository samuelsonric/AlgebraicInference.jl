# # Kalman Filter
using AlgebraicInference
using BenchmarkTools
using Catlab, Catlab.Graphics, Catlab.Programs, Catlab.WiringDiagrams
using Catlab.WiringDiagrams.MonoidalUndirectedWiringDiagrams: UntypedHypergraphDiagram
using Distributions
using GraphPlot
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
    kf = UntypedHypergraphDiagram{Symbol}(2)
    add_box!(kf, 2; name=:state)
    add_box!(kf, 4; name=:predict)
    add_box!(kf, 4; name=:measure)
    add_box!(kf, 2; name=Symbol("z$i"))
    
    add_wires!(kf, [
        (0, 1) => (2, 3),
        (0, 2) => (2, 4),
        (1, 1) => (2, 1),
        (1, 1) => (3, 1),
        (1, 2) => (2, 2),
        (1, 2) => (3, 2),
        (3, 3) => (4, 1),
        (3, 4) => (4, 2)])
    
    kf
end

function kalman(n)
    reduce((kf, i) -> ocompose(kalman_step(i), 1, kf), 2:n; init=kalman_step(1))
end

to_graphviz(kalman(5), box_labels=:name)
# We generate ``100`` points of data and solve the filtering problem. 
n = 100; kf = kalman(n); data = generate_data(n)

box_map = Dict(
    :state => normal(100I(2)),
    :predict => kernel(P, A),
    :measure => kernel(Q, B))

for i in 1:n
    box_map[Symbol("z$i")] = normal(data[i])
end

mean(oapply(kf, box_map))
#
@benchmark oapply(kf, box_map)
# Although we can solve the filtering problem using `oapply`, it is more efficient to use
# variable elimination. The function `inference_problem` turns the wiring diagram `kf` into
# an undirected graphical model.
kb, query = inference_problem(kf, box_map)
pg = primal_graph(kb)

gplot(pg)
# We compute a variable elimination order using the "min-fill" heuristic.
order = minfill!(pg, query)
jt = architecture(kb, order)

mean(answer_query(jt, query).box)
#
@benchmark answer_query(jt, query)
