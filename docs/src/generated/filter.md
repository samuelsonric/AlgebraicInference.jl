```@meta
EditURL = "<unknown>/literate/filter.jl"
```

# Kalman Filter

````@example filter
using AlgebraicInference
using Cairo
using Catlab, Catlab.Theories, Catlab.Graphics, Catlab.Programs
using DataFrames
using Fontconfig
using Gadfly
using LinearAlgebra: diag
````

The undirected wiring diagram `filter_diagram` represents five iterations of a one-dimensional Kalman filter.

````@example filter
filter_diagram = @relation (X₁, X₂, X₃, X₄, X₅) begin
    s₀(X₀)
    y₁(Y₁)
    y₂(Y₂)
    y₃(Y₃)
    y₄(Y₄)
    y₅(Y₅)
    s(X₀, X₁)
    s(X₁, X₂)
    s(X₂, X₃)
    s(X₃, X₄)
    s(X₄, X₅)
    m(X₁, Y₁)
    m(X₂, Y₂)
    m(X₃, Y₃)
    m(X₄, Y₄)
    m(X₅, Y₅)
end

to_graphviz(filter_diagram;
    box_labels = :name,
    graph_attrs = Dict(:size => "6!"),
)
````

We assign values to the boxes in `filter_diagram`,
```math
\begin{align*}
s_0   &= \mathcal{N}(10, 10000) \\
s(x)  &= \mathcal{N}(x, 0.15) \\
m(x)  &= \mathcal{N}(x, 0.01) \\
y_1   &= 50.486 \\
y_2   &= 50.963 \\
y_3   &= 51.597 \\
y_4   &= 52.001 \\
y_5   &= 52.518,
\end{align*}
```
where ``(y_1, \dots, y_5)`` are noisy measurements of the points
```math
(50.505, 50.994, 51.493, 52.001, 52.506),
```
and estimate the state at times ``1 \leq t \leq 5``.

````@example filter
hom_map = Dict(
    :s₀  => OpenGaussianDistribution(GaussianDistribution([10000;;], [10])),
    :y₁  => OpenGaussianDistribution(GaussianDistribution([0;;], [50.486])),
    :y₂  => OpenGaussianDistribution(GaussianDistribution([0;;], [50.963])),
    :y₃  => OpenGaussianDistribution(GaussianDistribution([0;;], [51.597])),
    :y₄  => OpenGaussianDistribution(GaussianDistribution([0;;], [52.001])),
    :y₅  => OpenGaussianDistribution(GaussianDistribution([0;;], [52.518])),
    :s   => OpenGaussianDistribution([1;;], GaussianDistribution([0.15;;], [0])),
    :m   => OpenGaussianDistribution([1;;], GaussianDistribution([0.01;;], [0])),
)

state = oapply(filter_diagram, hom_map)
μ = mean(state)
Σ = cov(state)

round.(μ; digits=2)
````

````@example filter
round.(Σ; digits=2)
````

Finally, we plot estimates against measurements and true values.

````@example filter
ys = DataFrame(Dict(
    :x => 1:5,
    :y_meas => [50.486, 50.963, 51.597, 52.001, 52.518],
    :y_true => [50.505, 50.994, 51.493, 52.001, 52.506],
    :y_pred => μ,
    :y_max  => μ + 2(√).(diag(Σ)),
    :y_min  => μ - 2(√).(diag(Σ)),
))

set_default_plot_size(23cm, 10cm)

plot(
    layer(ys, x=:x, y=:y_meas, color=["Measurement"], Geom.point),
    layer(ys, x=:x, y=:y_true, color=["True Value"], Geom.line),
    layer(ys, x=:x, y=:y_pred, color=["Estimate"], Geom.line),
    layer(ys, x=:x, ymin=:y_min, ymax=:y_max, color=["95% Confidence Interval"], Geom.ribbon),
    Guide.xlabel("time"),
    Guide.ylabel("y"),
    Coord.cartesian(xmin=1),
)
````

