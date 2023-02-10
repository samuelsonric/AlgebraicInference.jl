# # Kalman Filter -- Undirected
using AlgebraicInference
using Cairo
using Catlab, Catlab.Theories, Catlab.Graphics, Catlab.Programs
using DataFrames
using Fontconfig
using Gadfly
# The undirected wiring diagram `filter_diagram` represents estimated state after five iterations of a one-dimensional Kalman filter.
filter_diagram = @relation (X₅,) begin
    s₀(X₀)
    y₁(Y₁)
    y₂(Y₂)
    y₃(Y₃)
    y₄(Y₄)
    s(X₀, X₁)
    s(X₁, X₂)
    s(X₂, X₃)
    s(X₃, X₄)
    s(X₄, X₅)
    m(X₁, Y₁)
    m(X₂, Y₂)
    m(X₃, Y₃)
    m(X₄, Y₄)
end

to_graphviz(filter_diagram; box_labels = :name)
# We assign values to the boxes in `filter_diagram` according to Example 6 of [this tutorial](https://www.kalmanfilter.net/kalman1d_pn.html).
# ```math
# \begin{align*}
# s_0   &= \mathcal{N}(60, 10000) \\
# s(x)  &= \mathcal{N}(x, 0.0001) \\
# m(x)  &= \mathcal{N}(x, 0.01) \\
# y_1   &= 49.986 \\
# y_2   &= 49.963 \\
# y_3   &= 50.090 \\
# y_4   &= 50.001 \\
# \end{align*}
# ```
# where ``(y_1, \dots, y_5)`` are noisy measurements of the points
# ```math
# (50.005, 49.994, 49.993, 50.001).
# ```
# Then we estimate the state at time ``t = 5``.
hom_map = Dict(
    :s₀  => OpenGaussianDistribution(GaussianDistribution([10000;;], [60])),
    :y₁  => OpenGaussianDistribution(GaussianDistribution([49.986])),
    :y₂  => OpenGaussianDistribution(GaussianDistribution([49.963])),
    :y₃  => OpenGaussianDistribution(GaussianDistribution([50.090])),
    :y₄  => OpenGaussianDistribution(GaussianDistribution([50.001])),
    :s   => OpenGaussianDistribution([1;;], GaussianDistribution([0.0001;;])),
    :m   => OpenGaussianDistribution([1;;], GaussianDistribution([0.01;;])),
)

state = oapply(filter_diagram, hom_map)
μ = mean(state)[1]
σ = √cov(state)[1]

print("N($μ, $σ)")
# Finally, we plot our estimate against the true value at time ``t = 5``.
pdf = DataFrame(
    x = range(49.6, 50.4, 100),
    y = map(x -> exp(-.5(x - μ)^2 / σ^2) / σ / √(2π), range(49.6, 50.4, 100)),
)

set_default_plot_size(23cm, 10cm)

plot(
    layer(pdf, x=:x, y=:y, color=["estimate"], Geom.line),
    layer(xintercept=[50.006], color=["true value"], Geom.vline),
    Guide.xlabel("Temperature (Cᵒ)"),
)
