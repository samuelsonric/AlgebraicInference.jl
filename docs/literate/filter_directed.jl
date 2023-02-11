# # Kalman Filter -- Directed
using AlgebraicInference
using Cairo
using Catlab, Catlab.Theories, Catlab.Graphics, Catlab.Programs
using Convex
using DataFrames
using Fontconfig
using Gadfly
using SCS
using TikzPictures
# The directed wiring diagram `filter_diagram` represents estimated state after five iterations of a one-dimensional Kalman filter.

const F = FreeAbelianBicategoryRelations

ℝ = Ob(F, :ℝ)
s₀ = Hom(:s_0, mzero(F.Ob), ℝ)
y₁ = Hom(:y_1, mzero(F.Ob), ℝ)
y₂ = Hom(:y_2, mzero(F.Ob), ℝ)
y₃ = Hom(:y_3, mzero(F.Ob), ℝ)
y₄ = Hom(:y_4, mzero(F.Ob), ℝ)
q = Hom(:q, mzero(F.Ob), ℝ)
r = Hom(:r, mzero(F.Ob), ℝ)

circuitry = (
    (plus(ℝ) ⋅ Δ(ℝ) ⊕ id(ℝ) ⊕ id(ℝ))
    ⋅ (id(ℝ) ⊕ plus(ℝ) ⊕ id(ℝ))
    ⋅ (id(ℝ) ⊕ dcounit(ℝ))
)

filter_diagram = (
    s₀
    ⋅ (id(ℝ) ⊕ q ⊕ r ⊕ y₁) ⋅ circuitry
    ⋅ (id(ℝ) ⊕ q ⊕ r ⊕ y₂) ⋅ circuitry
    ⋅ (id(ℝ) ⊕ q ⊕ r ⊕ y₃) ⋅ circuitry
    ⋅ (id(ℝ) ⊕ q ⊕ r ⊕ y₄) ⋅ circuitry
    ⋅ (id(ℝ) ⊕ q) ⋅ plus(ℝ)
)

to_tikz(filter_diagram;
    orientation=TopToBottom,
    base_unit="6mm",
    props=["font=\\Large", "semithick"],
)
# We assign values to the boxes in `filter_diagram` according to Example 6 of [this tutorial](https://www.kalmanfilter.net/kalman1d_pn.html).
# ```math
# \begin{align*}
# s_0   &= \mathcal{N}(60, 10000) \\
# q     &= \mathcal{N}(0, 0.0001) \\
# r     &= \mathcal{N}(0, 0.01) \\
# y_1   &= 49.986 \\
# y_2   &= 49.963 \\
# y_3   &= 50.090 \\
# y_4   &= 50.001,
# \end{align*}
# ```
# where ``(y_1, \dots, y_4)`` are noisy measurements of the points
# ```math
# (50.005, 49.994, 49.993, 50.001).
# ```
# Then we estimate the state at time ``t = 5``.
ob_map = Dict(
    ℝ   => GaussDom(1),
)

hom_map = Dict(
    s₀  => OpenGaussianDistribution(GaussianDistribution([10000;;], [60])),
    y₁  => OpenGaussianDistribution(GaussianDistribution([49.986])),
    y₂  => OpenGaussianDistribution(GaussianDistribution([49.963])),
    y₃  => OpenGaussianDistribution(GaussianDistribution([50.090])),
    y₄  => OpenGaussianDistribution(GaussianDistribution([50.001])),
    q   => OpenGaussianDistribution(GaussianDistribution([0.0001;;])),
    r   => OpenGaussianDistribution(GaussianDistribution([0.01;;])),
)

state = functor(
    (GaussDom, OpenGaussianDistribution),
    filter_diagram;
    generators = merge(ob_map, hom_map),
)

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
