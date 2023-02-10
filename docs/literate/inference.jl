# # Bayesian Inference
using AlgebraicInference
using Cairo
using Catlab, Catlab.Theories, Catlab.Graphics, Catlab.Programs
using DataFrames
using Fontconfig
using Gadfly
# The undirected wiring diagram `posterior_diagram` represents the posterior distribution
# ```math
# Y \mid X = \text{evidence}
# ```
# of a generative model
# ```math
# \begin{align*}
# X         &\sim \text{prior} \\
# Y \mid X  &\sim \text{likelihood}.
# \end{align*}
# ```
posterior_diagram = @relation (X,) begin
    prior(X)
    likelihood(X, Y)
    evidence(Y)
end

to_graphviz(posterior_diagram; box_labels = :name)
# We assign values to the nodes in `posterior_diagram`.
# ```math
# \begin{align*}
# \text{prior}          &= \mathcal{N}(0, 1) \\
# \text{likelihood}(x)  &= \mathcal{N}(2x, 0.5) \\
# \text{evidence}       &= 3
# \end{align*}
# ```
# Then we compute the posterior
# ```math
# Y \mid X = 3.
# ```
hom_map = Dict(
    :prior      => OpenGaussianDistribution(GaussianDistribution([1;;])),
    :likelihood => OpenGaussianDistribution([2;;], GaussianDistribution([.5;;])),
    :evidence   => OpenGaussianDistribution(GaussianDistribution([3])),
)

posterior = oapply(posterior_diagram, hom_map)
μ = mean(posterior)[1]
σ = √cov(posterior)[1]

(round(μ), round(σ^2))
# Finally, we plot the posterior against the prior.
pdfs = DataFrame(
    x = range(-5, 5, 100),
    y₀ = map(x -> exp(-.5x^2) / √(2π), range(-5, 5, 100)),
    y₁ = map(x -> exp(-.5(x - μ)^2 / σ^2) / σ / √(2π), range(-5, 5, 100)),
)

set_default_plot_size(23cm, 10cm)

plot(
    layer(pdfs, x=:x, y=:y₀, color=["prior"], Geom.line),
    layer(pdfs, x=:x, y=:y₁, color=["posterior"], Geom.line),
    Guide.xlabel("x"),
    Guide.ylabel(""),
)
