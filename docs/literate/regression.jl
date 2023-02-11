# # Bayesian Linear Regression
using AlgebraicInference
using Cairo
using Catlab, Catlab.Theories, Catlab.Graphics, Catlab.Programs
using Convex
using DataFrames
using Fontconfig
using Gadfly
using SCS
using TikzPictures
# The diagram `posterior_diagram` represents the posterior weights of a Bayesian linear regression model.
const F = FreeAbelianBicategoryRelations

W  = Ob(F, :W)
Y  = Ob(F, :Y)
w  = Hom(:w, mzero(F.Ob), W)
ϵ  = Hom(Symbol("\\epsilon"), mzero(F.Ob), Y)
y  = Hom(:y, mzero(F.Ob), Y)
Φ  = Hom(Symbol("\\Phi"), W, Y)

posterior_diagram = (
    w
    ⋅ Δ(W)
    ⋅ (y ⊕ ϵ ⊕ Φ ⊕ id(W))
    ⋅ (id(Y) ⊕ plus(Y) ⊕ id(W))
    ⋅ (dcounit(Y) ⊕ id(W))
)

to_tikz(posterior_diagram;
    orientation=LeftToRight,
    base_unit="6mm",
    props=["font=\\Large", "semithick"]
)
# We assign values to the boxes in `posterior_diagram`.
# ```math
# \begin{align*}
# w         &= \mathcal{N}(0, 0.5I) \\
# \epsilon  &= \mathcal{N}(0, 0.04I) \\
# \Phi(w)   &= \begin{bmatrix} \phi(0.76) & \phi(0.50) & \phi(0.93) & \phi(0.38) \end{bmatrix}^\text{T} w \\
# y         &= (-1.03, -0.02, -0.31, 0.51),
# \end{align*}
# ```
# where ``\phi`` is a vector of nine Gaussian basis functions and ``y`` is a vector of noisy measurements of the function
# ```math
# f(x) = \sin (2 \pi x).
# ```
# Then we compute the posterior weights.
parameters = [
    (0.0, 0.1),
    (0.5, 0.1),
    (1.0, 0.1),
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5),
    (0.0, 2.5),
    (0.5, 2.5),
    (1.0, 2.5),
]

ϕ(x) = [exp(-(x - μ)^2 / σ^2 / 2) for (μ, σ) in parameters]

ob_map = Dict(
    W => GaussDom(9),
    Y => GaussDom(4),
)

hom_map = Dict(
    w => OpenGaussianDistribution(GaussianDistribution(
        [
            .5  0   0   0   0   0   0   0   0
            0   .5  0   0   0   0   0   0   0
            0   0   .5  0   0   0   0   0   0
            0   0   0   .5  0   0   0   0   0
            0   0   0   0   .5  0   0   0   0
            0   0   0   0   0   .5  0   0   0
            0   0   0   0   0   0   .5  0   0
            0   0   0   0   0   0   0   .5  0
            0   0   0   0   0   0   0   0   .5
        ],
    )),
    ϵ => OpenGaussianDistribution(GaussianDistribution(
        [
            .04 0   0   0
            0   .04 0   0
            0   0   .04 0
            0   0   0   .04
        ],
    )),
    y => OpenGaussianDistribution(GaussianDistribution(
        [-1.03, -0.02, -0.31, 0.51], 
    )),
    Φ => OpenGaussianDistribution(
        hcat(map(ϕ, [0.76, 0.50, 0.93, 0.38])...)',
    ),
)

posterior = functor(
    (GaussDom, OpenGaussianDistribution),
    posterior_diagram;
    generators = merge(ob_map, hom_map),
)

round.(mean(posterior); digits=2)
#
round.(cov(posterior); digits=2)
# Finally, we plot estimated values against true values and measurements.
measurements = DataFrame(
    x = [0.76, 0.50, 0.93, 0.38],
    y = [-1.03, -0.02, -0.31, 0.51],
)

ys = DataFrame(
    map(range(0, 1, 100)) do x
        ob_map = Dict(
            W => GaussDom(9),
            Y => GaussDom(1),
        )
        
        hom_map = Dict(
            w => posterior,
            ϵ => OpenGaussianDistribution(GaussianDistribution([.04;;])),
            Φ => OpenGaussianDistribution([ϕ(x)...;;]),
        )

        estimate = functor(
            (GaussDom, OpenGaussianDistribution),
            w ⋅ (ϵ ⊕ Φ) ⋅ plus(Y);
            generators = merge(ob_map, hom_map),
        )
        
        μ = mean(estimate)[1]
        σ = √cov(estimate)[1]
        (x=x, y_true=sin(2π * x), y_pred=μ, y_min=μ-2σ, y_max=μ+2σ)
    end
)

set_default_plot_size(23cm, 10cm)

plot(
    layer(measurements, x=:x, y=:y, color=["measurements"], Geom.point),
    layer(ys, x=:x, y=:y_true, color=["true values"], Geom.line),
    layer(ys, x=:x, y=:y_pred, ymin=:y_min, ymax=:y_max, color=["estimates"], Geom.line, Geom.ribbon),
    Coord.cartesian(xmin=0, xmax=1, ymin=-2, ymax=2),
    Guide.xlabel("x"),
    Guide.ylabel("y"),
)
