# # Bayesian Linear Regression
using AlgebraicInference
using Cairo
using Catlab, Catlab.Theories, Catlab.Graphics, Catlab.Programs
using Convex
using DataFrames
using Fontconfig
using Gadfly
using LinearAlgebra: I
using SCS
using TikzPictures
# The diagram `update_diagram` represents the posterior distribution of a Bayesian linear regression model.
const F = FreeAbelianBicategoryRelations

input_space  = Ob(F, :input_space)
output_space = Ob(F, :output_space)
input        = Hom(:input, input_space, output_space)
output       = Hom(:output, mzero(F.Ob), output_space)
error        = Hom(:error, mzero(F.Ob), output_space)
weights      = Hom(:weights, mzero(F.Ob), input_space)

update_diagram = (
    weights
    ⋅ Δ(input_space)
    ⋅ (error ⊕ input ⊕ id(input_space))
    ⋅ (plus(output_space) ⊕ id(input_space))
    ⋅ (output ⊕ id(output_space) ⊕ id(input_space))
    ⋅ (dcounit(output_space) ⊕ id(input_space))
)

to_tikz(update_diagram; base_unit="8mm")
# The diagram `predict_diagram` represents the predictive distribution of a Bayesian linear regression model.
predict_diagram = (
    weights
    ⋅ (error ⊕ input)
    ⋅ plus(output_space)
)

to_tikz(predict_diagram; base_unit="8mm")
# Inputs are transformed by nine Gaussian basis functions.
gbf(μ, s, x) = exp(-(x - μ)^2 / s^2 / 2)
ϕ(x) = [gbf(0.0, 0.1, x) gbf(0.5, 0.1, x) gbf(1.0, 0.1, x) gbf(0.0, 0.5, x) gbf(0.5, 0.5, x) gbf(1.0, 0.5, x) gbf(0.0, 2.5, x) gbf(0.5, 2.5, x) gbf(1.0, 2.5, x)];
# The error term is a centered Gaussian distribution with variance ``0.04``.
"""
    update(w, x, y)

Update weights `w` given input `x` and output `y`.
"""
function update(w, x, y)
    types = (GaussRelDom, GaussianRelation)

    generators = Dict(
        weights      => w,
        input_space  => GaussRelDom(9),
        output_space => GaussRelDom(1),
        input        => GaussianRelation(ϕ(x)),
        output       => GaussianRelation(GaussianDistribution([y])),
        error        => GaussianRelation(GaussianDistribution(Matrix(0.04I, 1, 1))),
    )

    functor(types, update_diagram; generators)
end

"""
    predict(w, x)

Predict an output given weights `w` and an input `x`.
"""
function predict(w, x)
    types = (GaussRelDom, GaussianRelation)

    generators = Dict(
        weights      => w,
        input_space  => GaussRelDom(9),
        output_space => GaussRelDom(1),
        input        => GaussianRelation(ϕ(x)),
        error        => GaussianRelation(GaussianDistribution(Matrix(0.04I, 1, 1))),
    )

    functor(types, predict_diagram; generators)
end

"""
    plot_predictions(w, xs, ys)

Plot predicted outputs against true outputs and observations.
"""
function plot_predictions(w, xs, ys)
    D₁ = DataFrame(x=xs, y=ys)
    D₂ = DataFrame(map(range(0, 1, 50)) do x
        Σ, μ, _... = params(predict(w, x))
        ytrue = sin(2π * x)
        ypred = μ[1]
        ymin = ypred - √abs(Σ[1, 1])
        ymax = ypred + √abs(Σ[1, 1])
        (x=x, ytrue=ytrue, ypred=ypred, ymin=ymin, ymax=ymax)
    end)

    plot(
        layer(D₁, x=:x, y=:y, color=[colorant"black"], Geom.point),
        layer(D₂, x=:x, y=:ytrue, color=[colorant"black"], Geom.line),
        layer(D₂, x=:x, y=:ypred, ymin=:ymin, ymax=:ymax, color=[colorant"red"], Geom.line, Geom.ribbon),
        Coord.cartesian(xmin=0, xmax=1, ymin=-2, ymax=2),
        Guide.xlabel(""),
        Guide.ylabel(""),
    )
end;
# The prior over the weights is a centered Gaussian distribution with covariance matrix ``0.5I``.
xs = [0.76, 0.50, 0.93, 0.38, 0.22, 0.98, 0.92, 0.39, 0.32, 0.34, 0.46, 0.10, 0.37, 0.08, 0.16, 0.97, 0.69, 0.63, 0.77, 0.23]
ys = [-1.03, -0.02, -0.31, 0.51, 0.80, -0.02, -0.28, 0.79, 0.80, 0.70, 0.21, 0.71, 0.93, 0.58, 0.77, -0.29, -1.11, -0.86, -1.06, 1.00]
ws = []

push!(ws, GaussianRelation(GaussianDistribution(Matrix(0.5I, 9, 9))))
for (x, y) in zip(xs, ys)
    push!(ws, update(ws[end], x, y))
end

p₁ = plot_predictions(ws[2], xs[1:1], ys[1:1])
p₂ = plot_predictions(ws[3], xs[1:2], ys[1:2])
p₃ = plot_predictions(ws[5], xs[1:4], ys[1:4])
p₄ = plot_predictions(ws[21], xs[1:20], ys[1:20])

set_default_plot_size(21cm, 16cm)
gridstack([p₁ p₂; p₃ p₄])
#
