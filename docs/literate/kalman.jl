# # Kalman Filter
# Constructing the Kalman filter described in [Example 9 -- Vehicle Location Estimation](https://www.kalmanfilter.net/multiExamples.html).
using AlgebraicInference
using Catlab, Catlab.Theories, Catlab.Graphics, Catlab.WiringDiagrams
import Convex, SCS
import TikzPictures

# First, we construct a diagram representing the filter's prediction step.
O = mzero(FreeAbelianBicategoryRelations.Ob)
X = Ob(FreeAbelianBicategoryRelations, :X)
Q = Hom(:Q, O, X)
F = Hom(:F, X, X)

kalman_predict = (F ⊕ Q) ⋅ plus(X)
to_tikz(kalman_predict)

# Next, we construct a diagram representing the filter's observation step.
Z = Ob(FreeAbelianBicategoryRelations, :Z)
R = Hom(:R, O, Z)
H = Hom(:H, X, Z)

kalman_observe = Δ(X) ⋅ (id(X) ⊕ H ⊕ R) ⋅ (id(X) ⊕ plus(Z))
to_tikz(kalman_observe)

# Finally, we construct a diagram representing two iterations of the filter.
P₀ = Hom(:P_0, O, X)
z₁ = Hom(:z_1, O, Z)
z₂ = Hom(:z_2, O, Z)

kalman_filter = P₀
kalman_filter = kalman_filter ⋅ kalman_predict ⋅ kalman_observe ⋅ (id(X) ⊕ dagger(z₁))
kalman_filter = kalman_filter ⋅ kalman_predict ⋅ kalman_observe ⋅ (id(X) ⊕ dagger(z₂))
to_tikz(kalman_filter)

# To perform a computation, we assign data to the boxes and wires of our filter diagram. The output is a *Gaussian relation*, representing the estimated state tat time ``t = 2`` given observations ``z_1`` and ``z_2``.
types = (GaussRelDom, GaussianRelation)

generators = Dict(
    X => GaussRelDom(6),
    Z => GaussRelDom(2),
    F => GaussianRelation([
        1   1   1/2 0   0   0
        0   1   1   0   0   0
        0   0   1   0   0   0
        0   0   0   1   1   1/2
        0   0   0   0   1   1
        0   0   0   0   0   1
    ]),
    H => GaussianRelation([
        1   0   0   0   0   0
        0   0   0   1   0   0
    ]),
    Q => GaussianRelation(GaussianDistribution([
        1/4 1/2 1/2 0   0   0
        1/2 1   1   0   0   0
        1/2 1   1   0   0   0
        0   0   0   1/4 1/2 1/2
        0   0   0   1/2 1   1
        0   0   0   1/2 1   1
    ] * 1/25)),
    R => GaussianRelation(GaussianDistribution([
        9   0
        0   9
    ])),
    P₀ => GaussianRelation(GaussianDistribution([
        500 0   0   0   0   0
        0   500 0   0   0   0
        0   0   500 0   0   0
        0   0   0   500 0   0
        0   0   0   0   500 0
        0   0   0   0   0   500
    ])),
    z₁ => GaussianRelation(GaussianDistribution([-393.66, 300.40])),
    z₂ => GaussianRelation(GaussianDistribution([-375.93, 301.78])),
)

d = functor(types, kalman_filter; generators)

# The covariance of the estimated state at time ``t = 2``.
round.(cov(d); digits=2)

# The mean of the estimated state at time ``t = 2``.
round.(mean(d); digits=2)
