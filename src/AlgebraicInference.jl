module AlgebraicInference

export QuadraticFunction, QuadDom, OpenQuadraticFunction, GaussianDistribution, GaussDom, OpenGaussianDistribution
export conjugate, cov, mean, oapply, params
export ∘, ⋅, □, ◊, Δ, ∇, ⊕, bottom, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, join, mcopy, meet, mmerge, mzero, oplus, plus, swap, top, zero

using Catlab, Catlab.ACSetInterface, Catlab.CategoricalAlgebra, Catlab.Theories, Catlab.WiringDiagrams
using LinearAlgebra

import Base: *, length
import Catlab.Theories: Hom, Ob
import Catlab.Theories: ∘, ⋅, □, ◊, Δ, ∇, ⊕, bottom, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, join, mcopy, meet, mmerge, mzero, oplus, plus, swap, top, zero
import Catlab.WiringDiagrams: oapply
import StatsAPI: params
import Statistics: cov, mean

include("./quadratic.jl")
include("./gaussian.jl")

end
