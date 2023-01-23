module AlgebraicInference

export QuadraticFunction, QuadDom, QuadraticBifunction, GaussianDistribution, GaussRelDom, GaussianRelation
export adjoint, conjugate, cov, mean, params, pushout
export ∘, ⋅, □, ◊, Δ, ∇, ⊕, bottom, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, join, mcopy, meet, mmerge, mzero, oplus, plus, swap, top, zero

using Catlab, Catlab.Theories
using LinearAlgebra

import Base: *, adjoint, length, splat
import Catlab.Theories: Hom, Ob
import Catlab.Theories: ∘, ⋅, □, ◊, Δ, ∇, ⊕, bottom, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, join, mcopy, meet, mmerge, mzero, oplus, plus, swap, top, zero
import StatsAPI: params
import Statistics: cov, mean

include("./quadratic.jl")
include("./gaussian.jl")

end
