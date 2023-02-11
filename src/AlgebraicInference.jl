module AlgebraicInference

export GaussDom, GaussianDistribution, OpenGaussianDistribution, OpenQuadraticFunction, ScheduledUntypedHypergraphDiagram, SchScheduledUntypedHypergraphDiagram, QuadDom, QuadraticFunction
export conjugate, cov, eval_schedule, mean, oapply, params
export ∘, ⋅, □, ◊, Δ, ∇, ⊕, bottom, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, join, mcopy, meet, mmerge, mzero, oplus, plus, swap, top, zero

using Catlab, Catlab.ACSetInterface, Catlab.CategoricalAlgebra, Catlab.Theories, Catlab.WiringDiagrams
using Catlab.WiringDiagrams.ScheduleUndirectedWiringDiagrams: SchScheduledUWD
using LinearAlgebra

import Base: *, length
import Catlab.Theories: Hom, Ob
import Catlab.Theories: ∘, ⋅, □, ◊, Δ, ∇, ⊕, bottom, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, join, mcopy, meet, mmerge, mzero, oplus, plus, swap, top, zero
import Catlab.WiringDiagrams: eval_schedule, oapply
import StatsAPI: params
import Statistics: cov, mean

include("./quadratic.jl")
include("./gaussian.jl")
include("./scheduled.jl")

end
