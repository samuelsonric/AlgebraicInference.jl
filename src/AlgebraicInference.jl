module AlgebraicInference

export AbstractSystem, ClassicalSystem, LabeledSystem, System, Valuation
export ⊗, ↓, cov, d, dof, fiber, mean, oapply 

using Catlab, Catlab.ACSetInterface, Catlab.CategoricalAlgebra, Catlab.WiringDiagrams
using LinearAlgebra
using OrderedCollections

import Base: ==, \, *, convert, length
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import StatsBase: dof
import Statistics: cov, mean

include("./systems.jl")
include("./utilities.jl")
include("./valuations.jl")

end
