module AlgebraicInference

# Systems
export DenseGaussianSystem, GaussianSystem
export ⊗, cov, invcov, marginal, normal, kernel, mean, oapply, pushforward, var

# Join Trees
export JoinTree

# Valuations
export UWDBox, Valuation
export combine, domain, duplicate, project

# Inference Problems
export InferenceProblem, MinFill, MinWidth, UWDProblem
export init

# Inference Solvers
export InferenceSolver, UWDSolver
export solve, solve!

using AbstractTrees
using Catlab.ACSetInterface, Catlab.Programs, Catlab.Theories, Catlab.WiringDiagrams
using FillArrays
using Graphs
using LinearAlgebra
using LinearSolve

using Catlab.CategoricalAlgebra: StructuredCospanOb, StructuredMulticospan
using Graphs: neighbors
using LinearAlgebra: checksquare

import AbstractTrees: ChildIndexing, NodeType, ParentLinks, children, nodetype, nodevalue,
                      parent
import Base: *, +, convert, length, one, zero
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import CommonSolve: init, solve, solve!
import Statistics: cov, mean, var

include("./systems.jl")
include("./valuations.jl")
include("./trees.jl")
include("./problems.jl")
include("./solvers.jl")
include("./utils.jl")

end
