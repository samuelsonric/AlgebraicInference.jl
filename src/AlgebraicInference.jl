module AlgebraicInference

# Inference Problems
export InferenceProblem, MinFill, MinWidth, UWDProblem
export init

# Join Trees
export JoinTree
export solve, solve!

# Systems
export GaussianSystem
export ⊗, canon, cov, invcov, marginal, normal, kernel, mean, oapply, pushfwd

# Valuations
export IdentityValuation, UWDBox, Valuation
export combine, domain, project

using AbstractTrees
using Base.Iterators: take, drop
using Catlab, Catlab.CategoricalAlgebra, Catlab.Programs, Catlab.WiringDiagrams
using Graphs
using LinearAlgebra
using LinearSolve
using MetaGraphsNext

using Graphs: add_edge!, add_vertex!, has_edge, neighbors, nv, vertices
using LinearAlgebra: checksquare

import AbstractTrees: ChildIndexing, NodeType, ParentLinks, children, nodetype, nodevalue,
                      parent
import Base: *, +, convert, length, one, zero
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import CommonSolve: init, solve, solve!
import StatsBase: dof
import Statistics: cov, mean

include("./systems.jl")
include("./valuations.jl")
include("./problems.jl")
include("./trees.jl")
include("./utils.jl")

end
