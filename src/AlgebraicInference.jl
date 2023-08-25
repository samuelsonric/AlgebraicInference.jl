module AlgebraicInference

# Systems
export CanonicalForm, DenseCanonicalForm, DenseGaussianSystem, GaussianSystem
export ⊗, cov, invcov, normal, kernel, mean, oapply, pushforward, var

# Inference Problems
export InferenceProblem
export init

# Inference Solvers
export InferenceSolver
export solve, solve!

# Algorithms
export EliminationAlgorithm, MinDegree, MinFill

using AbstractTrees
using BayesNets
using Catlab.ACSetInterface, Catlab.Programs, Catlab.Theories, Catlab.WiringDiagrams
using Distributions
using FillArrays
using LinearAlgebra
using LinearSolve

using Base: OneTo
using FillArrays: SquareEye, ZerosMatrix, ZerosVector
using LinearAlgebra: checksquare

import AbstractTrees
import Catlab
import CommonSolve
import Distributions
import Graphs
import Statistics

include("./systems.jl")
include("./factors.jl")
include("./graphs.jl")
include("./models.jl")
include("./algorithms.jl")
include("./problems.jl")
include("./trees.jl")
include("./solvers.jl")
include("./utils.jl")

end
