module AlgebraicInference


# Systems
export CanonicalForm, DenseCanonicalForm, DenseGaussianSystem, GaussianSystem
export ⊗, cov, invcov, normal, kernel, mean, oapply, var


# Inference Problems
export InferenceProblem
export init


# Inference Solvers
export InferenceSolver
export solve, solve!


# Algorithms
export EliminationAlgorithm, EliminationOrder, EliminationTree, AMDJL_AMD,
       CuthillMcKeeJL_RCM, MetisJL_ND, MinDegree, MinFill


using AbstractTrees
using BayesNets
using Catlab
using CommonSolve
using Distributions
using FillArrays
using LinearAlgebra
using LinearSolve
using Statistics


using Base: OneTo
using FillArrays: SquareEye, ZerosMatrix, ZerosVector
using LinearAlgebra: checksquare
using LinearSolve: SciMLLinearSolveAlgorithm

import AMD
import CuthillMcKee
import Graphs
import Metis


include("./kkt.jl")
include("./systems.jl")
include("./factors.jl")
include("./labels.jl")
include("./models.jl")
include("./elimination.jl")
include("./architectures.jl")
include("./problems.jl")
include("./solvers.jl")
include("./utils.jl")


end
