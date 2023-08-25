"""
    EliminationAlgorithm

An algorithm for ordering the vertices of an undirected graph. The ordering is used to
construct a join tree.
"""
abstract type EliminationAlgorithm end

"""
    MinDegree <: EliminationAlgorithm

The min-degree heuristic.
"""
struct MinDegree <: EliminationAlgorithm end

"""
    MinFill <: EliminationAlgorithm

The min-fill heuristic.
"""
struct MinFill <: EliminationAlgorithm end
