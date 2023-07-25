"""
    MinDegree

Contructs a covering join tree for an inference problem using the variable elimination
algorithm. Variables are eliminated according to the "minimum degree" heuristic.
"""
struct MinDegree end

"""
    MinFill

Contructs a covering join tree for an inference problem using the variable elimination
algorithm. Variables are eliminated according to the "minimum fill" heuristic.
"""
struct MinFill end
