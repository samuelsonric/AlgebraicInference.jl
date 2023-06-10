"""
    InferenceSolver{T₁, T₂}
"""
mutable struct InferenceSolver{T₁, T₂}
    jt::JoinTree{T₁, T₂}
    query::Vector{T₂}
end

"""
    solve(is::InferenceSolver)
"""
function solve(is::InferenceSolver{T}) where T
    domain = collect(Set(is.query))
    for node in PreOrderDFS(is.jt)
        if domain ⊆ node.domain
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent(child)::T)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent(node)::T)
            end
            return duplicate(project(factor, domain), is.query)
        end 
    end
    error("Query not covered by join tree.")
end

"""
    solve!(is::InferenceSolver)
"""
function solve!(is::InferenceSolver{T}) where T
    domain = collect(Set(is.query))
    for node in PreOrderDFS(is.jt)
        if domain ⊆ node.domain
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent!(child)::T)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent!(node)::T)
            end
            return duplicate(project(factor, domain), is.query)
        end 
    end
    error("Query not covered by join tree.")
end
