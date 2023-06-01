"""
    primal_graph(kb::AbstractVector{<:Valuation{T}}) where T

Construct the primal graph of the knowledge base `kb`.
"""
function primal_graph(kb::AbstractVector{<:Valuation{T}}) where T
    g = MetaGraph(Graph(); label_type=T)
    for ϕ in kb
        d = collect(domain(ϕ))
        n = length(d)
        for i in 1:n
            add_vertex!(g, d[i])
            for j in 1:i - 1
                add_edge!(g, d[i], d[j])
            end   
        end
    end
    g
end

"""
    minfill!(g::MetaGraph, query)

Compute a vertex elimination order using the min-fill heuristic.
"""
function minfill!(g::MetaGraph{<:Any, <:Any, T}, query) where T
    n = nv(g) - length(query)
    order = Vector{T}(undef, n)
    for i in 1:n
        q = [code_for(g, X) for X in query]
        X = argmin(X -> X in q ? typemax(Int) : fill_in_number(g, X), vertices(g))
        order[i] = label_for(g, X)
        eliminate!(g, X)
    end
    order
end

"""
    minwidth!(g::MetaGraph, query)

Compute a vertex elimination order using the min-width heuristic 
"""
function minwidth!(g::MetaGraph{<:Any, <:Any, T}, query) where T
    n = nv(g) - length(query)
    order = Vector{T}(undef, n)
    for i in 1:n
        q = [code_for(g, X) for X in query]
        X = argmin(X -> X in q ? typemax(Int) : degree(g, X), vertices(g))
        order[i] = label_for(g, X)
        eliminate!(g, X)
    end
    order
end
