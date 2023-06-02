"""
    Architecture{T₁, T₂} <: AbstractNode{T₁}

A join tree ``(V, E, \\lambda, D)``, along with a set of factors
```math
    \\left{ \\psi_i \\right}_{i \\in V}
``
and mailboxes
```math
    \\left{ \\left( \\mu_{i \\to \\mathtt{pa}(i)}, \\mu_{\\mathtt{pa}(i) \\to i} \\right) \\right}_{i \\in V}.
```
"""
mutable struct Architecture{T₁, T₂} <: AbstractNode{T₁}
    id::T₁
    domain::Vector{T₂}
    factor::Valuation{T₂}
    children::Vector{Architecture{T₁, T₂}}
    parent::Union{Nothing, Architecture{T₁, T₂}}
    message_from_parent::Union{Nothing, Valuation{T₂}}
    message_to_parent::Union{Nothing, Valuation{T₂}}

    function Architecture(id::T₁, domain::Vector{T₂}, factor::Valuation{T₂}) where {T₁, T₂}
        new{T₁, T₂}(id, domain, factor, Architecture{T₁, T₂}[], nothing, nothing, nothing)
    end
end

"""
    architecture(kb::AbstractVector{<:Valuation{T}}, order) where T

Construct a covering join tree for the knowledge base `kb` using the variable elimination
order `order`.
"""
function architecture(kb::AbstractVector{<:Valuation{T}}, order) where T
    kb = copy(kb); pg = primal_graph(kb)
    color = Bool[]
    nodes = Architecture{Int, T}[]
    e = IdentityValuation{T}()
    for X in order
        cl = collect(neighbor_labels(pg, X)); push!(cl, X)
        fa = e
        for i in length(kb):-1:1
            if X in domain(kb[i])
                fa = combine(fa, kb[i])
                deleteat!(kb, i)
            end
        end
        node = Architecture(length(nodes) + 1, cl, fa); push!(color, true)
        eliminate!(pg, code_for(pg, X))
        for _node in nodes
            if X in _node.domain && color[_node.id]
                push!(node.children, _node)
                _node.parent = node
                color[_node.id] = false
            end
        end
        push!(nodes, node)
    end
    node = Architecture(length(nodes) + 1, collect(labels(pg)), reduce(combine, kb; init=e))
    for _node in nodes
        if color[_node.id]
            push!(node.children, _node)
            _node.parent = node
            color[_node.id] = false
        end
    end
    node
end

function ChildIndexing(::Type{<:Architecture})
    IndexedChildren()
end

function NodeType(::Type{<:Architecture})
    HasNodeType()
end

function ParentLinks(::Type{<:Architecture})
    StoredParents()
end

function children(node::Architecture)
    node.children
end

function nodetype(::Type{T}) where T <: Architecture
    T
end

function nodevalue(node::Architecture)
    node.id
end

function parent(node::Architecture)
    node.parent
end

"""
    answer_query(jt::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}

Answer a query.

Let ``(V, E, \\lambda, D)`` be a join tree with factors ``\\{\\phi_i\\}_{i \\in V}`` and
``x`` a query covered by ``(V, E, \\lambda, D)``. Then `answer_query(jt, query)` solves the
inference problem
```math
\\left( \\bigotimes_{i \\in V} \\psi_i \\right)^{\\downarrow x}.
```
"""
function answer_query(jt::Architecture, query)
    for node in PreOrderDFS(jt)
        if query ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent(child))
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent(node))
            end
            return project(factor, query)
        end 
    end
    error("Query not covered by join tree.")
end

"""
    answer_query!(jt::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}

Answer a query, caching intermediate computations.

Let ``(V, E, \\lambda, D)`` be a join tree with factors ``\\{\\phi_i\\}_{i \\in V}`` and
``x`` be a query covered by ``(V, E, \\lambda, D)``. Then `answer_query!(jt, query)` solves
the inference problem
```math
\\left( \\bigotimes_{i \\in V} \\psi_i \\right)^{\\downarrow x}.
```
"""
function answer_query!(jt::Architecture, query)
    for node in PreOrderDFS(jt)
        if query ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent!(child))
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent!(node))
            end
            return project(factor, query)
        end 
    end
    error("Query not covered by join tree.")
end
