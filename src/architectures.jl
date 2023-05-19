"""
    Architecture{T₁, T₂ <: Variable} <: AbstractNode{T₁}

A labeled tree is a quadruple ``(V, E, \\lambda, D)``, where ``(V, E)`` is a tree and
``\\lambda: V \\to D`` is a labeling function.

Let ``D`` be the set of domains in a valuation algebra. A labeled tree 
``(V, E, \\lambda, D)`` is a *join tree* if it satisfies the running intersection property
if for two nodes ``i, j \\in V`` and ``X \\in \\lambda(i) \\cap \\lambda(j)``,
``X \\in \\lambda(k)`` for all nodes on the path between ``i`` and ``j``.

The type `Architecture` represents a join tree. It optionally associates to each vertex ``i``
a *factor* ``\\psi_i`` and *mailbox* ``(\\mu_{pa(i) \\to i}, \\mu_{i \\to pa(i)})``.

The factors are set by the function [`construct_factors!`](@ref). The mailboxes are set
by the function [`shenoy_shafer_architecture!`](@ref).
"""
mutable struct Architecture{T₁, T₂ <: Variable} <: AbstractNode{T₁}
    id::T₁
    domain::Set{T₂}
    children::Vector{Architecture{T₁, T₂}}
    parent::Union{Nothing, Architecture{T₁, T₂}}
    factor::Union{Nothing, Valuation{T₂}}
    message_from_parent::Union{Nothing, Valuation{T₂}}
    message_to_parent::Union{Nothing, Valuation{T₂}}

    function Architecture(id::T₁, domain::Set{T₂}) where {T₁, T₂}
        new{T₁, T₂}(id, domain, Architecture{T₁, T₂}[], nothing, nothing, nothing, nothing)
    end
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
    function construct_architecture(hyperedges::Vector{Set{T}},
                                    elimination_sequence::Vector{T}) where T <: Variable

Construct a join tree by eliminating the variables in `elimination_sequence`.
"""
function construct_architecture(hyperedges::Vector{Set{T}},
                                elimination_sequence::Vector{T}) where T <: Variable
    hyperedges = copy(hyperedges)
    color = Bool[]; nodes = Architecture{Int, T}[]
    for X in elimination_sequence
        mask = [X in s for s in hyperedges]
        cl = ∪(hyperedges[mask]...)
        keepat!(hyperedges, .!mask); push!(hyperedges, setdiff(cl, [X]))
        i = Architecture(length(nodes) + 1, cl); push!(color, true)
        for j in nodes
            if X in j.domain && color[j.id]
                push!(i.children, j)
                j.parent = i
                color[j.id] = false
            end
        end
        push!(nodes, i)
    end
    architecture = Architecture(length(nodes) + 1, ∪(hyperedges...))
    for j in nodes
        if color[j.id]
            push!(architecture.children, j)
            j.parent = architecture
            color[j.id] = false
        end
    end
    architecture
end

"""
    function construct_factors!(architecture::Architecture{T₁, T₂},
                                assignment_map::Vector{T₁},
                                knowledge_base::Vector{<:Valuation{T₂}};
                                identity=true) where {T₁, T₂}

Let ``(V, E, \\lambda, D)`` be a join tree, ``\\{\\phi_1, \\dots, \\phi_n \\}`` a knowledge
base, and ``a: \\{1, \\dots, \\n\\} \\to V`` an assignment mapping. If `identity=true`,
then `construct_factors!(architecture, assignment_map, knowledge_base, identity)` assigns to
each vertex ``i \\in V`` the factor
```math
    \\psi_i = e \\otimes \\bigotimes_{j:a(j)=i} \\phi_j,
```
where ``e`` is the identity element. If `identity=false`, then the function uses neutral
elements instead of the identity element. 
"""
function construct_factors!(architecture::Architecture{T₁, T₂},
                            assignment_map::Vector{T₁},
                            knowledge_base::Vector{<:Valuation{T₂}};
                            identity=true) where {T₁, T₂}
    node_map = Dict(node.id => node for node in PreOrderDFS(architecture))
    e = IdentityValuation{T₂}()
    for node in values(node_map)
        node.factor = identity ? e : neutral_valuation(node.domain)
    end
    for (i, j) in enumerate(assignment_map)
        node_map[j].factor = combine(node_map[j].factor, knowledge_base[i])
    end
end

"""
    collect_algorithm(architecture::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}

Answer a query using the collect algorithm.

Let ``(V, E, \\lambda, D)`` be a join tree with root ``r`` and factors
``\\{\\phi_i\\}_{i \\in V}``. Let ``x \\in \\lambdar(r)`` be a query. Then
`collect_algorithm(architecture, query)` solves the inference problem
```math
\\left( \\bigotimes_{i \\in V} \\psi_i \\right)^{\\downarrow x}
```

This algorithm will ignore the mailboxes in `architecture`.
"""
function answer_query(architecture::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}
    @assert query ⊆ architecture.domain
    factor = architecture.factor
    for child in architecture.children
        factor = combine(factor, message_to_parent(child))
    end
    project(factor, query)
end

"""
    shenoy_shafer_architecture!(architecture::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}

Answer a query using the Shenoy-Shafer architecture.

Let ``(V, E, \\lambda, D)`` be a join tree with factors ``\\{\\phi_i\\}_{i \\in V}``.
Let ``x`` be a query covered by ``(V, E, \\lambda, D)``. Then
`shenoy_shafer_architecture!(architecture, query)` solves the inference problem
```math
\\left( \\bigotimes_{i \\in V} \\psi_i \\right)^{\\downarrow x}
```

This algorithm will cache computation in the mailboxes in `architecture`, speeding up
subsequent invocations.
"""
function answer_query!(architecture::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}
    for node in PreOrderDFS(architecture)
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
