mutable struct JoinTree{T₁, T₂ <: Variable}
    id::T₁
    domain::Set{T₂}
    children::Vector{JoinTree{T₁, T₂}}
    parent::Union{Nothing, JoinTree{T₁, T₂}}
    factor::Union{Nothing, Valuation{T₂}}
    message_from_parent::Union{Nothing, Valuation{T₂}}
    message_to_parent::Union{Nothing, Valuation{T₂}}

    function JoinTree(id::T₁, domain::Set{T₂}) where {T₁, T₂}
        new{T₁, T₂}(id, domain, JoinTree{T₁, T₂}[], nothing, nothing, nothing, nothing)
    end
end

function construct_join_tree(hyperedges::Vector{Set{T}},
                             elimination_sequence::Vector{T}) where T <: Variable
    hyperedges = copy(hyperedges)
    color = Bool[]; nodes = JoinTree{Int, T}[]
    for X in elimination_sequence
        mask = [X in s for s in hyperedges]
        cl = ∪(hyperedges[mask]...)
        keepat!(hyperedges, .!mask); push!(hyperedges, setdiff(cl, [X]))
        i = JoinTree(length(nodes) + 1, cl); push!(color, true)
        for j in nodes
            if X in j.domain && color[j.id]
                push!(i.children, j)
                j.parent = i
                color[j.id] = false
            end
        end
        push!(nodes, i)
    end
    join_tree = JoinTree(length(nodes) + 1, ∪(hyperedges...))
    for j in nodes
        if color[j.id]
            push!(join_tree.children, j)
            j.parent = join_tree
            color[j.id] = false
        end
    end
    join_tree
end

function construct_factors!(join_tree::JoinTree{T₁, T₂},
                            assignment_map::Vector{T₁},
                            knowledge_base::Vector{<:Valuation{T₂}};
                            identity=true) where {T₁, T₂}
    node_map = Dict(node.id => node for node in PreOrderDFS(join_tree))
    e = IdentityValuation{T₂}()
    for node in values(node_map)
        node.factor = identity ? e : neutral_valuation(node.domain)
    end
    for (i, j) in enumerate(assignment_map)
        node_map[j].factor = combine(node_map[j].factor, knowledge_base[i])
    end
end

function message_to_parent(node::JoinTree)
    @assert !isroot(node)
    factor = node.factor
    for child in node.children
        factor = combine(factor, message_to_parent(child))
    end
    project(factor, domain(factor) ∩ node.parent.domain)
end

function message_to_parent!(node::JoinTree)
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent!(child))
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain)
    end
    node.message_to_parent
end

function message_from_parent!(node::JoinTree)
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent!(sibling))
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent!(node.parent))
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain)
    end
    node.message_from_parent
end

"""
    collect_algorithm(factors::AbstractVector{<:Valuation{T₁}},
                      domains::AbstractVector{T₂},
                      tree::Node{Int},
                      query::AbstractSet{T₁}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}

An implementation of the collect algorithm.
"""
function collect_algorithm(join_tree::JoinTree{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}
    @assert query ⊆ join_tree.domain
    factor = join_tree.factor
    for child in join_tree.children
        factor = combine(factor, message_to_parent(child))
    end
    project(factor, query)
end

"""
    shenoy_shafer_architecture!(mailboxes::AbstractDict{Tuple{Int, Int}, Valuation{T₁}},
                                factors::AbstractVector{<:Valuation{T₁}},
                                domains::AbstractVector{T₂},
                                tree::Node{Int},
                                query::AbstractSet{T₁}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}

An implementation of the Shenoy-Shafer architecture.
"""
function shenoy_shafer_architecture!(join_tree::JoinTree{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}
    for node in PreOrderDFS(join_tree)
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
    error()
end

###########################
# AbstractTrees interface #
###########################

function ChildIndexing(::Type{<:JoinTree})
    IndexedChildren()
end

function NodeType(::Type{<:JoinTree})
    HasNodeType()
end

function ParentLinks(::Type{<:JoinTree})
    StoredParents()
end

function children(node::JoinTree)
    node.children
end

function nodetype(::Type{T}) where T <: JoinTree
    T
end

function nodevalue(node::JoinTree)
    node.id
end

function parent(node::JoinTree)
    node.parent
end
