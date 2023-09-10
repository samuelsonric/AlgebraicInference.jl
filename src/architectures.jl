# A mailbox in the Shenoy-Shafer architecture.
mutable struct SSMailbox{T₁, T₂}
    factor::Union{Nothing, Factor{T₁, T₂}}
    message_to_parent::Union{Nothing, Factor{T₁, T₂}}
    message_from_parent::Union{Nothing, Factor{T₁, T₂}}
end


# The Shenoy-Shafer architecture.
mutable struct SSArchitecture{T₁, T₂, T₃}
    labels::Labels{T₁}
    factors::Vector{Factor{T₂, T₃}}
    order::EliminationOrder
    tree::EliminationTree
    v_to_fs::Vector{Vector{Int}}
    mailboxes::Vector{SSMailbox{T₂, T₃}}
    mailboxes_full::Bool
end


function SSMailbox{T₁, T₂}() where {T₁, T₂}
    SSMailbox{T₁, T₂}(nothing, nothing, nothing)
end


function SSArchitecture(
    labels::Labels,
    factors::Vector{Factor{T₁, T₂}},
    order::EliminationOrder,
    tree::EliminationTree,
    v_to_fs::Vector{Vector{Int}}) where {T₁, T₂}

    mailboxes = [SSMailbox{T₁, T₂}() for _ in 1:length(labels)]
    mailboxes_full = false

    SSArchitecture(
        labels,
        factors,
        order,
        tree,
        v_to_fs,
        mailboxes,
        mailboxes_full)
end


function SSArchitecture(model::GraphicalModel, order::EliminationOrder)
    labels = copy(model.labels)
    factors = copy(model.factors)
    v_to_fs = deepcopy(model.v_to_fs)

    tree = EliminationTree(model.graph, order)

    for v in order, f in v_to_fs[v], w in factors[f].vars
        if v != w
            setdiff!(v_to_fs[w], f)
        end
    end

    SSArchitecture(labels, factors, order, tree, v_to_fs)
end


# Answer a query.
function CommonSolve.solve!(arch::SSArchitecture, query)
    if !arch.mailboxes_full
        fill_mailboxes!(arch)
    end

    vars = [arch.labels.index[l] for l in query]

    for v in arch.order
        if vars ⊆ [v; arch.tree[v]]
            fac = factor!(arch, v)

            for t in childindices(arch.tree, v)
                msg = message_to_parent!(arch, t)
                fac = combine(fac, msg)
            end

            if v != rootindex(arch.tree)
                msg = message_from_parent!(arch, v)
                fac = combine(fac, msg)
            end

            fac = project(fac, vars)
    
            return permute(fac, vars)
        end 
    end

    error("Query not covered by join tree.")
end


function fill_mailboxes!(arch::SSArchitecture)
    for v in arch.order[1:end - 1] # Collect phase
        message_to_parent!(arch, v)
    end

    for v in arch.order[end - 1:-1:1] # Distribute phase
        message_from_parent!(arch, v)
    end

    arch.mailboxes_full = true
end


# Compute the join tree factor
# ψᵥ
function factor!(arch::SSArchitecture{<:Any, T₁, T₂}, v::Int) where {T₁, T₂}
    mbx = arch.mailboxes[v]

    if isnothing(mbx.factor)
        fac = zero(Factor{T₁, T₂})

        for f in arch.v_to_fs[v]
            fac = combine(fac, arch.factors[f])
        end

        mbx.factor = fac
    end

    mbx.factor::Factor{T₁, T₂}
end


# Compute the message
# μ v → pa(v)
function message_to_parent!(arch::SSArchitecture{<:Any, T₁, T₂}, v::Int) where {T₁, T₂}
    @assert v != rootindex(arch.tree)

    mbx = arch.mailboxes[v]

    if isnothing(mbx.message_to_parent)
        fac = factor!(arch, v)

        for t in childindices(arch.tree, v)
            msg = message_to_parent!(arch, t)
            fac = combine(fac, msg)
        end

        mbx.message_to_parent = project(fac, arch.tree[v])
    end

    mbx.message_to_parent::Factor{T₁, T₂}
end


# Compute the message
# μ pa(v) → v
function message_from_parent!(arch::SSArchitecture{<:Any, T₁, T₂}, v::Int) where {T₁, T₂}
    @assert v != rootindex(arch.tree)

    mbx = arch.mailboxes[v]

    if isnothing(mbx.message_from_parent)
        u = parentindex(arch.tree, v)

        fac = factor!(arch, u)

        for t in childindices(arch.tree, u)
            if t != v
                msg = message_to_parent!(arch, t)
                fac = combine(fac, msg)
            end
        end

        if u != rootindex(arch.tree)
            msg = message_from_parent!(arch, u)
            fac = combine(fac, msg)
        end

        mbx.message_from_parent = project(fac, arch.tree[v])
    end

    mbx.message_from_parent::Factor{T₁, T₂}
end
