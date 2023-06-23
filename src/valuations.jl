"""
    Valuation{T}

A filler for a box in an undirected wiring diagram, labeled with the junctions to which the
box is incident.
"""
struct Valuation{T}
    hom::T
    labels::Vector{Int}
    index::Dict{Int, Int}

    function Valuation{T}(hom, labels, index) where T
        @assert length(labels) == length(index)
        new{T}(hom, labels, index)
    end
end

function Valuation{T}(hom, labels) where T
    n = length(labels)
    Valuation{T}(hom, labels, Dict(zip(labels, 1:n)))
end

function convert(::Type{Valuation{T}}, ϕ::Valuation) where T
    Valuation{T}(ϕ.hom, ϕ.labels, ϕ.index)
end

"""
    length(ϕ::Valuation)

Get the size of the domain of ``\\phi``.
"""
function length(ϕ::Valuation)
    length(ϕ.labels)
end

"""
    domain(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
function domain(ϕ::Valuation)
    ϕ.labels
end

"""
    combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
function combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T
    port_labels = [ϕ₁.labels..., ϕ₂.labels...]
    outer_port_labels = ϕ₁.labels ∪ ϕ₂.labels
    junction_labels = outer_port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(ϕ₁.labels)); add_box!(wd, length(ϕ₂.labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    hom = oapply(wd, [ϕ₁.hom, ϕ₂.hom])
    Valuation{T}(hom, outer_port_labels)
end

function combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T <: GaussianSystem
    hom, labels = combine(ϕ₁.hom, ϕ₂.hom, ϕ₁.labels, ϕ₂.labels, ϕ₁.index, ϕ₂.index)
    Valuation{T}(hom, labels)
end

"""
    project(ϕ::Valuation, x)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
#=
function project(ϕ::Valuation{T}, x) where T
    @assert x ⊆ ϕ.labels
    port_labels = ϕ.labels
    outer_port_labels = collect(x)
    junction_labels = port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(port_labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, i; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    hom = oapply(wd, [ϕ.hom])
    Valuation{T}(hom, outer_port_labels)
end
=#

function project(ϕ::Valuation{T}, x) where T
    n = length(ϕ)
    wd = cospan_diagram(UntypedUWD, 1:n, map(l -> ϕ.index[l], x), n)
    Valuation{T}(oapply(wd, [ϕ.hom]), x)
end

#=
function project(ϕ::Valuation{T}, x) where T <: GaussianSystem
    @assert x ⊆ ϕ.labels
    m = in.(ϕ.labels, [x])
    Valuation{T}(marginal(ϕ.hom, m), ϕ.labels[m])
end
=#

"""
    extend(ϕ::Valuation, objects, x)

Perform the vacuous extension ``\\phi^{\\uparrow x}``
"""
function extend(ϕ::Valuation{T}, objects, x) where T
    port_labels = ϕ.labels
    outer_port_labels = x
    junction_labels = port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(port_labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, i; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    hom = oapply(wd, [ϕ.hom], isnothing(objects) ? nothing : objects[junction_labels])
    Valuation{T}(hom, outer_port_labels)
end

"""
    one(::Type{Valuation{T}}) where T

Construct an identity element.
"""
one(::Type{Valuation{T}}) where T

function one(::Type{Valuation{T}}) where {L, T <: StructuredMulticospan{L}}
    Valuation{T}(id(munit(StructuredCospanOb{L})), [])
end

function one(::Type{Valuation{T}}) where T <: GaussianSystem
    Valuation{T}(zero(T, 0), [])
end

# TODO: Docstring
function pull_onto(ϕ::Valuation{T}, x) where T
    port_labels = ϕ.labels
    outer_port_labels = x
    junction_labels = port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(port_labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, i; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    convert(T, oapply(wd, [ϕ.hom]))
end

