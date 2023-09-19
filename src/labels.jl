struct Labels{T} <: AbstractVector{T}
    labels::Vector{T}
    index::Dict{T, Int}

    function Labels{T}(labels, index) where T
        @assert length(labels) == length(index)

        new{T}(labels, index)
    end
end


function Labels{T}(labels::AbstractVector) where T
    index = Dict(l => v for (v, l) in enumerate(labels))

    Labels{T}(labels, index)
end


function Labels(labels::Vector{T}, index::Dict{T, Int}) where T
    Labels{T}(labels, index)
end


function Labels(labels::AbstractVector{T}) where T
    Labels{T}(labels)
end


function Base.delete!(labels::Labels, l)
    v = labels.index[l]
    m = labels.labels[end]

    labels.labels[v] = m
    labels.index[m] = v

    pop!(labels.labels)
    pop!(labels.index, l)
end


function Base.copy(labels::Labels)
    Labels(
        deepcopy(labels.labels),
        deepcopy(labels.index))
end


############################
# AbstractVector Interface #
############################


function Base.size(A::Labels)
    (length(A.labels),)
end


function Base.getindex(A::Labels, i::Int)
    A.labels[i]
end


function Base.IndexStyle(::Type{<:Labels})
    IndexLinear()
end


function Base.setindex!(A::Labels, v, i::Int)
    A.labels[i] = v
    A.index[v] = i
end
