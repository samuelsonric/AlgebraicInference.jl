struct KKT{T}
    cache::T
end

# Construct a KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite.
function KKT(A::AbstractMatrix, B::AbstractMatrix)
    A = convert(AbstractMatrix{Float64}, A)
    B = convert(AbstractMatrix{Float64}, B)
    n = size(B, 1)
    K = [
        A B'
        B Matrix(0I, n, n) ]
    b = zeros(size(K, 1))
    KKT(init(LinearProblem(K, b), KrylovJL_MINRES()))
end

# Solve for x:
# [ A B'] [ x ] = [ f ]
# [ B 0 ] [ y ]   [ g ]
# where A is positive semidefinite.
function solve!(K::KKT, f::AbstractVector, g::AbstractVector)
    f = convert(AbstractVector{Float64}, f)
    g = convert(AbstractVector{Float64}, g)
    n = length(f)
    K.cache.b = [f; g]
    solve!(K.cache)[1:n]
end

# Solve for X:
# [ A B'] [ X ] = [ F ]
# [ B 0 ] [ Y ]   [ G ]
# where A is positive semidefinite.
function solve!(K::KKT, F::AbstractMatrix, G::AbstractMatrix)
    n = size(F, 1)
    size(F, 2) == 0 ? zeros(n, 0) : begin
        F = convert(AbstractMatrix{Float64}, F)
        G = convert(AbstractMatrix{Float64}, G)
        mapslices([F; G]; dims=1) do b
            K.cache.b = b
            solve!(K.cache)[1:n]
        end
    end
end

# Compute the vacuous extension
# Σ ↑ l
function extend(l, _l, Σ::GaussianSystem{
    <:AbstractMatrix{T₁},
    <:AbstractMatrix{T₂},
    <:AbstractVector{T₃},
    <:AbstractVector{T₄},
    <:Any}) where {T₁, T₂, T₃, T₄}
    
    n = length(l); _n = length(_l)
    P = zeros(T₁, n, n)
    S = zeros(T₂, n, n)
    p = zeros(T₃, n)
    s = zeros(T₄, n)
    
    for _i in 1:_n
        i = findfirst(X -> X == _l[_i], l)
        P[i, i] = Σ.P[_i, _i]
        S[i, i] = Σ.S[_i, _i]        
        p[i] = Σ.p[_i]
        s[i] = Σ.s[_i]
        
        for _j in _i+1:_n
            j = findfirst(X -> X == _l[_j], l)
            P[i, j] = Σ.P[_i, _j]
            S[i, j] = Σ.S[_i, _j]
            P[j, i] = Σ.P[_j, _i]
            S[j, i] = Σ.S[_j, _i]
        end
    end
   
    σ = Σ.σ 
    GaussianSystem(P, S, p, s, σ)
end

# Compute the message
# μ i -> pa(i)
function message_to_parent(node::JoinTree{<:Any, T}) where T
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent(child)::T)
        end
        project(factor, domain(factor) ∩ node.parent.domain)
    else
        node.message_to_parent
    end
end

# Compute the message
# μ pa(i) -> i
function message_from_parent(node::T₂) where {T₁, T₂ <: JoinTree{<:Any, T₁}}
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent(sibling)::T₁)
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent(node.parent::T₂)::T₁)
        end
        project(factor, domain(factor) ∩ node.domain)
    else
        node.message_from_parent
    end
end

# Compute the message
# μ i -> pa(i),
# caching intermediate computations.
function message_to_parent!(node::JoinTree{<:Any, T}) where T
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent!(child)::T)
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain)
    end
    node.message_to_parent
end

# Compute the message
# μ pa(i) -> i,
# caching intermediate computations.
function message_from_parent!(node::T₂) where {T₁, T₂ <: JoinTree{<:Any, T₁}}
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent!(sibling)::T₁)
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent!(node.parent::T₂)::T₁)
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain)
    end
    node.message_from_parent
end

# The fill-in number of vertex v.
function fill_in_number(g::MetaGraph, v)
    fi = 0
    ns = neighbors(g, v)
    len = length(ns)
    for i in 1:len-1
        for j in i+1:len
            if !has_edge(g, ns[i], ns[j])
                fi += 1
            end
        end
    end
    fi
end

# Eliminate the vertex v.
function eliminate!(g::MetaGraph, v)
    ns = neighbors(g, v)
    len = length(ns)
    for i in 1:len-1
        for j in i+1:len
            add_edge!(g, label_for(g, ns[i]), label_for(g, ns[j]))
        end
    end
    rem_vertex!(g, v)
end
