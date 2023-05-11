"""
    AbstractSystem

Abstract type for Gaussian systems. 

Subtypes should specialize the following methods:
- [`length(Σ::AbstractSystem)`](@ref)
- [`fiber(Σ::AbstractSystem)`](@ref)
- [`mean(Σ::AbstractSystem)`](@ref)
- [`cov(Σ::AbstractSystem)`](@ref)
- [`*(M::AbstractMatrix, Σ::AbstractSystem)`](@ref)
- [`\\(M::AbstractMatrix, Σ::AbstractSystem)`](@ref)
- [`⊗(Σ₁::AbstractSystem, Σ₂::AbstractSystem)`](@ref)

References:
- J. C. Willems, "Open Stochastic Systems," in *IEEE Transactions on Automatic Control*, 
  vol. 58, no. 2, pp. 406-421, Feb. 2013, doi: 10.1109/TAC.2012.2210836.
"""
abstract type AbstractSystem end

"""
    ClassicalSystem{T₁ <: AbstractMatrix, T₂ <: AbstractVector} <: AbstractSystem

A classical Gaussian system. See [`AbstractSystem`](@ref).
"""
struct ClassicalSystem{T₁ <: AbstractMatrix, T₂ <: AbstractVector} <: AbstractSystem
    Γ::T₁
    μ::T₂
    
    @doc """
        ClassicalSystem(Γ::AbstractMatrix, μ::AbstractVector)

    Construct a classical Gaussian system with mean ``\\mu`` and covariance ``\\Gamma``.
    """
    function ClassicalSystem(Γ::T₁, μ::T₂) where {T₁ <: AbstractMatrix, T₂ <: AbstractVector}
        @assert size(Γ, 1) == size(Γ, 2) == length(μ)
        new{T₁, T₂}(Γ, μ)
    end
end

"""
    System{T₁ <: AbstractMatrix, T₂, T₃} <: AbstractSystem

A Gaussian system. See [`AbstractSystem`](@ref).
"""
struct System{T₁ <: AbstractMatrix, T₂, T₃} <: AbstractSystem
    R::T₁
    ϵ::ClassicalSystem{T₂, T₃}

    @doc """
        System(R::AbstractMatrix, ϵ::ClassicalSystem)    

    Let ``R`` be an ``m \\times n`` matrix and ``\\epsilon`` an ``m``-variate random
    vector with mean ``\\mu`` and covariance ``\\Gamma``.

    If ``\\mu \\in \\mathtt{image}(R : \\Gamma)``, then there exists a random variable
    ``\\hat{w}`` taking values in ``(\\mathbb{R}^n, \\sigma R)`` that almost-surely solves
    the convex program
    ```math
        \\begin{align*}
            \\underset{w}{\\text{minimize }} & E(\\epsilon, w) \\\\
            \\text{subject to }              & Rw \\in \\mathtt{image}(\\Gamma) + \\epsilon,
        \\end{align*}
    ```
    where
    ```math
        \\sigma R = \\{ R^{-1}B \\mid B \\in \\mathcal{B}(\\mathbb{R}^m) \\}

    ```
    and ``E(-, w)`` is the negative log-density of the multivariate normal distribution
    ``\\mathcal{N}(Rw, \\Gamma)``.

    If ``\\mu \\in \\mathtt{image}(R : \\Gamma)``, then `System(R, ϵ)` constructs the
    Gaussian system ``\\Sigma = (\\mathbb{R}^n, \\sigma R, P)``, where ``P`` is the
    distribution of ``\\hat{w}``.

    In particular, if ``R`` has full row-rank, then ``Rw = \\epsilon`` is a kernel
    representation of ``\\Sigma``.
    """
    function System(R::T₁, ϵ::ClassicalSystem{T₂, T₃}) where {T₁ <: AbstractMatrix, T₂, T₃}
        @assert size(R, 1) == length(ϵ)
        new{T₁, T₂, T₃}(R, ϵ)
    end

end

"""
    ClassicalSystem(Γ::AbstractMatrix)

Construct a classical Gaussian system with mean ``\\mathbf{0}`` and covariance
``\\Gamma``.
"""
function ClassicalSystem(Γ::AbstractMatrix)
    μ = zero(diag(Γ))
    ClassicalSystem(Γ, μ)
end

"""
    ClassicalSystem(μ::AbstractVector)

Construct a classical Gaussian system with mean ``\\mu`` and covariance ``\\mathbf{0}``.
"""
function ClassicalSystem(μ::AbstractVector)
    Γ = zero(μ * μ')
    ClassicalSystem(Γ, μ)
end

"""
    System(R::AbstractMatrix)

Let ``R`` be an ``m \\times n`` matrix. Then `System(R)` constructs the deterministic
Gaussian system  ``\\Sigma = (\\mathbb{R}^n, \\sigma R, P),`` where
```math
    \\sigma R = \\{ R^{-1}B \\mid B \\in \\mathcal{B}(\\mathbb{R}^m)\\}
```
and
```math
    P(R^{-1}B) = \\begin{cases}
        1 & 0 \\in B     \\\\
        0 & \\text{else}
    \\end{cases}.
```
"""
function System(R::AbstractMatrix)
    ϵ = ClassicalSystem(zero(R * R'))
    System(R, ϵ)
end

function System(Σ::ClassicalSystem)
    R = one(Σ.Γ)
    ϵ = Σ
    System(R, ϵ)
end

function convert(::Type{System}, Σ::ClassicalSystem)
    System(Σ)
end

function convert(::Type{ClassicalSystem}, Σ::System)
    @assert Σ.R == one(Σ.R)
    Σ.ϵ
end

"""
    length(Σ::AbstractSystem)

Let ``\\Sigma = (\\mathbb{R}^n, \\mathcal{E}, P)``. Then `length(Σ)` gets the dimension
``n``.
"""
length(Σ::AbstractSystem) = length(mean(Σ))

function length(Σ::System)
    size(Σ.R, 2)
end

function ==(Σ₁::ClassicalSystem, Σ₂::ClassicalSystem)
    (Σ₁.Γ == Σ₂.Γ) && (Σ₁.μ == Σ₂.μ)
end

function ==(Σ₁::System, Σ₂::System)
    (Σ₁.R == Σ₂.R) && (Σ₁.ϵ == Σ₂.ϵ)
end

"""
    ⊗(Σ₁::AbstractSystem, Σ₂::AbstractSystem)

Compute the product ``\\Sigma_1 \\times \\Sigma_2``.
"""
function ⊗(Σ₁::AbstractSystem, Σ₂::AbstractSystem)
    convert(System, Σ₁) ⊗ convert(System, Σ₂)
end

function ⊗(Σ₁::ClassicalSystem, Σ₂::ClassicalSystem)
    Γ = Σ₁.Γ ⊕ Σ₂.Γ
    μ = [Σ₁.μ; Σ₂.μ]
    ClassicalSystem(Γ, μ)
end

function ⊗(Σ₁::System, Σ₂::System)
    R = Σ₁.R ⊕ Σ₂.R
    ϵ = Σ₁.ϵ ⊗ Σ₂.ϵ
    System(R, ϵ)
end

"""
    *(M::AbstractMatrix, Σ::AbstractSystem)

Let ``M`` be an ``n \\times m`` matrix, and let
``\\Sigma = (\\mathbb{R}^m, \\mathcal{E}, P)``. Then `M * Σ` computes the Gaussian system
``\\Sigma' = (\\mathbb{R}^n, \\mathcal{E}', P')``, where
```math
    \\mathcal{E}' = \\{ B \\in \\mathcal{B}(\\mathbb{R}^n) \\mid M^{-1}B \\in \\mathcal{E} \\}
```
and
```math
    P'(B) = P(M^{-1}B).
```
"""
*(M::AbstractMatrix, Σ::AbstractSystem)

function *(M::AbstractMatrix, Σ::ClassicalSystem)
    @assert size(M, 2) == length(Σ)
    Γ = M * Σ.Γ * M'
    μ = M * Σ.μ
    ClassicalSystem(Γ, μ)
end

function *(M::AbstractMatrix, Σ::System)
    @assert size(M, 2) == length(Σ)
    ιL, ιR = pushout(Σ.R, M)
    R = ιR
    ϵ = ιL * Σ.ϵ
    System(R, ϵ)
end

#TODO: Docstring
"""
    \\(M::AbstractMatrix, Σ::AbstractSystem)
"""
\(M::AbstractMatrix, Σ::AbstractSystem)

function \(M::AbstractMatrix, Σ::ClassicalSystem)
    @assert size(M, 1) == length(Σ)
    R = M
    ϵ = Σ
    System(R, ϵ)
end

function \(M::AbstractMatrix, Σ::System)
    @assert size(M, 1) == length(Σ)
    R = Σ.R * M
    ϵ = Σ.ϵ
    System(R, ϵ)
end

"""
    fiber(Σ::AbstractSystem)

Compute a basis for the fiber of ``Σ``.
"""
fiber(Σ::AbstractSystem)

function fiber(Σ::ClassicalSystem)
    falses(length(Σ), 0)
end

function fiber(Σ::System)
    nullspace(Σ.R)
end

"""
    dof(Σ::AbstractSystem)

Get the number of degrees of freedom of ``Σ``.
"""
dof(Σ::AbstractSystem) = length(fiber(Σ))

dof(::ClassicalSystem) = 0

"""
    mean(Σ::AbstractSystem)

Let ``Rw = \\epsilon`` be any kernel representation of ``\\Sigma``. Then `mean(Σ)` computes
a vector ``\\mu`` such that ``R\\mu`` is the mean of ``\\epsilon``.

In particular, if ``\\Sigma`` is a classical Gaussian system, then ``\\mu`` is the mean of ``\\Sigma``.
"""
mean(Σ::AbstractSystem)

function mean(Σ::ClassicalSystem)
    copy(Σ.μ)
end

function mean(Σ::System)
    solve_mean(Σ.ϵ.Γ, Σ.R', Σ.ϵ.μ)
end

"""
    cov(Σ::AbstractSystem)

Let ``Rw = \\epsilon`` be any kernel representation of ``\\Sigma``. Then `cov(Σ)` computes
a matrix ``\\Gamma`` such that ``R \\Gamma R^\\mathsf{T}`` is the covariance of ``\\epsilon``.

In particular, if ``\\Sigma`` is a classical Gaussian system, then ``\\Gamma`` is the
covariance of ``\\Sigma``.
"""
cov(Σ::AbstractSystem)

function cov(Σ::ClassicalSystem)
    copy(Σ.Γ)
end

function cov(Σ::System)
    solve_cov(Σ.ϵ.Γ, Σ.R')
end

#TODO: Docstring
"""
    oapply(composite::UndirectedWiringDiagram,
           box_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: AbstractSystem}
"""
function oapply(composite::UndirectedWiringDiagram,
                box_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: AbstractSystem}
    boxes = [box_map[x] for x in subpart(composite, :name)]
    oapply(composite, boxes)
end

#TODO: Docstring
"""
    oapply(composite::UndirectedWiringDiagram,
           boxes::AbstractVector{T}) where T <: AbstractSystem
"""
function oapply(composite::UndirectedWiringDiagram,
                boxes::AbstractVector{T}) where T <: AbstractSystem
    @assert nboxes(composite) == length(boxes)
    L = Bool[junction(composite, i; outer=false) == j
             for i in ports(composite; outer=false),
                 j in junctions(composite)]
    R = Bool[junction(composite, i; outer=true ) == j
             for i in ports(composite; outer=true ),
                 j in junctions(composite)]
    Σ = reduce(⊗, boxes; init=ClassicalSystem(Bool[]))
    R * (L \ Σ)
 end
