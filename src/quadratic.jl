"""
    QuadraticFunction(Q, a)

A convex quadratic function of the form
```math
f(x) = \\langle x, Qx + a \\rangle,
```
where `Q` is positive semidefinite.
"""
struct QuadraticFunction{T₁ <: AbstractMatrix, T₂ <: AbstractVector}
    Q::T₁
    a::T₂
end

"""
    QuadraticFunction(Q::UniformScaling, a::AbstractVector)

Construct the convex quadratic function 
```math
f(x) = \\langle x, Qx + a \\rangle,
```
where ``Q`` is positive semidefinite.
"""
function QuadraticFunction(Q::UniformScaling, a::AbstractVector)
    Q = Q(length(a))
    QuadraticFunction(Q, a)
end

"""
    QuadraticFunction(a::AbstractVector)

Construct the linear function
```math
f(x) = \\langle x, a \\rangle.
```
"""
function QuadraticFunction(a::AbstractVector)
    Q = 0I
    QuadraticFunction(Q, a)
end

"""
    QuadraticFunction(Q::AbstractMatrix)

Construct the convex quadratic function 
```math
f(x) = \\langle x, Qx \\rangle,
```
where ``Q`` is positive semidefinite.
"""
function QuadraticFunction(Q::AbstractMatrix)
    a = zeros(size(Q, 1))
    QuadraticFunction(Q, a)
end

length(f::QuadraticFunction) = length(f.a)

"""
    *(f::QuadraticFunction, M::Union{UniformScaling, AbstractMatrix})

Compute the inverse image of `f` under `M`, given by
```math
(fM)(y) = f(My).
```
"""
function *(f::QuadraticFunction, M::Union{UniformScaling, AbstractMatrix})
    Q = M' * f.Q * M
    a = M' * f.a
    QuadraticFunction(Q, a)
end

# Computes the *closure of* the image of f under M.
"""
    *(M::Union{UniformScaling, AbstractMatrix}, f::QuadraticFunction)

Compute the image of `f` under `M`, given by
```math
(Mf)(y) = \\inf \\{ f(x) \\mid y = Mx \\}.
```
Returns a quintuple ``(Q, a, \\alpha, B, b).`` If ``b \\neq 0``, then ``Mf = -\\infty``. Otherwise,
```math
(Mf)(y) =
\\begin{cases}
\\langle y, \\frac{1}{2}Qy + a \\rangle + \\alpha   & By = 0 \\\\
\\infty                                             & \\text{else}
\\end{cases}.
```
"""
function *(M::Union{UniformScaling, AbstractMatrix}, f::QuadraticFunction)
    M⁺ = pinv(M)
    Q⁺ = pinv((I - M⁺ * M) * f.Q * (I - M⁺ * M))

    Q = M⁺' * (I - f.Q * Q⁺) * f.Q * M⁺
    a = M⁺' * (I - f.Q * Q⁺) * f.a
    α = -1/2 * f.a' * Q⁺ * f.a
    B = I - M * M⁺
    b = (I - M⁺ * M) * (I - f.Q * Q⁺) * f.a
    (Q, a, α, B, b)
end

"""
    oplus(f₁::QuadraticFunction, f₂::QuadraticFunction)

Compute the direct sum of `f₁` and `f₂`, given by 
```math
(f_1 \\oplus f_2)(x, y) = f_1(x) + f_2(y).
```
"""
function oplus(f₁::QuadraticFunction, f₂::QuadraticFunction)
    Q = f₁.Q ⊕ f₂.Q
    a = [f₁.a; f₂.a]
    QuadraticFunction(Q, a)
end

⊕(f₁::QuadraticFunction, f₂::QuadraticFunction) = oplus(f₁, f₂)

"""
    OpenQuadraticFunction(L, R, f)

An open quadratic function is a decorated cospan
```math
\\left(
\\begin{aligned}
\\begin{CD}
m   @>L>> k @<R<< n
\\end{CD}
\\end{aligned},
\\enspace
\\begin{aligned}
f
\\end{aligned}
\\right),
```
where ``L``, ``R`` are matrices and ``f: \\mathbb{R}^k \\to \\mathbb{R}`` is a convex quadratic function. This data determines a convex bifunction
```math
F(x, y) = f^*(Ry - Lx),
```
where ``f^*`` is the convex conjuate of ``f``.
"""
struct OpenQuadraticFunction{T₁ <: AbstractMatrix, T₂ <: AbstractMatrix, T₃, T₄}
    L::T₁
    R::T₂
    f::QuadraticFunction{T₃, T₄}
end

"""
    OpenQuadraticFunction(L::AbstractMatrix, f::QuadraticFunction)

Construct the bifunction
```math
F(x, y) = f^*(y - Lx),
```
where ``f^*`` is the convex conjugate of ``f``.
"""
function OpenQuadraticFunction(L::AbstractMatrix, f::QuadraticFunction)
    n = length(f)
    R = Matrix(I, n, n)
    OpenQuadraticFunction(L, R, f)
end

"""
    OpenQuadraticFunction(L::AbstractMatrix)

Construct the indicator bifunction corresponding to `L`, given by
```math
F(x, y) =
\\begin{cases}
0       & Lx = y \\\\
\\infty & \\text{else}
\\end{cases}.
```
"""
function OpenQuadraticFunction(L::AbstractMatrix)
    n = size(L, 1)
    f = QuadraticFunction(zeros(n))
    OpenQuadraticFunction(L, f)
end

"""
    OpenQuadraticFunction(f::QuadraticFunction)

Construct the bifunction 
```math
F(*, y) = f^*(y),
```
where ``f^*`` is the convex conjugate of ``f``.
"""
function OpenQuadraticFunction(f::QuadraticFunction)
    n = length(f)
    L = zeros(n, 0)
    OpenQuadraticFunction(L, f)
end

"""
    conjugate(F::OpenQuadraticFunction)

Compute the convex conjugate of `F`.

Returns a quintuple ``(Q, a, \\alpha, B, b).`` If ``b \\neq 0``, then ``F^* = -\\infty``. Otherwise,
```math
F^*(x, y) =
\\begin{cases}
\\langle (x, y), \\frac{1}{2}Q(x, y) + a \\rangle + \\alpha & B(x, y) = 0 \\\\
\\infty                                                     & \\text{else}
\\end{cases}.
```
"""
conjugate(F::OpenQuadraticFunction) = [-F.L F.R]' * F.f

"""
    QuadDom(n)

The Euclidian space ``\\mathbb{R}^n``.
"""
struct QuadDom{T <: Integer}
    n::T
end

function pushout(L::AbstractMatrix, R::AbstractMatrix)
    n = size(L, 1)
    P = nullspace([L' -R'])'
    ιL = P[:, 1:n]
    ιR = P[:, n+1:end]
    (ιL, ιR)
end

@instance ThAbelianBicategoryRelations{QuadDom, OpenQuadraticFunction} begin
    
    """
        dom(F::OpenQuadraticFunction)

    The domain of a bifunction is the dimension of its first variable.
    """
    dom(F::OpenQuadraticFunction) = QuadDom(size(F.L, 2))

    """
        codom(F::OpenQuadraticFunction)

    The codomain of a bifunction is the dimension of its second variable.
    """
    codom(F::OpenQuadraticFunction) = QuadDom(size(F.R, 2))
 
    """
        mzero(::Type{QuadDom})

    Construct the Euclidean space ``\\mathbb{R}^0``.
    """
    mzero(::Type{QuadDom}) = QuadDom(0)

    """
        oplus(X::QuadDom, Y::QuadDom)

    Compute the direct sum of ``X`` and ``Y``.
    """
    oplus(X::QuadDom, Y::QuadDom) = QuadDom(X.n + Y.n)
   
    """
        dagger(F::OpenQuadraticFunction)

    Compute the dagger of `F`, given by
    ```math
    F^\\dagger(y, x) = F(x, y).
    ```
    """
    dagger(F::OpenQuadraticFunction) = OpenQuadraticFunction(F.R, F.L, F.f * -I)

    """
        compose(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction)
    
    Compute the product of `F₁` and `F₂`, given by
    ```math
    (F_2F_1)(x, z) = \\inf \\{ F_1(x, y) + F_2(y, z) \\mid y \\}.
    ```
    """
    function compose(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction)
        @assert codom(F₁) == dom(F₂)
        ιL, ιR = pushout(F₁.R, F₂.L)
        L = ιL * F₁.L
        R = ιR * F₂.R
        f = (F₁.f ⊕ F₂.f) * [ιL ιR]'
        OpenQuadraticFunction(L, R, f)
    end

    """
        oplus(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction)

    Compute the direct sum of `F₁` and `F₂`, given by
    ```math
    (F_1 \\oplus F_2)((x_1, x_2), (y_1, y_2)) = F_1(x_1, y_1) + F_2(x_2, y_2).
    ```
    """
    function oplus(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction)
        L = F₁.L ⊕ F₂.L
        R = F₁.R ⊕ F₂.R
        f = F₁.f ⊕ F₂.f
        OpenQuadraticFunction(L, R, f)
    end

    """
        meet(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction)

    Compute the sum of `F₁` and `F₂`, given by
    ```math
    (F_1 + F_2)(x, y) = F_1(x, y) + F_2(x, y).
    ```
    """
    meet(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction) = Δ(X) ⋅ (F₁ ⊕ F₂) ⋅ ∇(X)

    """
        join(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction)
    
    Compute the infimal convolution of `F₁`  and `F₂`, given by
    ```math
    (F_1 \\, \\Box \\, F_2)(x^*, y^*) = \\inf \\{ F_1(x^*, y^*) + F_2(x - x^*, y - y^*) \\mid x, y \\}.
    ```
    """
    join(F₁::OpenQuadraticFunction, F₂::OpenQuadraticFunction) = coplus(X) ⋅ (F₁ ⊕ F₂) ⋅ plus(X)

    """
        id(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F(x, y) =
    \\begin{cases}
    0       & x = y \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    function id(X::QuadDom)
        L = Matrix(I, X.n, X.n)
        OpenQuadraticFunction(L)
    end

    """
        zero(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F(*, y) =
    \\begin{cases}
    0       & y = 0 \\\\
    \\infty & \\text{else}
    \\end{cases}
    ```
    """
    function zero(X::QuadDom)
        L = zeros(X.n, 0)
        OpenQuadraticFunction(L)
    end

    """
        delete(X::QuadDom)

    Construct the indicator bifunction 
    ```math
    F(x, *) = 0.
    ```
    """
    function delete(X::QuadDom)
        L = zeros(0, X.n)
        OpenQuadraticFunction(L)
    end
    
    """
        mcopy(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F(x, (y_1, y_2)) =
    \\begin{cases}
    0       & x = y_1 = y_2 \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    function mcopy(X::QuadDom)
        L = [Matrix(I, X.n, X.n); Matrix(I, X.n, X.n)]
        OpenQuadraticFunction(L)
    end

    """
        plus(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F((x_1, x_2), y) =
    \\begin{cases}
    0       & x_1 + x_2 = y \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    function plus(X::QuadDom)
        L = [Matrix(I, X.n, X.n) Matrix(I, X.n, X.n)]
        OpenQuadraticFunction(L)
    end
    
    """
        dunit(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F(*, (y_1, y_2)) =
    \\begin{cases}
    0       & y_1 = y_2 \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    dunit(X::QuadDom) = □(X) ⋅ Δ(X)
    
    """
        cozero(X::QuadDom)
    
    Construct the indicator bifunction
    ```math
    F(x, *) =
    \\begin{cases}
    0       & x = 0 \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    cozero(X::QuadDom) = dagger(zero(X))

    """
        create(X::QuadDom)

    Construct the indicator bifunction 
    ```math
    F(*, y) = 0.
    ```
    """
    create(X::QuadDom) = dagger(delete(X))
    
    """
        mmerge(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F((x_1, x_2), y) =
    \\begin{cases}
    0       & x_1 = x_2 = y \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    mmerge(X::QuadDom) = dagger(mcopy(X))

    """
        coplus(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F((x_1, x_2), y) =
    \\begin{cases}
    0       & x_1 + x_2 = y \\\\
    \\infty & \\text{else}
    \\end{cases}
    ```
    """
    coplus(X::QuadDom) = dagger(plus(X))

    """
        dcounit(X::QuadDom)

    Construct the indicator bifunction
    ```math
    F((x_1, x_2), *) =
    \\begin{cases}
    0       & x_1 = x_2 \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    dcounit(X::QuadDom) = dagger(dunit(X)) 

    """
        swap(X::QuadDom, Y::QuadDom)
    
    Construct the indicator bifunction
    ```math
    F((x_1, x_2), (y_1, y_2)) =
    \\begin{cases}
    0       & x_1 = y_2, x_2 = y_1 \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    function swap(X::QuadDom, Y::QuadDom)
        L = [zeros(Y.n, X.n) Matrix(I, Y.n, Y.n); Matrix(I, X.n, X.n) zeros(X.n, Y.n)]
        OpenQuadraticFunction(L)
    end

    """
        top(X::QuadDom, Y::QuadDom)

    Construct the indicator bifunction 
    ``math
    F(x, y) = 0.
    ```
    """
    top(X::QuadDom, Y::QuadDom) = ◊(X) ⋅ □(Y)

    """
        bottom(X::QuadDom, Y::QuadDom)

    Construct the indicator bifunction
    ```math
    F(x, y) =
    \\begin{cases}
    0       & x = y = 0 \\\\
    \\infty & \\text{else}
    \\end{cases}.
    ```
    """
    bottom(X::QuadDom, Y::QuadDom) = cozero(X) ⋅ zero(Y)
end

"""
    oapply(composite::UndirectedWiringDiagram, hom_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: OpenQuadraticFunction}

Compose open quadratic functions according to an undirected wiring wiring diagram.
"""
function oapply(composite::UndirectedWiringDiagram, hom_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: OpenQuadraticFunction}
    cospans = [hom_map[x] for x in subpart(composite, :name)]
    oapply(composite, cospans)
end

"""
    oapply(composite::UndirectedWiringDiagram, cospans::AbstractVector{T}) where T <: OpenQuadraticFunction

Compose open quadratic functions according to an undirected wiring wiring diagram.
"""
function oapply(composite::UndirectedWiringDiagram, cospans::AbstractVector{T}) where T <: OpenQuadraticFunction
    @assert nboxes(composite) == length(cospans)
    l = FinFunction(subpart(composite, :junction), njunctions(composite))
    r = FinFunction(subpart(composite, :outer_junction), njunctions(composite))
    L = [l(i) == j for i in dom(l), j in codom(l)]
    R = [r(i) == j for i in dom(r), j in codom(r)]
    F = reduce(⊕, dunit(dom(F)) ⋅ (id(dom(F)) ⊕ F) for F in cospans)
    OpenQuadraticFunction(F.L, F.R * L, F.f) ⋅ OpenQuadraticFunction(R)
end
