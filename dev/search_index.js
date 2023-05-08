var documenterSearchIndex = {"docs":
[{"location":"api/#Library-Reference","page":"Library Reference","title":"Library Reference","text":"","category":"section"},{"location":"api/#Gaussian-Systems","page":"Library Reference","title":"Gaussian Systems","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"AbstractSystem\nClassicalSystem\nSystem\n\nClassicalSystem(::AbstractMatrix, ::AbstractVector)\nClassicalSystem(::AbstractMatrix)\nClassicalSystem(::AbstractVector)\nSystem(::AbstractMatrix, ::ClassicalSystem)\nSystem(::AbstractMatrix)\n\nlength(::AbstractSystem)\ndof(::AbstractSystem)\nfiber(::AbstractSystem)\nmean(::AbstractSystem)\ncov(::AbstractSystem)\n*(::AbstractMatrix, ::AbstractSystem)\n\\(::AbstractMatrix, ::AbstractSystem)\n⊗(::AbstractSystem, ::AbstractSystem)\noapply(composite::UndirectedWiringDiagram, hom_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: AbstractSystem}\noapply(composite::UndirectedWiringDiagram, boxes::AbstractVector{T}) where T <: AbstractSystem","category":"page"},{"location":"api/#AlgebraicInference.AbstractSystem","page":"Library Reference","title":"AlgebraicInference.AbstractSystem","text":"AbstractSystem\n\nAbstract type for Gaussian systems. \n\nSubtypes should support the following methods:\n\nlength(Σ::AbstractSystem)\nfiber(Σ::AbstractSystem)\nmean(Σ::AbstractSystem)\ncov(Σ::AbstractSystem)\n*(M::AbstractMatrix, Σ::AbstractSystem)\n\\(M::AbstractMatrix, Σ::AbstractSystem)\n⊗(Σ₁::AbstractSystem, Σ₂::AbstractSystem)\n\nReferences:\n\nJ. C. Willems, \"Open Stochastic Systems,\" in IEEE Transactions on Automatic Control,  vol. 58, no. 2, pp. 406-421, Feb. 2013, doi: 10.1109/TAC.2012.2210836.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.ClassicalSystem","page":"Library Reference","title":"AlgebraicInference.ClassicalSystem","text":"ClassicalSystem{T₁ <: AbstractMatrix, T₂ <: AbstractVector} <: AbstractSystem\n\nA classical Gaussian system.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.System","page":"Library Reference","title":"AlgebraicInference.System","text":"System{T₁ <: AbstractMatrix, T₂, T₃} <: AbstractSystem\n\nA Gaussian system.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.ClassicalSystem-Tuple{AbstractMatrix, AbstractVector}","page":"Library Reference","title":"AlgebraicInference.ClassicalSystem","text":"ClassicalSystem(Γ::AbstractMatrix, μ::AbstractVector)\n\nConstruct a classical Gaussian system with mean mu and covariance Gamma.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.ClassicalSystem-Tuple{AbstractMatrix}","page":"Library Reference","title":"AlgebraicInference.ClassicalSystem","text":"ClassicalSystem(Γ::AbstractMatrix)\n\nConstruct a classical Gaussian system with mean mathbf0 and covariance Gamma.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.ClassicalSystem-Tuple{AbstractVector}","page":"Library Reference","title":"AlgebraicInference.ClassicalSystem","text":"ClassicalSystem(μ::AbstractVector)\n\nConstruct a classical Gaussian system with mean mu and covariance mathbf0.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.System-Tuple{AbstractMatrix, ClassicalSystem}","page":"Library Reference","title":"AlgebraicInference.System","text":"System(R::AbstractMatrix, ϵ::ClassicalSystem)\n\nLet R be an m times n matrix, and let epsilon be an m-variate random vector with mean mu and covariance Gamma.\n\nIf mu in mathttimage(R  Gamma), then there exists a random variable hatw taking values in (mathbbR^n sigma R) that almost-surely solves the convex program\n\n    beginalign*\n        undersetwtextminimize   E(epsilon w) \n        textsubject to                Rw in mathttimage(Gamma) + epsilon\n    endalign*\n\nwhere\n\n    sigma R =  R^-1B mid B in mathcalB(mathbbR^m) \n\n\nand E(- w) is the negative log-density of the multivariate normal distribution mathcalN(Rw Gamma).\n\nIf mu in mathttimage(R  Gamma), then System(R, ϵ) constructs the Gaussian system Sigma = (mathbbR^n sigma R P), where P is the distribution of hatw.\n\nIn particular, if R has full row-rank, then Rw = epsilon is a kernel representation of Sigma.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.System-Tuple{AbstractMatrix}","page":"Library Reference","title":"AlgebraicInference.System","text":"System(R::AbstractMatrix)\n\nLet R be an m times n matrix. Then System(R) constructs the deterministic Gaussian system  Sigma = (mathbbR^n sigma R P) where\n\n    sigma R =  R^-1B mid B in mathcalB(mathbbR^m)\n\nand\n\n    P(R^-1B) = begincases\n        1  0 in B     \n        0  textelse\n    endcases\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.length-Tuple{AbstractSystem}","page":"Library Reference","title":"Base.length","text":"length(Σ::AbstractSystem)\n\nLet Sigma = (mathbbR^n mathcalE P). Then length(Σ) gets the dimension n.\n\n\n\n\n\n","category":"method"},{"location":"api/#StatsAPI.dof-Tuple{AbstractSystem}","page":"Library Reference","title":"StatsAPI.dof","text":"dof(Σ::AbstractSystem)\n\nGet the number of degrees of freedom of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.fiber-Tuple{AbstractSystem}","page":"Library Reference","title":"AlgebraicInference.fiber","text":"fiber(Σ::AbstractSystem)\n\nCompute a basis for the fiber of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.mean-Tuple{AbstractSystem}","page":"Library Reference","title":"Statistics.mean","text":"mean(Σ::AbstractSystem)\n\nLet Rw = epsilon be any kernel representation of Sigma. Then mean(Σ) computes a vector mu such that Rmu is the mean of epsilon.\n\nIn particular, if Sigma is a classical Gaussian system, then mu is the mean of Sigma.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.cov-Tuple{AbstractSystem}","page":"Library Reference","title":"Statistics.cov","text":"cov(Σ::AbstractSystem)\n\nLet Rw = epsilon be any kernel representation of Sigma. Then cov(Σ) computes a matrix Gamma such that R Gamma R^mathsfT is the covariance of epsilon.\n\nIn particular, if Sigma is a classical Gaussian system, then Gamma is the covariance of Sigma.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:*-Tuple{AbstractMatrix, AbstractSystem}","page":"Library Reference","title":"Base.:*","text":"*(M::AbstractMatrix, Σ::AbstractSystem)\n\nLet M be an n times m matrix, and let Sigma = (mathbbR^m mathcalE P). Then M * Σ computes the Gaussian system Sigma = (mathbbR^n mathcalE P), where\n\n    mathcalE =  B in mathcalB(mathbbR^n) mid M^-1B in mathcalE \n\nand\n\n    P(B) = P(M^-1B)\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:\\-Tuple{AbstractMatrix, AbstractSystem}","page":"Library Reference","title":"Base.:\\","text":"\\(M::AbstractMatrix, Σ::AbstractSystem)\n\n\n\n\n\n","category":"method"},{"location":"api/#Catlab.Theories.:⊗-Tuple{AbstractSystem, AbstractSystem}","page":"Library Reference","title":"Catlab.Theories.:⊗","text":"⊗(Σ₁::AbstractSystem, Σ₂::AbstractSystem)\n\nCompute the product Sigma_1 times Sigma_2.\n\n\n\n\n\n","category":"method"},{"location":"api/#Catlab.WiringDiagrams.WiringDiagramAlgebras.oapply-Union{Tuple{T₂}, Tuple{T₁}, Tuple{AbstractUWD, AbstractDict{T₁, T₂}}} where {T₁, T₂<:AbstractSystem}","page":"Library Reference","title":"Catlab.WiringDiagrams.WiringDiagramAlgebras.oapply","text":"oapply(composite::UndirectedWiringDiagram,\n       box_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: AbstractSystem}\n\n\n\n\n\n","category":"method"},{"location":"api/#Catlab.WiringDiagrams.WiringDiagramAlgebras.oapply-Union{Tuple{T}, Tuple{AbstractUWD, AbstractVector{T}}} where T<:AbstractSystem","page":"Library Reference","title":"Catlab.WiringDiagrams.WiringDiagramAlgebras.oapply","text":"oapply(composite::UndirectedWiringDiagram,\n       boxes::AbstractVector{T}) where T <: AbstractSystem\n\n\n\n\n\n","category":"method"},{"location":"api/#Valuations","page":"Library Reference","title":"Valuations","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"Valuation\nLabeledBox\n\nLabeledBox(::Any, ::OrderedSet)\n\nd(::Valuation)\n⊗(::Valuation, ::Valuation)\n↓(::Valuation, ::AbstractSet)\n-(::Valuation, ::Any)\n\nconstruct_inference_problem(::UndirectedWiringDiagram, ::AbstractDict)\nconstruct_inference_problem(::UndirectedWiringDiagram, ::AbstractVector)\nconstruct_elimination_sequence(::AbstractSet{T}, ::AbstractSet) where T <: AbstractSet\nfusion_algorithm(::AbstractSet{T}, ::Any) where T <: Valuation","category":"page"},{"location":"api/#AlgebraicInference.Valuation","page":"Library Reference","title":"AlgebraicInference.Valuation","text":"Valuation\n\nAbstract type for valuations.\n\nSubtypes should support the following methods:\n\nd(ϕ::Valuation)\n↓(ϕ::Valuation, x::AbstractSet)\n⊗(ϕ₁::Valuation, ϕ₂::Valuation)\n\nReferences:\n\nPouly, M.; Kohlas, J. Generic Inference. A Unified Theory for Automated Reasoning; Wiley: Hoboken, NJ, USA, 2011.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.LabeledBox","page":"Library Reference","title":"AlgebraicInference.LabeledBox","text":"LabeledBox{T₁, T₂} <: Valuation\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.LabeledBox-Tuple{Any, OrderedSet}","page":"Library Reference","title":"AlgebraicInference.LabeledBox","text":"LabeledBox(box, labels::OrderedSet)\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.d-Tuple{Valuation}","page":"Library Reference","title":"AlgebraicInference.d","text":"d(ϕ::Valuation)\n\nGet the domain of phi.\n\n\n\n\n\n","category":"method"},{"location":"api/#Catlab.Theories.:⊗-Tuple{Valuation, Valuation}","page":"Library Reference","title":"Catlab.Theories.:⊗","text":"⊗(ϕ₁::Valuation, ϕ₂::Valuation)\n\nPerform the combination phi_1 otimes phi_2.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.:↓-Tuple{Valuation, AbstractSet}","page":"Library Reference","title":"AlgebraicInference.:↓","text":"↓(ϕ::Valuation, x::AbstractSet)\n\nPerform the projection phi^downarrow x.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:--Tuple{Valuation, Any}","page":"Library Reference","title":"Base.:-","text":"-(ϕ::Valuation, X)\n\nPerform the variable elimination phi^-X.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.construct_inference_problem-Tuple{AbstractUWD, AbstractDict}","page":"Library Reference","title":"AlgebraicInference.construct_inference_problem","text":"construct_inference_problem(composite::UndirectedWiringDiagram,\n                            box_map::AbstractDict)\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.construct_inference_problem-Tuple{AbstractUWD, AbstractVector}","page":"Library Reference","title":"AlgebraicInference.construct_inference_problem","text":"construct_inference_problem(composite::UndirectedWiringDiagram,\n                            boxes::AbstractVector)\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.construct_elimination_sequence-Union{Tuple{T}, Tuple{AbstractSet{T}, AbstractSet}} where T<:AbstractSet","page":"Library Reference","title":"AlgebraicInference.construct_elimination_sequence","text":"construct_elimination_sequence(domains::AbstractSet{T},\n                               query::AbstractSet) where T <: AbstractSet\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.fusion_algorithm-Union{Tuple{T}, Tuple{AbstractSet{T}, Any}} where T<:Valuation","page":"Library Reference","title":"AlgebraicInference.fusion_algorithm","text":"fusion_algorithm(factors::AbstractSet{T},\n                 elimination_sequence) where T <: Valuation\n\nAn implementation of Shenoy's fusion algorithm (algorithm 3.1 in Generic Inference).\n\n\n\n\n\n","category":"method"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"EditURL = \"https://github.com/samuelsonric/AlgebraicInference.jl/blob/master/docs/literate/regression.jl\"","category":"page"},{"location":"generated/regression/#Linear-Regression","page":"Linear Regression","title":"Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"using AlgebraicInference\nusing Catlab, Catlab.Graphics, Catlab.Programs\nusing LinearAlgebra\nusing StatsPlots","category":"page"},{"location":"generated/regression/#Frequentist-Linear-Regression","page":"Linear Regression","title":"Frequentist Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Consider the Gauss-Markov linear model","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"    y = X beta + epsilon","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"where X is a n times m matrix, beta is an m times 1 vector, and epsilon is an n times 1 normally distributed random vector with mean mathbf0 and covariance W. If X has full column rank, then the best linear unbiased estimator for beta is the random vector","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"    hatbeta = X^+ (I - (Q W Q)^+ Q W)^mathsfT y","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"where X^+ is the Moore-Penrose psuedoinverse of X, and","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Q = I - X X^+","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"References:","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Albert, Arthur. \"The Gauss-Markov Theorem for Regression Models with Possibly Singular Covariances.\" SIAM Journal on Applied Mathematics, vol. 24, no. 2, 1973, pp. 182–87.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"X = [ 1 0\n      0 1\n      0 0 ]\n\nW = [ 1 0 0\n      0 1 0\n      0 0 1 ]\n\ny = [ 1\n      1\n      1 ]\n\nQ = I - X * pinv(X)\nβ̂ = pinv(X) * (I - pinv(Q * W * Q) * Q * W)' * y","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"To solve for hatbeta using AlgebraicInference.jl, we construct an undirected wiring diagram.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"diagram = @relation (a₁, a₂) begin\n    X(a₁, a₂, b₁, b₂, b₃)\n    +(b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃)\n    ϵ(c₁, c₂, c₃)\n    y(d₁, d₂, d₃)\nend\n\nto_graphviz(diagram;\n    box_labels         = :name,\n    implicit_junctions = true,\n)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Then we assign values to the boxes in diagram and compute the result.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"P = [ 1 0 0 1 0 0\n      0 1 0 0 1 0\n      0 0 1 0 0 1 ]\n\nhom_map = Dict(\n    :X => System([-X I]),\n    :+ => System([-P I]),\n    :ϵ => ClassicalSystem(W),\n    :y => ClassicalSystem(y),\n)\n\nβ̂ = mean(oapply(diagram, hom_map))","category":"page"},{"location":"generated/regression/#Bayesian-Linear-Regression","page":"Linear Regression","title":"Bayesian Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Let rho = mathcalN(m V) be our prior belief about beta. Then our posterior belief hatrho is a bivariate normal distribution with mean","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"  hatm = m - V X^mathsfT (X V X + W)^+ (X m - y)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"and covariance","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"  hatV = V - V X^mathsfT (X V X + W)^+ X V","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V = [ 1 0\n      0 1 ]\n\nm = [ 0\n      0 ]\n\nm̂ = m - V * X' * pinv(X * V * X' + W) * (X * m - y)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V̂ = V - V * X' * pinv(X * V * X' + W) * X * V","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"To solve for hatrho using AlgebraicInference.jl, we construct an undirected wiring diagram.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"diagram = @relation (a₁, a₂) begin\n    ρ(a₁, a₂)\n    X(a₁, a₂, b₁, b₂, b₃)\n    +(b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃)\n    ϵ(c₁, c₂, c₃)\n    y(d₁, d₂, d₃)\nend\n\nto_graphviz(diagram;\n    box_labels         = :name,\n    implicit_junctions = true,\n)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Then we assign values to the boxes in diagram and compute the result.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"hom_map = Dict(\n    :ρ => ClassicalSystem(V, m),\n    :X => System([-X I]),\n    :+ => System([-P I]),\n    :ϵ => ClassicalSystem(W),\n    :y => ClassicalSystem(y),\n)\n\nm̂ = mean(oapply(diagram, hom_map))","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V̂ = cov(oapply(diagram, hom_map))","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"covellipse!(m, V, aspect_ratio=:equal, label=\"prior\")\ncovellipse!(m̂, V̂, aspect_ratio=:equal, label=\"posterior\")","category":"page"},{"location":"#AlgebraicInference.jl","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"AlgebraicInference.jl is a library for compositional Bayesian inference. It builds on Catlab.jl.","category":"page"}]
}
