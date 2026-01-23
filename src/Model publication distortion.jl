cd(dirname(@__FILE__))
cd("..")

using Pkg
Pkg.activate(".")  # activate this project's environment
Pkg.instantiate()  # make sure all packages installed

using Random, IrrationalConstants, Format, Distributions, Interpolations, Base.Iterators, FastGaussQuadrature, Optim, LogExpFunctions, CSV, DataFrames, DataFramesMeta, ForwardDiff, LinearAlgebra, Roots, QuadGK, Statistics, 
       InverseFunctions, StatsAPI, StatsBase, StatsModels, RegressionTables, Unicode, CairoMakie, Makie, ExcelFiles, XLSX, RData, SpecialFunctions, ThreadsX, HCubature, KissSmoothing

const 𝒩 = Normal()
const z̄ = quantile(𝒩, .975)  # 1.96

@inline diffcdf(N,b,a) = cdf(N,b) - cdf(N,a)
@inline hr(d::Normal,x) = (t=(x-Distributions.location(d))/Distributions.scale(d)) > 1e4 ? (t+1/t)/Distributions.scale(d) : exp(logpdf(d,x) - logccdf(d,x))  # standard normal hazard ratio/inverse Mills ratio
@inline sqrt0(x::T) where {T} = x<0 ? zero(T) : sqrt(x)


#
# generalized t distribution: adds μ and σ parameters
#
struct GenT{T<:Real} <: ContinuousUnivariateDistribution
	μ::T; σ::T; ν::T

	lnσ::T
	tdist::TDist{T}  # underlying Student's t distribution

	GenT(μ::T, σ::T, ν::T) where {T<:Real} = new{T}(μ, σ, ν, log(σ), TDist{T}(ν))
end
Distributions.pdf(     d::GenT, x::Real) = pdf(     d.tdist, (x - d.μ) / d.σ) / d.σ
Distributions.logpdf(  d::GenT, x::Real) = logpdf(  d.tdist, (x - d.μ) / d.σ) - d.lnσ
Distributions.cdf(     d::GenT, x::Real) = cdf(     d.tdist, (x - d.μ) / d.σ)
Distributions.logcdf(  d::GenT, x::Real) = logcdf(  d.tdist, (x - d.μ) / d.σ)
Distributions.quantile(d::GenT, p::Real) = quantile(d.tdist, p) * d.σ + d.μ


# to parameterize an n-vector of probabilities summing to 1 with an unbounded (n-1)-vector, apply logistic transform to latter, then map to squared spherical coordinates
# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates, https://math.stackexchange.com/questions/2861449/parameterizations-of-the-unit-simplex-in-mathbbr3
function RⁿtoSimplex(q::AbstractVector{T}) where {T}
	if iszero(length(q))
		T[1]
	elseif isone(length(q))
		t = cospi(logistic(q[]))^2 |> (x -> isnan(x) ? zero(T) : x)
		T[t, 1-t]
	else
		p = Vector{T}(undef, length(q)+1)
		Πsin² = one(T)
		@inbounds for i ∈ eachindex(q)
			sin², cos² = (q[i] |> logistic |> sincospi).^2
			p[i] = Πsin² * cos²
			Πsin² *= sin²
		end
		p[end] = Πsin²
		replace!(p, NaN=>0)
	end
end
function SimplextoRⁿ(p::AbstractVector{T}) where {T}
	q = Vector{T}(undef, length(p)-1)
	sum = p[end]
	@inbounds for i ∈ reverse(eachindex(q))
		sum += p[i]
		q[i] = acos(√(p[i] / sum)) / π
	end
	q .= logit.(q)
end
InverseFunctions.inverse(::typeof(SimplextoRⁿ)) = RⁿtoSimplex

# transform to constrain parameters
get0(::Vector{T}) where {T} = T[]
put0(::Vector{T}) where {T} = T[0]  # constant 0
InverseFunctions.inverse(::typeof(get0)) = put0
get1(::Vector{T}) where {T} = T[]
put1(::Vector{T}) where {T} = T[1]  # constant 1
InverseFunctions.inverse(::typeof(get1)) = put1
get1000(::Vector{T}) where {T} = T[]
put1000(::Vector{T}) where {T} = T[1,0,0,0]  # constant 1,0,0,0
InverseFunctions.inverse(::typeof(get1000)) = put1000

# # transform to constrain pDFR to have pR=0
# get_pR0(v::Vector{T}) where {T} = v[1:3]
# put_pR0(v::Vector{T}) where {T} = T[v; 0]
# InverseFunctions.inverse(::typeof(get_pR0)) = put_pR0
# get_pF0(v::Vector{T}) where {T} = [v[1]; v[3:4]]
# put_pF0(v::Vector{T}) where {T} = [v[1]; 0; v[2:3]]
# InverseFunctions.inverse(::typeof(get_pF0)) = put_pF0
# get_pHR0(v::Vector{T}) where {T} = v[1:2]
# put_pHR0(v::Vector{T}) where {T} = T[v; 0; 0]
# InverseFunctions.inverse(::typeof(get_pHR0)) = put_pHR0
# get_pDR0(v::Vector{T}) where {T} = v[2:3]
# put_pDR0(v::Vector{T}) where {T} = T[0; v; 0]
# InverseFunctions.inverse(::typeof(get_pDR0)) = put_pDR0

# functions to map x <-> fill(x,k) for k=1,2,3,4
shared1(x) = [x[1]]
shared2(x) = [x[1]]
shared3(x) = [x[1]]
shared4(x) = [x[1]]
const shared = shared1, shared2, shared3, shared4
fill1(x) = fill(x[],1)
fill2(x) = fill(x[],2)
fill3(x) = fill(x[],3)
fill4(x) = fill(x[],4)
InverseFunctions.inverse(::typeof(shared1)) = fill1
InverseFunctions.inverse(::typeof(shared2)) = fill2
InverseFunctions.inverse(::typeof(shared3)) = fill3
InverseFunctions.inverse(::typeof(shared4)) = fill4

log1m(x) = log(x - 1)
expp1(x) = exp(x) + 1
InverseFunctions.inverse(::typeof(log1m)) = expp1

bcast = Broadcast.BroadcastFunction  # short-hand for forming the broadcasting version of a function, which works with InverseFunctions

# to speed multiple calls with only z varying, pre-compute objects that don't depend on z
function lnfZcondΩ_prep(ω; NLegendre, σ::Vector{T}, μₘ, σₘ) where {T}
	Z₀, W = gausslegendre(NLegendre)  # nodes and weights for Gauss-Legendre quadrature over [-1,1]
	Z₀ = Z₀*z̄; W = W*z̄

	lnf_z₀ᵢ = @. logpdf(𝒩, Z₀-ω)
	z̄divσ   = z̄/σ[]
	Z₀divσ  = Z₀/σ[]
	lnI_H   = @. logdiffcdf(𝒩, Z₀divσ+z̄divσ, Z₀divσ-z̄divσ)
	S_H     = @. ccdf(𝒩, z̄divσ+Z₀divσ) + ccdf(𝒩, z̄divσ-Z₀divσ)  # Pr[success per p-hack try | z₀], assumed = to researcher's mean expectation thereof
	_μₘ     = @. S_H * μₘ[]
	_σₘ     = @. S_H * σₘ[]
	μ̃ₘ      = @. _μₘ + _σₘ^2 * lnI_H
	F_insig = W' * (@. exp(logsumexp(logccdf(Normal(0,_σₘ),_μₘ-1), 
														       (_μₘ + .5_σₘ^2 * lnI_H) * lnI_H + logcdf(Normal(0,_σₘ),μ̃ₘ-1)) + lnf_z₀ᵢ))

	(Z₀ = Z₀, W = W,
		z̄divσ  = z̄divσ, Z₀divσ = Z₀divσ,
		lnf_z₀ᵢ = lnf_z₀ᵢ,
		F_insig = F_insig,
		lnf_Z₁condZ₀ = Vector{T}(undef, NLegendre)
  )
end

# compute log [f(z|ω)] & F(file drawer|ω)
function lnfZcondΩ(z, ω; modelabsz=false, NLegendre=50, pDFR, σ, μₘ, σₘ, o=lnfZcondΩ_prep(ω; NLegendre, σ, μₘ, σₘ))  # sometimes called with only σ a Dual(), don't know why
	zdivσ = z/σ[]

	# Pr[no p-hacking|z₀] & Pr[insig p-hack result|z₀], using math for partial first moment E[X^m|m>1] where m ~ 𝒩(μₘ,σₘ²) and x = Pr[|z₁ⱼ|<z̄|z₀] (p-hack failure on try j)
	function lnF_no_phack_and_insig_phack(z₀divσ)
		lnI_H = logdiffcdf(𝒩, z₀divσ+o.z̄divσ, z₀divσ-o.z̄divσ)
		S_H = ccdf(𝒩, o.z̄divσ+z₀divσ) + ccdf(𝒩, o.z̄divσ-z₀divσ)  # Pr[success per p-hack try | z₀], assumed = to researcher's mean expectation thereof
		_μₘ, _σₘ = S_H * μₘ[], S_H * σₘ[]
		μ̃ₘ = _μₘ + _σₘ^2 * lnI_H - 1
		COVₘ = -μ̃ₘ/_σₘ

		logcdf(𝒩,(1-_μₘ)/_σₘ),  # lnF_no_phack
		  logccdf(𝒩, COVₘ) + (_μₘ + .5*_σₘ^2 * lnI_H) * lnI_H  # lnF_insig_phack
	end
	
	# Compure Pr[z₁,₁, ..., z₁,₍ₘ₋₁₎ < z₁ₘ | z₀]. This times f(z₁ₘ|z₀)=𝒩(z₀,σ²) is the p-hacking distsribution (lnf_phack)
	# so lnf_phack() isn't quite the right name
	function lnf_phack(z₀divσ, abszdivσ)
		lnI_H = logdiffcdf(𝒩, z₀divσ+abszdivσ, z₀divσ-abszdivσ)
		S_H = ccdf(𝒩, o.z̄divσ+z₀divσ) + ccdf(𝒩, o.z̄divσ-z₀divσ)  # Pr[success per p-hack try | z₀], assumed = to researcher's mean expectation thereof
		_μₘ, _σₘ = S_H * μₘ[], S_H * σₘ[]
		μ̃ₘ = _μₘ + _σₘ^2 * lnI_H - 1
		COVₘ = -μ̃ₘ/_σₘ

		# hazard ratio-based expression, log(hr(𝒩, COVₘ) * _σₘ + μ̃ₘ + 1)
		logccdf_𝒩_COVₘ = logccdf(𝒩,COVₘ)
		if COVₘ < -1e3
			loghr_exp = 1 + _σₘ / COVₘ  # for x->-∞, log(hr(𝒩, COVₘ) * _σₘ + μ̃ₘ + 1) -> 1 +_σₘ/COVₘ
		else
			if COVₘ < 1e4
				loghr_exp = exp(logpdf(𝒩,COVₘ) - logccdf_𝒩_COVₘ)
			else
				loghr_exp = COVₘ + 1/ COVₘ  # for large x, hr ≈ x + 1/x
			end
			loghr_exp = log(loghr_exp * _σₘ + μ̃ₘ + 1)
		end

		(_μₘ-1 + .5_σₘ^2 * lnI_H) * lnI_H + logccdf(𝒩, COVₘ) + loghr_exp
	end

	if	!(-eps() < z < eps())
		modelabsz && (neg2zdivσ = -2zdivσ)

		@inbounds for k ∈ 1:(NLegendre + 1) ÷ 2  # for each non-positive z₀ quadrature point (most calculations symmetric in z₀, and all if modelabsz)
			lnf_Z₁condZ₀ₖ = log(o.W[k]) + lnf_phack(o.Z₀divσ[k], abs(zdivσ))

			# component asymmetric in z₀
			o.lnf_Z₁condZ₀[k] = logpdf(𝒩, o.Z₀divσ[k]-zdivσ) + lnf_Z₁condZ₀ₖ
			if modelabsz
				o.lnf_Z₁condZ₀[k] += log1pexp(o.Z₀divσ[k] * neg2zdivσ)  # log [ϕ(a-b) + ϕ(a+b)] = log[ϕ(a-b)] + log[1+exp(-2ab)]
				o.lnf_Z₁condZ₀[end+1-k] = o.lnf_Z₁condZ₀[k]
			else
				o.lnf_Z₁condZ₀[end+1-k] = logpdf(𝒩, o.Z₀divσ[k]+zdivσ) + lnf_Z₁condZ₀ₖ  # compute for -z₀ as well as z₀
			end
		end
	end

	lnf_z₀ᵢⱼ = modelabsz ? logpdf(𝒩, ω-z) + log1pexp(log(-2ω*z)) : logpdf(𝒩, ω-z)  # log [ϕ(a-b) + ϕ(a+b)] = log[ϕ(a-b)] + log[1+exp(-2ab)]

	pD, pF, pR = pDFR

	if -z̄ ≤ z ≤ z̄
		lnF_no_phack, lnF_insig_phack = lnF_no_phack_and_insig_phack(zdivσ)
		lnF_no_sig_phack = pR < eps() ? log(pD) + lnF_no_phack : logsumexp(log(pR+pD) + lnF_no_phack, log(pR) + lnF_insig_phack)
	end

	if -eps() ≤ z ≤ eps()
		∫ = lnF_no_sig_phack + lnf_z₀ᵢⱼ
	else
		lnf_Z₁ = @inbounds logsumexp(o.lnf_Z₁condZ₀[k] + o.lnf_z₀ᵢ[k] for k ∈ 1:NLegendre) - log(σ[])  # constant left out of lnf_Z₁condZ₀ for speed
		∫ = -z̄ ≤ z ≤ z̄ ? 
				  pD < eps() ? 
						          lnF_no_sig_phack + lnf_z₀ᵢⱼ :
						logsumexp(lnF_no_sig_phack + lnf_z₀ᵢⱼ, log(pD) + lnf_Z₁) :
					logsumexp(                     lnf_z₀ᵢⱼ,           lnf_Z₁)
	end

	∫, pF * o.F_insig
end

# compute f(z|ω) & F(file drawer|ω)
_fZcondΩ(args...; kwargs...) = lnfZcondΩ(args...; kwargs...) |> x->(exp(x[1]), x[2])

 # f(z|ω). If truncate=true (the default), returns the density conditional on publication
fZcondΩ(z, ω; modelabsz=false, NLegendre=50, pDFR, σ, μₘ, σₘ, truncate=true) = _fZcondΩ(z, ω; modelabsz, NLegendre, pDFR, σ, μₘ, σₘ) |> (y -> truncate ? y[1]/(1 - y[2]) : y[1])


# cdf of z|ω
function FZcondΩ(z, ω; rtol=.001, order=13, pDFR, modelabsz=false, NLegendre=50, kwargs...)
	# println("Entering FZcondΩz=$z ω=$ω rtol=$rtol order=$order pDFR=$pDFR kwargs=$kwargs modelabsz=$modelabsz NLegendre=$NLegendre")
	o = lnfZcondΩ_prep(ω; NLegendre, kwargs...)
	endpoints = modelabsz ? [0, z̄] : [-Inf, -z̄, z̄]  # since f(z|ω) jumps at ±z̄, do quadrature separately in each range
	endpoints = [endpoints[findall(<(z), endpoints)]; z]
	quadgk(_z->exp(lnfZcondΩ(_z,ω; NLegendre, o, pDFR, kwargs...)[1]), endpoints...; rtol, order)[1] / (1 - pDFR[2] * o.F_insig)
end

quantFcondΩ(q, ω; kwargs...) = find_zero(z -> q - FZcondΩ(z, ω; kwargs...), (-20,20), Roots.ITP())  # ITP algorithm works well

# likelihood for a collection (vector, step range) of z's for plotting
# If truncate=true (default), returns the truncated density, i.e., conditional on publication
function fZ(z; modelabsz=false, NHermite=50, NLegendre=50, p, μ, τ, ν, pDFR, σ, μₘ, σₘ, truncate=true)
  M = HnFmodel(z; d=length(τ), NHermite, NLegendre, modelabsz)
  ∫, G = _HnFll(M; p,μ,τ,ν,pDFR,σ,μₘ, σₘ)
	∫ .= exp.(∫)
  truncate && (∫ ./= 1 - G)
  ∫
end

# f(z), f(ω), f(ω|z), E[ω|z]
# inconsistency: z should be a scalar for fΩcondZ but a vector or other iterable for EΩcondZ
@inline fΩ(ω; p, μ, τ, ν) = p'pdf.(GenT.(μ,τ,ν), ω)
@inline lnfΩ(ω; p, μ, τ, ν) = @inbounds logsumexp(log(p[i]) + logpdf(GenT(μ[],τ[i],ν[i]), ω) for i ∈ eachindex(p))
@inline fZ₀condΩ(z₀,ω) = pdf(𝒩,z₀-ω)
@inline lnfZ₀condΩ(z₀,ω) = logpdf(𝒩,z₀-ω)
fΩcondZ(ω, z; p, μ, τ, ν, NHermite=50, NLegendre=50, kwargs...) = fZcondΩ(z, ω; NLegendre, kwargs..., truncate=false) * fΩ(ω; p, μ, τ, ν) / fZ([z]; p, μ, τ, ν, kwargs..., NLegendre, NHermite, truncate=false)[]
EΩcondZ(z; rtol=.00001, maxevals=1e4, p, μ, τ, ν, NHermite=50, NLegendre=50, kwargs...) = [quadgk(ω -> ω * fZcondΩ(zᵢ, ω; kwargs..., NLegendre, truncate=false) * fΩ(ω; p, μ, τ, ν), -20, 20; rtol, maxevals)[1] for zᵢ∈z] ./ 
                                                                      fZ(z; p, μ, τ, ν, kwargs..., NLegendre, NHermite, truncate=false)


# CIs
Cquant(α, z; kwargs...) = find_zero(ω -> α - FZcondΩ(z, ω; kwargs...), (-20,20), Roots.ITP())  # Andrews & Kasy (2019), eq. 2
CI(    α, z; kwargs...) = Cquant(α/2, z; kwargs...), Cquant(1-α/2, z; kwargs...)


# object to hold pre-computed stuff for hack'n'file log likelihood computation
struct HnFmodel
	modelabsz::Bool  # modeling |z|?
	d::Vector{Int}  # number of mixture components; scalar stored as mutable vector
	z::Vector{Float64}  # all data
	wt::Vector{Float64}  # observation weights
	N::Int  # number of z's in data
	insig::BitVector  # which z's are in insignificant region
	approxzero::BitVector 	# which z's are basically zero
	NHermite::Int  # number of quadrature points for integration over z₀ to compute f(z₀)
	Ω::Vector{Float64}; WHermite::Vector{Float64}; lnWpΩ²::Vector{Float64}  # quadrature nodes & weights
	NLegendre::Int  # number of quadrature points
	Z₀::Vector{Float64}; WLegendre::Vector{Float64}; lnWLegendre::Vector{Float64}  # quadrature nodes & weights
  penalty::Function
	lnf_z₀_ikdict::Dict{DataType, Vector}  # collections of pre-allocated arrays for use in likelihood computation, separate for Float64, ForwardDiff.Dual, etc.
	Z₀divσdict::Dict{DataType, Vector}  # collections of pre-allocated arrays for use in likelihood computation, separate for Float64, ForwardDiff.Dual, etc.
	lnF_no_sig_phackdict::Dict{DataType, Vector}
	f_Z₁condZ₀dict::Dict{DataType, Matrix}
	F_insig_Z₀dict::Dict{DataType, Vector}
	∫dict::Dict{Tuple{DataType, Int64}, Matrix}

	function HnFmodel(z, wt=Float64[]; d::Int, modelabsz=false, NHermite=50, NLegendre=50, penalty::Function=(; kwargs...)->0.)
		Ω, WHermite = gausshermite(NHermite, normalize=true)

		Z₀, W = gausslegendre(NLegendre)  # nodes and weights for Gauss-Legendre quadrature over [-1,1]
		Z₀ .*= z̄; W .*= z̄  # change of variables to quadrature over [-z̄, z̄]
		
		new(modelabsz, [d], z, wt/mean(wt), length(z), -z̄.≤z.≤z̄, -eps().<z.<eps(), NHermite, Ω, WHermite, log.(WHermite).+.5Ω.^2, NLegendre, Z₀, W, log.(W), penalty, Dict(), Dict(), Dict(), Dict(), Dict(), Dict())
	end
end

# to prevent "MethodError: ==(::ForwardDiff.Dual{ForwardDiff.Tag{var"#objective#178"{…}, Float64}, Float64, 11}, ::IrrationalConstants.Invsqrt2) is ambiguous."
import Base.==
==(a::ForwardDiff.Dual, b::IrrationalConstants.Invsqrt2) = a == Float64(b)


#
# Hack'n'file log likelihood
#

# Compute observation-level likelihood (not log likelihood), file-drawer mass, and expected fraction of initially insignificant results
function _HnFll(M::HnFmodel; p::AbstractVector{T}, μ::AbstractVector{T}, τ::AbstractVector{T}, ν::AbstractVector{T}, pDFR::AbstractVector, σ::Vector, μₘ::Vector, σₘ::Vector) where {T}
  pD, pF, pR = pDFR

	is = findall(>(1e-6), p)  # ~nonzero mixture components
	_d = length(is)

	# fetch previoiusly allocated objects of needfed element type, or allocate if needed
	∫                 = get!(                 M.∫dict, (T,_d), Matrix{T}(undef,M.N,_d))::Matrix{T}
	lnf_Z₁condZ₀      = get!(        M.f_Z₁condZ₀dict,  T    , Matrix{T}(undef,M.N,M.NLegendre))::Matrix{T}
	lnf_z₀ᵢ           = get!(         M.lnf_z₀_ikdict,  T    , Vector{T}(undef, M.NLegendre))::Vector{T}
	Z₀divσ            = get!(            M.Z₀divσdict,  T    , Vector{T}(undef, M.NLegendre))::Vector{T}
	lnF_no_sig_phack  = get!(  M.lnF_no_sig_phackdict,  T    , Vector{T}(undef, M.N))::Vector{T}
	F_insig_Z₀        = get!(        M.F_insig_Z₀dict,  T    , Vector{T}(undef, M.NLegendre))::Vector{T}

	@. Z₀divσ = M.Z₀ / σ[]; z̄divσ = z̄ / σ[]

	# Pr[no p-hacking|z₀] & Pr[insig p-hack result|z₀], using math for partial first moment E[X^m|m>1] where m ~ 𝒩(μₘ,σₘ²) and x = Pr[|z₁ⱼ|<z̄|z₀] (p-hack failure on try j)
	function lnF_no_phack_and_insig_phack(z₀divσ)
		lnI_H = logdiffcdf(𝒩, z₀divσ+z̄divσ, z₀divσ-z̄divσ)
		S_H = ccdf(𝒩, z₀divσ+z̄divσ) + cdf(𝒩, z₀divσ-z̄divσ)  # Pr[success per p-hack try | z₀], assumed = to researcher's mean expectation thereof
		_μₘ, _σₘ = S_H * μₘ[], S_H * σₘ[]
		COVₘ = -μₘ[]/σₘ[] - _σₘ * lnI_H + 1/_σₘ  # = -μ̃ₘ/_σₘ, with μ̃ₘ = _μₘ + _σₘ^2 * lnI_H - 1; writing it this way reduces NaN in Hessian

		return logcdf(𝒩, 1/_σₘ-μₘ[]/σₘ[]),  # lnF_no_phack; argument is (1-_μₘ)/_σₘ; writing it this way reduces NaN in Hessian
		       logccdf(𝒩, COVₘ) + (_μₘ + .5*_σₘ^2 * lnI_H) * lnI_H  # lnF_insig_phack
	end
	
	# Compure Pr[z₁,₁, ..., z₁,₍ₘ₋₁₎ < z₁ₘ | z₀]. This times f(z₁ₘ|z₀)=𝒩(z₀,σ²) is the p-hacking distsribution
	# so lnf_phack() isn't quite the right name
	S_H_z₀	 = ccdf.(𝒩, Z₀divσ .+ z̄divσ) .+ cdf(𝒩, Z₀divσ .- z̄divσ)  # Pr[success per p-hack try | z₀], assumed = to researcher's mean expectation thereof
	_μₘ_z₀, _σₘ_z₀ = S_H_z₀ * μₘ[], S_H_z₀ * σₘ[]

	@inbounds function lnf_phack(k, abszdivσ)
		lnI_H = logdiffcdf(𝒩, Z₀divσ[k]+abszdivσ, Z₀divσ[k]-abszdivσ)
		μ̃ₘ = _μₘ_z₀[k] + _σₘ_z₀[k]^2 * lnI_H - 1
		COVₘ = -μₘ[]/σₘ[] - _σₘ_z₀[k] * lnI_H + 1/_σₘ_z₀[k]  # = -μ̃ₘ/_σₘ; writing it this way may reduce NaNs in Hessians

		# hazard ratio-based expression, log(hr(𝒩, COVₘ) * _σₘ + μ̃ₘ + 1)
		logccdf_𝒩_COVₘ = logccdf(𝒩,COVₘ)
		if COVₘ < -1e3
			loghr_exp = 1 + _σₘ_z₀[k] / COVₘ  # for x->-∞, log(hr(𝒩, COVₘ) * _σₘ + μ̃ₘ + 1) -> 1 +_σₘ/COVₘ
		else
			if COVₘ < 1e4
				loghr_exp = exp(logpdf(𝒩,COVₘ) - logccdf_𝒩_COVₘ)
			else
				loghr_exp = COVₘ + 1 / COVₘ  # for largish x, hr ≈ x + 1/x
			end
			loghr_exp = log(loghr_exp * _σₘ_z₀[k] + μ̃ₘ + 1)
		end

		return (_μₘ_z₀[k]-1 + .5_σₘ_z₀[k]^2 * lnI_H) * lnI_H + logccdf_𝒩_COVₘ + loghr_exp
	end
	
	# f(z₁|z₀) to be convolved with f(z₀) later using Legendre quadrature; already includes the Legendre weights
	Threads.@threads for j ∈ eachindex(M.z)
		@inbounds begin
			z = M.z[j]; zdivσ = z/σ[]

			if !M.approxzero[j]
				M.modelabsz && (neg2zdivσ = -2zdivσ)
				for k ∈ 1:(M.NLegendre + 1) ÷ 2  # for each non-positive z₀ quadrature point (most calculations symmetric in z₀, and all if modelabsz)
					lnf_Z₁condZ₀ⱼₖ = M.lnWLegendre[k] + lnf_phack(k, abs(zdivσ))

					# sole component asymmetric in z₀ (if !modelabsz); NaN here means f(z₁|z₀)=0 within achieved numerical precision
					lnf_Z₁condZ₀[j,k] = logpdf(𝒩, Z₀divσ[k]-zdivσ) + lnf_Z₁condZ₀ⱼₖ
					if M.modelabsz
						lnf_Z₁condZ₀[j,k] += log1pexp(Z₀divσ[k] * neg2zdivσ)  # log [ϕ(a-b) + ϕ(a+b)] = log[ϕ(a-b)] + log[1+exp(-2ab)]
						lnf_Z₁condZ₀[j,end+1-k] = lnf_Z₁condZ₀[j,k]  # same for ±z₀
					else
						lnf_Z₁condZ₀[j,end+1-k] = logpdf(𝒩, Z₀divσ[k]+zdivσ) + lnf_Z₁condZ₀ⱼₖ  # compute for -z₀ as well as z₀
					end
				end
			end
 
			if M.insig[j]
				lnF_no_phack, lnF_insig_phack = lnF_no_phack_and_insig_phack(zdivσ)
				lnF_no_sig_phack[j] = pR < eps() ? log(pD) + lnF_no_phack : logsumexp(log(pR+pD) + lnF_no_phack, log(pR) + lnF_insig_phack)
			end
		end
	end

	@inbounds for k ∈ eachindex(M.Z₀)
		lnF_no_phack, lnF_insig_phack = lnF_no_phack_and_insig_phack(Z₀divσ[k])
		F_insig_Z₀[k] = M.WLegendre[k] * (exp(lnF_no_phack) + exp(lnF_insig_phack))
	end

	I₀ = F_insig = zero(T)  # accumulators for expected number of initially insig results, and number of publish/file-drawer/p-hack decision junctures
	@inbounds for _i ∈ 1:_d  # iterate over non~zero mixture components
		i = is[_i]

		# f(z_0) for ith mixture component, integrating out ω with Gauss-Hermite quadrature
		# because this is an inner loop, economize by manually computing the log t pdf while avoiding redundant work
		τᵢ² = τ[i]^2; _τᵢ² = 1+1/τᵢ²; sqrt_τᵢ² = √_τᵢ²
		halfinv_τᵢ² = .5 / _τᵢ²
		_νᵢ = ν[i]/2 + .5
		D = (1 + τ[i]^2) * ν[i]
		Cᵢ = log(p[i]) - logbeta(ν[i]/2,.5) - .5log(D)  # contains constant factor in t pdf, in logs
		lnf_z₀_i(z₀) = logsumexp(begin  # ln [∫_(-∞)^∞ ϕ(z₀;ω)t(ω;μ,τᵢ²,νᵢ)dω] sans ln Cᵢ factor
																d = (z₀ - μ[]) / sqrt_τᵢ²
																lnwpx² - halfinv_τᵢ² * (x - d / τᵢ²)^2 - log1p((x + d)^2 / D) * _νᵢ
														 end
														 for (x,lnwpx²) ∈ zip(M.Ω, M.lnWpΩ²))

		I₀ᵢ = F_insigᵢ = zero(T)
		@inbounds for k ∈ eachindex(M.Z₀)  # for each z₀ quadrature point
			lnf_z₀ᵢ[k] = lnf_z₀ᵢₖ = lnf_z₀_i(M.Z₀[k])
			I₀ᵢ += exp(M.lnWLegendre[k] + lnf_z₀ᵢₖ)
			F_insigᵢ += F_insig_Z₀[k] * exp(lnf_z₀ᵢₖ)
		end
		F_insig += exp(Cᵢ) * F_insigᵢ
		I₀ += exp(Cᵢ) * I₀ᵢ

		Threads.@threads for j ∈ eachindex(M.z)  # for each z value/interpolation point
			@inbounds begin
				z = M.z[j]

				lnf_z₀ᵢⱼ = M.modelabsz ? logsumexp(lnf_z₀_i(z), lnf_z₀_i(-z)) : lnf_z₀_i(z)
				if M.approxzero[j]
					∫[j,_i] = Cᵢ + lnF_no_sig_phack[j] + lnf_z₀ᵢⱼ
				else
					try
						lnf_Z₁ = logsumexp(lnf_Z₁condZ₀[j,k] + lnf_z₀ᵢ[k] for k ∈ eachindex(M.Z₀) if !isnan(lnf_Z₁condZ₀[j,k])) - log(σ[])  # log(σ[]) term left out of lnf_Z₁condZ₀ for speed
						∫[j,_i] = Cᵢ + (M.insig[j] ?
															pD < eps() ?
																          lnF_no_sig_phack[j] + lnf_z₀ᵢⱼ
															:
																logsumexp(lnF_no_sig_phack[j] + lnf_z₀ᵢⱼ, log(pD) + lnf_Z₁)
														:
															logsumexp(                        lnf_z₀ᵢⱼ,           lnf_Z₁)
													 )
					catch _  # logsumexp will fail if _all_ lnf_Z₁condZ₀[j,k] are NaN, which would be failure of Legendre quadrature
						∫[j,_i] = Cᵢ + (M.insig[j] ? pD < eps() ? -Inf :   # if σₘ is extremely small, estimated Pr(z₁|z₀) could be computed as exactly 0 for all integration points z₀; if pD=0 too, then estimated Pr[z]=0, and loglik evaluator has failed
																											 lnF_no_sig_phack[j] + lnf_z₀ᵢⱼ :
																				 lnf_z₀ᵢⱼ)
					end
				end
			end
		end
	end
  logsumexp!(lnF_no_sig_phack, ∫), pF * F_insig, I₀  # sum across mixture components, into `lnF_no_sig_phack` because it's the right size and already allocated
end

# returns negative of penalized log likelihood
function HnFll(M::HnFmodel; pDFR, kwargs...)
	∫, G, I₀ = _HnFll(M; pDFR, kwargs...)
	(iszero(length(M.wt)) ? ThreadsX.sum(∫) : dot(M.wt,∫)) - xlog1py(M.N, -G) + M.penalty(; pDFR, file_drawer_insig = G/I₀, kwargs...)
end


# simulate hack'n'file data generating process with integer m
# returns named tuple of true z's (ω), initial measurements (z), and reported results
# NaN = file-drawered
# if truncate=true (the default), restricts all return results to published studies
function HnFDGP(N::Int; p::Vector{Float64}, μ::Vector{Float64}=[0.], τ::Vector{Float64}, ν::Vector{Float64}, pDFR::Vector{Float64}, σ::Vector{Float64}, μₘ::Vector{Float64}, σₘ::Vector{Float64}, modelabsz::Bool=false, truncate::Bool=true)
	ω = Vector{Float64}(undef,N)
	z₀ = similar(ω)
	z₁ = fill(NaN,N)
	m = zeros(Int,N)
	z = similar(ω)
	Tμτν = GenT.(μ, τ, ν)

	Threads.@threads for i ∈ eachindex(z₀)  # for each simulated study
		@inbounds begin
			j = rand(Distributions.Categorical(p))  # pick mixture component
			ω[i] = ωᵢ = rand(Tμτν[j])
			z₀[i] = z₀ᵢ = ωᵢ + rand(𝒩)  # initial measurement, variance 1 around ω

			if abs(z₀ᵢ) > z̄  # if initial result significant, publish as is
				z[i] = z₀ᵢ
			else
				S_H = 1 - diffcdf(Normal(z₀ᵢ, σ[]),z̄,-z̄)  # Pr[p-hack success per try | z₀]
				m[i] = mᵢ = floor(Int, rand(S_H * Normal(μₘ[], σₘ[])))  # number of measurements to be taken if p-hacking
				if mᵢ<1	 # no p-hacking
					z₁[i] = z₁ᵢ = z₀ᵢ
				else
					batch = rand(Normal(z₀ᵢ, σ[]), mᵢ)  # m measurements
					absbatch = abs.(batch)
					j = findfirst(==(maximum(absbatch)), absbatch)  # most significant of batch
					z₁[i] = z₁ᵢ = batch[j]
					if z₁ᵢ < -z̄ || z̄ < z₁ᵢ  # if significant, publish and stop
						z[i] = z₁ᵢ
						continue
					end
				end
				r = rand(Distributions.Categorical(pDFR))
				z[i] = r==1 ? z₁ᵢ : r==2 ? NaN : z₀ᵢ  # publish final, significant result; file-drawer; publish initial, insignificant result
			end
			modelabsz && (z[i] = abs(z[i]))
		end
	end

	if truncate
		keep = @. !isnan(z) # && abs(z)<10
		ω, z₀, m, z₁, z  = ω[keep], z₀[keep], m[keep], z₁[keep], z[keep]
	end
	(ω=ω, z₀=z₀, m=m, z₁=z₁, z=z)
end

@kwdef mutable struct HnFresult<:RegressionModel
	estname::String
	modelabsz::Bool
	converged::Bool
	coefdict::NamedTuple
	coefnames::Vector{String}
	coef::Vector{Float64}
	vcov::Matrix{Float64}
	k::Int  # number of dof consumed
	n::Int  # sample size
	d::Int  # number of mixture components possibly net of deletion of trivial ones
	ll::Float64
	BIC::Float64 =  k*log(n)-2ll
	se::Vector{Float64} = sqrt0.(diag(vcov))
	z::Vector{Float64} = coef ./ se
	𝒩::Vector{Union{Missing, Normal{Float64}}} = [isnan(s) ? missing : Normal(c,s) for (c,s) ∈ zip(coef,se)]
end


#
# Setup to report HnFresult's with RegressionTables.jl. A lot of work!
#
begin
	# StatsAPI.aic( R::HnFresult) = 2 * (R.k − R.ll)
	# StatsAPI.aicc(R::HnFresult) = 2 * (R.k + R.k * (R.k − 1) / (R.n − R.k − 1) − R.ll)
	StatsAPI.bic( R::HnFresult) = R.k * log(R.n) − 2R.ll
	StatsAPI.coef(R::HnFresult) = R.coef
	StatsAPI.coefnames(R::HnFresult) = R.coefnames
	# StatsAPI.confint(R::HnFresult; level::Real=0.95) = [quantile.(R.𝒩, (1-level)/2) cquantile.(R.𝒩, (1-level)/2)]
	# StatsAPI.coeftable(R::HnFresult; level::Real=0.95) = (CI = confint(R; level);
	# 					                                             CoefTable([R.coef, 
	# 																											            R.se, 
	# 																																  R.z,
	# 																																  2ccdf.(𝒩, abs.(R.z)), 
	# 																																  eachcol(CI)...],
	# 																											           ["Estimate", "Std.Error", "z value", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
	# 																											           R.coefnames,
	# 																											           4,
	# 																											           3))
	StatsAPI.dof(R::HnFresult) = R.k
	# StatsAPI.informationmatrix(R::HnFresult; expected::Bool = true) = 
	StatsAPI.isfitted(R::HnFresult) = true
	StatsAPI.islinear(R::HnFresult) = false
	# StatsAPI.loglikelihood(model::HnFresult, observation) = 
	StatsAPI.loglikelihood(R::HnFresult) = R.ll
	StatsAPI.nobs(R::HnFresult) = R.n
	StatsAPI.vcov(R::HnFresult) = R.vcov
	StatsAPI.weights(R::HnFresult) = UnitWeights(R.n)
	StatsAPI.dof_residual(R::HnFresult) = R.n - R.k
	# StatsAPI.fitted(R::HnFresult) = 
	StatsAPI.responsename(R::HnFresult) = R.estname
	# StatsModels.formula(R::HnFresult) = Term(R.estname) ~ sum(Term.(R.coefnames))

	RegressionTables._responsename(x::HnFresult) = StatsAPI.responsename(x)
	RegressionTables._coefnames(x::HnFresult) = coefnames(x)
	RegressionTables.default_print_control_indicator(x::AbstractRenderType) = false

	struct Converged <: RegressionTables.AbstractRegressionStatistic val::Union{Bool, Nothing} end
	Converged(m::HnFresult) = Converged(m.converged)
	RegressionTables.label(render::AbstractRenderType, x::Type{Converged}) = "Converged"

	Base.repr(render::AbstractRenderType, x::LogLikelihood; args...) = format(RegressionTables.value(x); commas=true, precision=0) # https://github.com/jmboehm/RegressionTables.jl/issues/160#issuecomment-2139998831
	Base.repr(render::AbstractRenderType, x::BIC; args...) = format(RegressionTables.value(x); commas=true, precision=0) # https://github.com/jmboehm/RegressionTables.jl/issues/160#issuecomment-2139998831
	Base.repr(render::AbstractRenderType, x::Converged; args...) = RegressionTables.value(x) ? "Yes" : "No"
end


# set up and fit model
# any extra keyword arguments are passed to Optim.Options
function HnFfit(z::Vector, wt::Vector=Float64[]; d=1, NLegendre=50, NHermite=50, from::NamedTuple=NamedTuple(), xform::NamedTuple=NamedTuple(),
									methods::Vector=[NewtonTrustRegion()], estname="", modelabsz::Bool=false, penalty::Function=(; kwargs...)->0., kwargs...)

	println("\nModeling $estname data with $d mixture component(s)")
	
	# set starting values & parameter transformes, allowing caller to override defaults
	from  = merge((p=fill(1/d,d), μ=[0.]     , τ=collect(LinRange(1,d,d)), ν=fill(1.,d), pDFR=fill(1/3,3), σ=[1.]      , μₘ=[0.]    , σₘ=[10.]     ),  from)
  xform = merge((p=SimplextoRⁿ, μ=identity , τ=bcast(log)              , ν=bcast(log), pDFR=SimplextoRⁿ, σ=bcast(log), μₘ=identity, σₘ=bcast(log)), xform)

	M = HnFmodel(z, wt; d, modelabsz, NLegendre, NHermite, penalty)
	
	_from = pairs(from)
	fromxform = [xform[p](v) for (p,v) ∈ _from]  # starting values in optimization parameter space

	# indexes to extract individual parameter vectors from full parameter vector
	extractor = zip(keys(_from), Iterators.accumulate((ind,f)->f isa Number ? (last(ind)+1) : last(ind)+1:last(ind)+length(f), fromxform, init=0))

	xformer(x) = (p=>inverse(xform[p])(x[e]) for (p,e) ∈ extractor)  # map primary parameters into full model space, expressed as functions of optimization parameters, e.g. exp(log(σ))
	objective(x) = (#=println(collect(first(p)=>getfield.(last(p),:value) for p in xformer(x)));=# -HnFll(M; xformer(x)...))
	θ = vcat(fromxform...)

	res = nothing
	for method ∈ methods
		res = Optim.optimize(objective, θ, method, Optim.Options(; merge((iterations=250, show_trace=true), kwargs)...), autodiff=:forward)
		θ = Optim.minimizer(res)
	end

	invxform = θ -> [θ[e] |> inverse(xform[p]) for (p,e) ∈ extractor]
	coefdict_maker(v) = NamedTuple(p=>inverse(xform[p])(v[e]) for (p,e) ∈ extractor)
	coefdict = coefdict_maker(θ)

	Δ = ForwardDiff.jacobian(v->vcat(invxform(v)...), θ)  # Jacobian of full model parameters & derived stats wrt optimization parameters
	H = ForwardDiff.hessian(objective, θ)  # Hessian of log likelihood wrt optimization parameters
	V = try pinv(H) catch _ fill(NaN, size(H)) end  # covariance matrix of optimization parameters
	vcov = Δ * V * Δ'  # covariance matrix of full model parameters
	vcov[diagind(vcov)] .= max.(0, vcov[diagind(vcov)])

	# se = NamedTuple([p=> iszero(length(e)) ? zeros(length(inverse(xform[p])(θ[e]))) :
	# 											(e isa Int ? ForwardDiff.derivative : ForwardDiff.jacobian)(inverse(xform[p]), θ[e]) |>
	# 												(Δᵢ -> Δᵢ isa Number ? sqrt0(V[e,e])*abs(Δᵢ) : sqrt0.(diag(Δᵢ * V[e,e] * Δᵢ')))
	# 									for (p,e) ∈ extractor])

	converged = Optim.converged(res)

	t = findall(x->abs(x)>.001, coefdict[:p])  # non-trivial mixture components
	if length(t) < d
		println("Dropping mixture components with negligible weight: keeping $(length(t)) of $d components")
		coefdict = (p=coefdict.p[t], μ=coefdict.μ, τ=coefdict.τ[t], ν=coefdict.ν[t], pDFR=coefdict.pDFR, σ=coefdict.σ, μₘ=coefdict.μₘ, σₘ=coefdict.σₘ)
		I = vcat(t, 1+d, t.+(1+d), t.+(1+2d), 2+3d:size(vcov,1))  # indexes of kept parameters in full parameter vector
		vcov = vcov[I,I]
		M.d[] = d = length(t)
	end

	one2D = first(Unicode.graphemes("₁₂₃₄"),d)
	coefnames = vcat("p".*one2D, "μ", "τ".*one2D, "ν".*one2D, "pD", "pF", "pR", "σ", "μₘ", "σₘ")
	HnFresult(; estname, modelabsz, converged, coefdict, coefnames, coef=vcat(coefdict...), vcov, k=length(θ), n=size(z,1), d, ll=-Optim.minimum(res))
end

function add_derived_stats!(est::HnFresult)
	function derived_stats(; p,μ,τ,ν,pDFR,σ,μₘ,σₘ)
		pD, pF, pR = pDFR

		lnI_H(z₀, zlim=z̄) = logdiffcdf(Normal(z₀,σ[]), abs(zlim), -abs(zlim))

		f_ωz₀(v) = ((ω,z₀)=v; fZ₀condΩ(z₀,ω) * fΩ(ω;p,μ,τ,ν))  # f(z₀)

		f_no_phack(z₀) = begin  # Pr[no p-hacking| |z₀|<z̄]
				S_H = ccdf(𝒩, z̄/σ[]+z₀/σ[]) + ccdf(𝒩, z̄/σ[]-z₀/σ[])
				_μₘ, _σₘ = S_H * μₘ[], S_H * σₘ[]
				ccdf(Normal(0,_σₘ),_μₘ-1)
		end
		f_phacked_insig_z(z₀) = begin  #  Pr[p-hacking tried and |z₁|<z̄ | |z₀|<z̄]
				lnx = lnI_H(z₀)
				S_H = ccdf(𝒩, z̄/σ[]+z₀/σ[]) + ccdf(𝒩, z̄/σ[]-z₀/σ[])
				_μₘ, _σₘ = S_H * μₘ[], S_H * σₘ[]
				μ̃ₘ = _μₘ + _σₘ^2 * lnx
				exp((_μₘ + .5_σₘ^2 * lnx) * lnx + logcdf(Normal(0,_σₘ),μ̃ₘ-1))
		end

		f_Z₁(z₁) = hcubature(v->begin  # distribution of p-hacked z₁
												  (ω,z₀)=v
													-eps() < z₁ < eps() && return 0.
												  lnx = lnI_H(z₀,z₁)
													S_H = ccdf(𝒩, z̄/σ[]+z₀/σ[]) + ccdf(𝒩, z̄/σ[]-z₀/σ[])
													_μₘ, _σₘ = S_H * μₘ[], S_H * σₘ[]
													μ̃ₘ = _μₘ + _σₘ^2 * lnx
													exp(logpdf(Normal(z₀,σ[]), z₁) + (_μₘ-1 + _σₘ^2 * .5lnx) * lnx + logcdf(Normal(0,_σₘ),μ̃ₘ-1)) * (hr(Normal(0,_σₘ), 1-μ̃ₘ)*_σₘ^2 + μ̃ₘ) *
															fZ₀condΩ(z₀,ω) * fΩ(ω;p,μ,τ,ν)
                        end, [-100., -z̄], [100., z̄]; initdiv=10)[1]

		I₀      = hcubature(f_ωz₀, [-100,-z̄], [100, z̄]; initdiv=10, rtol=1e-3,)[1]  # Pr[true insigificant]
		F_no_phack = hcubature(v->((ω,z₀)=v; f_no_phack(z₀) * f_ωz₀(v)), [-100., -z̄], [100., z̄]; initdiv=10, rtol=1e-3)[1]  # Pr[|z₀|,|z₁| ≤ z̄]
		F_phacked_insig_z = hcubature(v->((ω,z₀)=v; f_phacked_insig_z(z₀) * f_ωz₀(v)), [-100., -z̄], [100., z̄]; initdiv=10, rtol=1e-3)[1]  # Pr[|z₀|,|z₁| ≤ z̄]
		F_insig = F_no_phack + F_phacked_insig_z
		S₂₄     = hcubature(f_ωz₀, [-100,-4], [100,-2]; initdiv=10)[1] + hcubature(f_ωz₀, [-100,2], [100,4]; initdiv=10, rtol=1e-3)[1]  # validly "marginally significant" (2<|z|<4)
		Sh₂₄    = quadgk(z₁->f_Z₁(z₁),-4,-2; rtol=1e-3)[1] +   # p-hacked "marginally significant"
		          quadgk(z₁->f_Z₁(z₁), 2, 4; rtol=1e-3)[1]

		# equivocation of Ω w.r.t. reported Z
		infty = 15
		_M = HnFmodel(Float64[]; d=est.d)  # to avoid putting an unoptimized 2-d integral inside a 1-D integral, here use the likelihood evaluator to compute file-drawered fraction
		H_Ω_Z(;p,μ,τ,ν,pDFR,σ,μₘ,σₘ) = (-hcubature(v->((ω,z)=v; t=fΩ(ω; p, μ, τ, ν) *  fZcondΩ( z,ω;pDFR,σ,μₘ,σₘ, truncate=false); xlogy(t,t/fZ([z ]  ; p, μ, τ, ν, pDFR, σ, μₘ, σₘ, truncate=false)[] )), [-infty,-infty], [infty,   -z̄]; initdiv=10, rtol=1e-3)[1]
                                    -hcubature(v->((ω,z)=v; t=fΩ(ω; p, μ, τ, ν) *  fZcondΩ( z,ω;pDFR,σ,μₘ,σₘ, truncate=false); xlogy(t,t/fZ([z ]  ; p, μ, τ, ν, pDFR, σ, μₘ, σₘ, truncate=false)[] )), [-infty,-z̄    ], [infty,    z̄]; initdiv=10, rtol=1e-3)[1]
                                    -hcubature(v->((ω,z)=v; t=fΩ(ω; p, μ, τ, ν) *  fZcondΩ( z,ω;pDFR,σ,μₘ,σₘ, truncate=false); xlogy(t,t/fZ([z ]  ; p, μ, τ, ν, pDFR, σ, μₘ, σₘ, truncate=false)[] )), [-infty, z̄    ], [infty,infty]; initdiv=10, rtol=1e-3)[1]
         -(pDFR[2]<eps() ? pDFR[2] : quadgk(   ω->(         t=fΩ(ω; p, μ, τ, ν) * _fZcondΩ(0.,ω;pDFR,σ,μₘ,σₘ)[2]             ; xlogy(t,t/_HnFll(_M; p, μ, τ, ν, pDFR, σ, μₘ, σₘ                )[2])), -infty, infty)[1]))
		entropy_gain = H_Ω_Z(;p,μ,τ,ν,pDFR,σ,μₘ,σₘ) - H_Ω_Z(;p,μ,τ,ν,pDFR=[1.,0,0], σ,μₘ=[-100.],σₘ=[1.])

		H_z₀(τ_multiplier) = (-hcubature(v -> ((ω,_)=v; xexpx(lnfΩ(ω; p, μ, τ=τ*τ_multiplier, ν) + lnfZ₀condΩ(v...))), [-infty,-infty], [infty,   -z̄]; initdiv=10, rtol=1e-3)[1]
						              -hcubature(v -> ((ω,_)=v; xexpx(lnfΩ(ω; p, μ, τ=τ*τ_multiplier, ν) + lnfZ₀condΩ(v...))), [-infty,-z̄    ], [infty,    z̄]; initdiv=10, rtol=1e-3, atol=1e-3)[1]
						              -hcubature(v -> ((ω,_)=v; xexpx(lnfΩ(ω; p, μ, τ=τ*τ_multiplier, ν) + lnfZ₀condΩ(v...))), [-infty, z̄    ], [infty,infty]; initdiv=10, rtol=1e-3, atol=1e-3)[1])

		equiv_sample_reduction = 1 - find_zero(τ_multiplier -> H_z₀(τ_multiplier) - (H_z₀(1) - entropy_gain), (0.01, 1.5); rtol=1e-3)

		[
			pF*F_insig                              # fraction of all studies file-drawered
			pF*F_insig / I₀                         # fraction of initially insignificant studies file-drawered
			pR * F_insig / I₀ + pD * F_no_phack     # fraction of initially insignificant published as is
			1 - F_insig / I₀                        # fraction of initially insignificant that lead to published, significant, p-hacked results
			pD * (F_insig / I₀ - F_no_phack)        # fraction of initially insignificant that lead to published, insignificant, p-hacked results

			pD / (1-pF) * (1 - F_no_phack / F_insig * I₀)     # fraction of published insignificant results that are p-hacked
			(I₀ - F_insig) / (1 - F_insig)          # fraction of significant results that are p-hacked
			Sh₂₄ / (Sh₂₄ + S₂₄)                     # p-hacked fraction of "marginally significant" in Star Wars (2<|z|<4)

			entropy_gain/log(2)                     # H(Ω|Z) - H(Ω|Z₀), in bits
			equiv_sample_reduction
		]
	end
	# [ForwardDiff.derivative(σ->derived_stats(;p=est.coefdict.p,μ=est.coefdict.μ,τ=est.coefdict.τ,ν=est.coefdict.ν,pDFR=est.coefdict.pDFR,σ=[σ],m=est.coefdict.m), .5)[9] for est ∈ (Setal, GMpolisci, GMsoc, SW, BCH, ABetal, vZSS, V)]

	d = length(est.coefdict.p)
	est.coef = est.coef[1:3d+7]  # remove old derived stats if any; for debugging, allows this function to be called repeatedly
	est.vcov = est.vcov[1:3d+7, 1:3d+7]
	est.coefnames = est.coefnames[1:3d+7]

	extractor = zip(keys(est.coefdict), Iterators.accumulate((ind,f)->f isa Number ? (last(ind)+1) : last(ind)+1:last(ind)+length(f), est.coefdict, init=0))
	coefdict_maker(v) = NamedTuple(p=>v[e] for (p,e) ∈ extractor)
	Δ = ForwardDiff.jacobian(v->vcat(v..., derived_stats(;coefdict_maker(v)...)), est.coef)  # Jacobian of full model parameters & derived stats wrt full model parameters
	est.vcov = Δ * est.vcov * Δ'  # covariance matrix of full model parameters
	est.vcov[diagind(est.vcov)] .= max.(0, est.vcov[diagind(est.vcov)])
	est.coef = vcat(est.coefdict..., derived_stats(;est.coefdict...)...)
	est.coefnames = vcat(est.coefnames, "overall_file_drawer_frac", "frac_insig_file_drawered", 
										 "frac_insig_pubbed_as_is", "sig_p_hacked_frac", "insig_p_hacked_frac",
										 "p_hacked_frac_of_pubbed_insig", "p_hacked_frac_of_sig", "p_hacked_frac_of_marg_sig", "H(Ω|Z)-H(Ω|Z₀)", "equiv_sample_reduction")
	show(regtable(est))
	est
end

function HnFestimate(df::DataFrame, z::Symbol, wt=nothing; dmax=2, estname="", NLegendre=250, NHermite=50, kwargs...)
	# for speed, collapse duplicates in data set
	gdf = isnothing(wt) ? @combine(@groupby(df, z), :z=first($z), :wt=length($z)) :
												@combine(@groupby(df, z), :z=first($z), :wt=sum(  $wt))

	results = [HnFfit(gdf.z, gdf.wt; d, estname="$estname$d", NLegendre, NHermite, kwargs...) for d ∈ 1:dmax]
	est = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	est.n = size(df,1)  # insert correct sample size, from before collapsing the data
	add_derived_stats!(est)
end

function HnFplot(z, est, wt::Vector=Float64[]; NLegendre=50, NHermite=50, zplot=-5+1e-3:.01:5, ωplot=zplot, title::String="", noAKplots::Bool=true)
	t = est.coefdict
	kwargsω = (p=t.p, μ=t.μ, τ=t.τ, ν=t.ν)
	kwargsz = (pDFR=t.pDFR, σ=t.σ, μₘ=t.μₘ, σₘ=t.σₘ)
	kwargsz0 = (pDFR=[1.,0.,0.], σ=[1.], μₘ=[-10.], σₘ=[.0001])  # no distortion

	f = Figure(size=(1500,900))

	# empirical distribution of z's + model fit
	Axis(f[1,1], xlabel="z", ylabel="Density", limits=(est.modelabsz ? 0 : -10, 10, nothing,nothing))
	hist!(z, normalization=:pdf, bins=floor(Int,2√size(z,1)), weights=length(wt)==0 ? Makie.automatic : wt, 
	        label="Actual published effects", color=(:slategray,.4))  # outline histogram of data

	s,e = extrema(z); _zplot = s:.01:e

  pplottrue                     = map(z->dot(t.p, pdf.(GenT.(kwargsω.μ, t.τ, t.ν),  z)), _zplot)
  est.modelabsz && (pplottrue .+= map(z->dot(t.p, pdf.(GenT.(kwargsω.μ, t.τ, t.ν), -z)), _zplot))
	pplottrue ./= 1 - est.coef[findfirst(==("overall_file_drawer_frac"), est.coefnames)]

	pplotinitial = fZ(_zplot; kwargsω..., kwargsz0..., modelabsz=est.modelabsz, NLegendre, NHermite)
	pplotfit     = fZ(_zplot; kwargsω..., kwargsz ..., modelabsz=est.modelabsz, NLegendre, NHermite)

	lines!(_zplot, pplottrue, label="Model: true effects", color=Makie.wong_colors()[3])
	lines!(_zplot, pplotinitial, label="Model: initial estimates", color=Makie.wong_colors()[1])
	lines!(_zplot, pplotfit, label="Model: published estimates", color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)

	# distribution of z | ω=2
	ω = 2.
	Axis(f[1,2], xlabel="Reported z | true z = $ω", ylabel="Density")
	lines!(zplot, fZcondΩ.(zplot, ω; kwargsz0..., NLegendre), label="updating from prior")
	lines!(zplot, fZcondΩ.(zplot, ω; kwargsz..., NLegendre), label="updating from prior + p-hacking", color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)
	
	# distribution of ω | z=2
	_z = 2.
	Axis(f[2,1], xlabel="True z | reported z = $_z", ylabel="Density")
	lines!(ωplot, fΩcondZ.(ωplot, _z; kwargsω..., kwargsz0..., NLegendre, NHermite), label="updating from prior")
	lines!(ωplot, fΩcondZ.(ωplot, _z; kwargsω..., kwargsz..., NLegendre, NHermite), label="updating from prior + research p-hacking", color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)
	
	# frequentist equal-tailed CI's as fn of z--Andrews & Kasy (2014), Figure 2
	CIs0 = Cquant.([.025 .5 .975], zplot; rtol=.0001, kwargsz0..., NLegendre)
	CIs  = Cquant.([.025 .5 .975], zplot; rtol=.0001, kwargsz ..., NLegendre)
	Axis(f[1,3], xlabel="Reported z", ylabel="Point estimate and 95% CI for true z", xticks=-5:5, yticks=-6:6)
	lines!(zplot, CIs0[:,1], color=Makie.wong_colors()[1], label="No adjustment")
	lines!(zplot, CIs0[:,2], color=Makie.wong_colors()[1], linestyle=:dash)
	lines!(zplot, CIs0[:,3], color=Makie.wong_colors()[1])
	lines!(zplot, CIs[:,1], color=Makie.wong_colors()[6], label="Adjusting for p-hacking")
	lines!(zplot, CIs[:,2], color=Makie.wong_colors()[6], linestyle=:dash)
	lines!(zplot, CIs[:,3], color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)

	t = findall(x->abs(x)<.1, CIs[:,1])
	if length(t)> 0
		s,e = extrema(findall(x->abs(x)<.1, CIs[:,1]))
		lb = linear_interpolation(denoise(CIs[:,1], factor=.1)[1][s:e], zplot[s:e])(0.)  # McCrary, Christensen, and Fanelli (2016)-style z thresholds for p<.05
		s,e = extrema(findall(x->abs(x)<.1, CIs[:,3]))
		ub = linear_interpolation(denoise(CIs[:,3], factor=.1)[1][s:e], zplot[s:e])(0.)
		scatter!([lb;ub],[0.;0], color=Makie.wong_colors()[6])
		text!(lb, 0., text=format("{:03.2f}", lb), align=(:right, :bottom), fontsize=18)
		text!(ub, 0., text=format("{:03.2f}", ub), align=(:left, :top), fontsize=18)
	end

	# Posterior mean of ω as fn of Z
	pplot0 = EΩcondZ(zplot; kwargsω..., kwargsz0..., NLegendre, NHermite)
	pplot  = EΩcondZ(zplot; kwargsω..., kwargsz..., NLegendre, NHermite)
	Axis(f[2,2], xlabel="Reported z", ylabel="Expected true z")
	lines!(zplot, zplot, label="As is", color=Makie.wong_colors()[3])
	lines!(zplot, pplot0, label="Updating from prior", color=Makie.wong_colors()[1])
	lines!(zplot, pplot , label="updating from prior + p-hacking", color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)

	# E[ω] discount
	Axis(f[2,3], xlabel="Reported z", ylabel="Discount multiplier" #=, yticks=0:.1:1.5 , limits=(nothing,nothing,0.,nothing)=#)
	lines!(zplot[zplot.>.2], Float16.(pplot0[zplot.>.2]./zplot[zplot.>.2]), label="updating from prior")  # https://discourse.julialang.org/t/range-step-cannot-be-zero/66948/11?u=droodman
	lines!(zplot[zplot.>.2], Float16.(pplot[zplot.>.2]./zplot[zplot.>.2]), label="updating from prior + p-hacking", color=Makie.wong_colors()[6])
  y = EΩcondZ([2]; kwargsω..., kwargsz0..., NLegendre, NHermite)[] / 2
  scatter!(2, y, color=Makie.wong_colors()[1])
	text!(2, y, text=format("{:03.2f}", y), align=(:center, :bottom), fontsize=18)
  y = EΩcondZ([2]; kwargsω..., kwargsz... , NLegendre, NHermite)[] / 2
  scatter!(2, y, color=Makie.wong_colors()[6])
	text!(2, y, text=format("{:03.2f}", y), align=(:center, :top), fontsize=18)
  axislegend(position=:lt, framevisible = false)

	title=="" || (f[0, 1:3] = Label(f, title))
	f |> display
	save("output/$(est.estname) all.png", f)

	# Plots modeled on Andrews & Kasy (2019)
	if !noAKplots
		fAK = Figure(size=(1000,500))
		fAK[0, 1:2] = Label(fAK, title)
		Axis(fAK[1,1], xlabel="True z", ylabel="Median bias in reported z")
		lines!(ωplot, zeros(size(ωplot)))
		lines!(ωplot, quantFcondΩ.(.5, ωplot; kwargsz..., NLegendre) .- ωplot)

		Axis(fAK[1,2], xlabel="True z", ylabel="Coverage of reported 95% CI")
		lines!(ωplot, fill(.95, size(ωplot)...))
		lines!(ωplot, @. FZcondΩ(ωplot+z̄, ωplot; kwargsz..., NLegendre)-FZcondΩ(ωplot-z̄, ωplot; kwargsz..., NLegendre))
		fAK |> display
		save("output/$(est.estname) A&K Fig1.png", fAK)
	end
end


#
# check model with simulation
#

p = [.7,.3]
μ = [0.7]
τ = [1.2,2.7]
ν = [2., 20.]
pD = .4
pF = .4
pR = .2
σ = [.7]
μₘ = [15.]
σₘ = [10.]
d = length(p)
modelabsz = false
pDFR = [pD, pF, pR]
kwargs = (p=p, μ=μ, τ=τ, ν=ν, pDFR=pDFR, σ=σ, μₘ=μₘ, σₘ=σₘ, modelabsz=modelabsz)

n = 100_000
Random.seed!(1232)
sim = HnFDGP(n; kwargs..., truncate=true)

f = Figure()
Axis(f[1,1], limits=(modelabsz ? 0 : -10, 10, nothing,nothing))
hist!(sim.z[abs.(sim.z).<100], bins=10*2*100, normalization=:pdf)
zplot = (modelabsz ? 0 : -10):.01:10
lines!(zplot, fZ(zplot; NHermite=50, NLegendre=50, kwargs...), color=:orange, label="True model")
f|>display

res = HnFfit(sim.z; d, modelabsz, NLegendre=50, estname="simulated", extended_trace=false)  # penalized maximum likelihood
print(res.coefdict)
lines!(zplot, fZ(zplot; modelabsz, res.coefdict...), color=:blue, label="Fitted model")

f[0,:] = Label(f, "Simulation vs model")
axislegend(position=:lt, framevisible=false)
colsize!(f.layout, 1, Relative(1))
f |> display


#
# model real data
#

@time begin
	# penalty function for parameters that can generate singularities
  penalty(; τ::Vector{T}, σ::Vector{T}, σₘ::Vector{T}, file_drawer_insig::T, kwargs...) where {T} = 
		logpdf(Normal(0,50), log(σ[])) + 
		logpdf(Normal(0,50), log(σₘ[])) + 
		sum(logpdf(Normal(0,5), log(τᵢ)) for τᵢ ∈ τ) +
		logpdf(Beta(2,1),file_drawer_insig)

	# van Zwet, Schwab, and Senn (2021) data, https://osf.io/xq4b2
	df = DataFrame(CSV.File("data/van Zwet, Schwab, and Senn 2021/CochraneEffects.csv"))
	@. @subset!(df, abs(:z)<20 && :"outcome.nr"==1 && :RCT=="yes" && :"outcome.group"=="efficacy")  # vZSS uses 20
	Random.seed!(29384)
	df = combine(groupby(df, :"study.id.sha1"), :z => sample => :z)  # randomly choose among primary outcomes
  vZSS = HnFestimate(df, :z; penalty, estname="vZSS")
	HnFplot(df.z, vZSS; title="van Zwet, Schwab, and Senn (2021) data")

	# Schuemie et al. (2013), https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fsim.5925&file=Appendix+G+Revision.xlsx
	df = DataFrame(XLSX.readtable("data/Schuemie et al. 2013/appendix g revision.xlsx", "NeatTable", first_row=2, infer_eltypes=true)...)
	@. df.z = log(df."Effect estimate") / (log(df."Upper bound of 95% CI" / df."Lower bound of 95% CI") / 2z̄)
	@. @subset!(df, abs(:z)<20)
	disallowmissing!(df, :z)
  Setal = HnFestimate(df, :z; penalty, estname="Setal")
	HnFplot(df.z, Setal; title="Schuemie et al. (2013) data")

	# Star Wars, DOI 10.1257/app.20150044, openicpsr.org/openicpsr/project/113633/version/V1/view?path=/openicpsr/113633/fcr:versions/V1/brodeur_le_sangnier_zylberberg_replication/Data/Final/final_stars_supp.dta&type=file
	df = DataFrame(CSV.File("data/Brodeur et al. 2016/final_stars_supp.csv"))
	df.z = df.coefficient_num ./ df.standard_deviation_num
	@. @subset!(df, lowercase(:main)=="yes" && !ismissing(:z) && abs(:z)<20)
	disallowmissing!(df, :z)
  SW = HnFestimate(df, :z, :weight_table; penalty, estname="SW")
	HnFplot(df.z, SW, df.weight_table; title="Brodeur et al. (2016) data")

	# Brodeur, Cook, and Heyes 2020, DOI 10.1257/aer.20190687, openicpsr.org/openicpsr/project/120246/version/V1/view?path=/openicpsr/120246/fcr:versions/V1/MM-Data.dta&type=file
	df = DataFrame(CSV.File("data/Brodeur, Cook, and Heyes 2020/MM Data.csv"))
	df.z = df.mu ./ df.sd
	@. @subset!(df, !ismissing(:z) && !isnan(:z) && abs(:z)<20)
	disallowmissing!(df, :z)
	hist(df.z, bins=100) |> display
	df.z .= abs.(df.z)
  BCH = HnFestimate(df, :z; penalty, modelabsz=true, estname="BCH")
	HnFplot(df.z, BCH; title="Brodeur, Cook, and Heyes (2020) data")

	# Arel-Bundock et al. 2026
	df = DataFrame(CSV.File("data/Arel-Bundock et al. 2026/arel-bundock_briggs.csv"))
	@. @subset!(df, !ismissing.(:z_stat) && abs(:z_stat)<20)
  ABetal = HnFestimate(df, :z_stat; penalty, estname="ABetal")
	HnFplot(df.z_stat, ABetal; title="Arel-Bundock et al. (2026) data")

	# Vivalt 2020, DOI 10.1093/jeea/jvaa019, https://figshare.com/articles/dataset/Replication_files_for_How_Much_Can_We_Generalize_from_Impact_Evaluations_/12048600/1
	df = DataFrame(CSV.File("data/Vivalt 2020/data_unstandardized.csv"))
	df.z = df.treatmentcoefficient ./ df.treatmentstandarderror
	@. @subset!(df, abs(:z)<20)
  V = HnFestimate(df, :z; penalty, estname="V")
	HnFplot(df.z, V; title="Vivalt (2020) data")

	# Gerber and Malhotra 2008 poli sci, DOI 10.1177/1532673X09350979 https://www.nowpublishers.com/article/details/supplementary-info/100.00008024_supp.rar
	df = [DataFrame(load("data/Gerber and Malhotra 2008a/AJPS_Data.xls", "All Studies"))[2:end,[:x4,:x6]]
				DataFrame(load("data/Gerber and Malhotra 2008a/APSR_Data.xls", "All Studies"))[2:end,[:x4,:x6]] ]
	@. @subset!(df, !ismissing(:x4))
	df.z = Float64.(df.x6)
	@. @subset!(df, abs(:z)<20)
  GMpolisci = HnFestimate(df, :z; penalty, estname="GMpolisci")
	HnFplot(df.z, GMpolisci; title="Gerber & Malhotra (2008a) data")

	# Gerber and Malhotra 2008 sociology, DOI 10.1177/0049124108318973
	df = [DataFrame(load("data/Gerber and Malhotra 2008b/ASR (9.26.06).xls", "ASR", ncols=7))
				DataFrame(load("data/Gerber and Malhotra 2008b/ASR (9.26.06).xls", "AJS", ncols=7))
				DataFrame(load("data/Gerber and Malhotra 2008b/ASR (9.26.06).xls", "TSQ", ncols=7))]
	@. @subset!(df, !ismissing(:Z) && abs(:Z)<20)
  GMsoc = HnFestimate(df, :Z; penalty, estname="GMsoc")
	HnFplot(df.Z, GMsoc; title="Gerber & Malhotra (2008b) data")

	# Barnett and Wren 2019 ~1M sample, DOI: 10.1136/bmjopen-2019-032506, https://github.com/agbarnett/intervals/blob/master/data/Georgescu.Wren.RData
	df = DataFrame(RData.load("data/Georgescu and Wren 2018/Georgescu.Wren.RData")["complete"])
	@. df.ci_level[ismissing(df.ci_level) || df.ci_level==.0095 || df.ci_level==.05] = .95
	@. df.z = log(df.mean) / (ifelse(ismissing(df.lower) || iszero(df.lower), log(df.upper / df.mean), log(df.upper / df.lower) / 2) / cquantile(𝒩, (1 - df.ci_level)/2))
	@. @subset!(df, !ismissing(:z) && !ismissing(:lower) && iszero(:mistake) && !isnan(:z) && !isinf(:z) && abs(:z)<20)
	BW = HnFestimate(df, :z; penalty, estname="BW")
	HnFplot(df.z, BW; title="Barnett and Wren (2019) data")

	@. @subset!(df, :source=="Abstract")
  BWAbstr = HnFestimate(df, :z; penalty, estname="BWAbstr")
	HnFplot(df.z, BWAbstr; title="Barnett and Wren (2019) data, abstracts only")

	table = regtable(Setal, GMpolisci, GMsoc, SW, BCH, ABetal, vZSS, V, BW, BWAbstr;
							estim_decoration = (coef,p)->coef,  # no stars
							regression_statistics = [Nobs #=, Converged, LogLikelihood, BIC=#],
							print_estimator_section = false,
							keep = ["p₁", "p₂", "p₃", "p₄", "μ", "τ₁", "τ₂", "τ₃", "τ₄", "ν₁", "ν₂", "ν₃", "ν₄", "pF", "pD", "pR", "σ", "μₘ", "σₘ", "frac_insig_file_drawered", "frac_insig_pubbed_as_is", "p_hacked_frac_of_pubbed_insig", "p_hacked_frac_of_sig", "p_hacked_frac_of_marg_sig","H(Ω|Z)-H(Ω|Z₀)", "equiv_sample_reduction"],
							estimformat = "%0.3g",
							statisticformat = "%0.3g",
							number_regressions = false,
							file = "output/results.txt")
end