cd(dirname(@__FILE__))
cd("..")

using Pkg
Pkg.activate(".")  # activate this project's environment
Pkg.instantiate()  # make sure all packages installed

using Random, IrrationalConstants, Format, Distributions, Interpolations, Base.Iterators, FastGaussQuadrature, Optim, LogExpFunctions, CSV, DataFrames, DataFramesMeta, ForwardDiff, LinearAlgebra, Roots, QuadGK, Statistics, 
       InverseFunctions, StatsAPI, StatsBase, StatsModels, RegressionTables, Unicode, CairoMakie, Makie, ExcelFiles, XLSX, RData, SpecialFunctions, ThreadsX, HCubature

const 𝒩 = Normal()
const z̄ = quantile(𝒩, .975)  # 1.96

@inline diffcdf(N,b,a) = cdf(N,b) - cdf(N,a)
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

# # transform to constrain pDFHR to have pR=0
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


# compute f(z|ω) & F(file drawer|ω). Return in provided 2-vector y
function _fZcondΩ!(y, z, ω; modelabsz=false, NLegendre=50, pDFHR::Vector{T}, σ::Vector{T}, m::Vector{T}) where {T}
	pD, _, pH, pR = pDFHR
  lnpH = log(pH)

	Z₀, W = gausslegendre(NLegendre)  # nodes and weights for Gauss-Legendre quadrature over [-1,1]
	Z₀ .*= z̄  # change of variables to quadrature over [-z̄, z̄]
  lnWLegendre = log.(W) .+ log(z̄)

	zdivσ, z̄divσ = z/σ[], z̄/σ[]

  file_drawer = ∫ = 0.
	b = zdivσ; absb = abs(b)
	@inbounds for k ∈ 1:NLegendre  # p-hacking; integrate out z₀ over [-z̄, z̄]
		a = Z₀[k] / σ[]
    B = lnWLegendre[k] + logpdf(𝒩, Z₀[k]-ω) - log1mexp(lnpH + logdiffcdf(𝒩, a+z̄divσ, a-z̄divσ) * m[])
    file_drawer += exp(B) 

		if a+absb ≉ a-absb
      F = logpdf(𝒩, b-a) + logdiffcdf(𝒩, a+absb, a-absb) * (m[]-1)
			modelabsz && (F += log1pexp(-2b * a))  # log [ϕ(a-b) + ϕ(a+b)] = log[ϕ(a-b)] + log[1+exp(-2ab)]
			∫ += exp(B + F)
		end
	end
	∫ *= m[] / σ[] * pH  # density contribution from p-hacking

	f_z = pdf(𝒩, z-ω)
	modelabsz && (f_z += pdf(𝒩, z+ω))

	∫ += f_z  # contribution from publishing original stat without p-hacking
	if -z̄ ≤ z ≤ z̄
		∫ *= pD  # in insignificant range, same formulas, but times pD
    ∫ += pR * f_z / (1 - exp(lnpH + logdiffcdf(𝒩, zdivσ+z̄divσ, zdivσ-z̄divσ) * m[]))  # contribution from reverting to original stat after p-hacking
	end
	y .= ∫, file_drawer
end

_fZcondΩ(z, ω; kwargs...) = _fZcondΩ!(Vector{Float64}(undef,2), z, ω; kwargs...)

 # f(z|ω). If truncate=true (the default), returns the density conditional on publication
fZcondΩ(z, ω; modelabsz=false, NLegendre=50, pDFHR, σ, m, truncate=true) = _fZcondΩ(z, ω; modelabsz, NLegendre, pDFHR, σ, m) |> (y -> truncate ? y[1]/(1 - pDFHR[2]*y[2]) : y[1])
 
# the most time-consuming plotting is of the confidence intervals: for various values of ω, 
# the cdf F(z|ω) is numerically calculated, many times--iteratively seeking where it hits, e.g., .025 and .975
# to save time, pre-compute all components of f(z|ω) that do not depend on z, notably logdiffcdf(𝒩(0,σ), Z₀[k]+z̄, Z₀[k]-z̄)
function FZcondΩ(z, ω; modelabsz::Bool=false, NLegendre=50, pDFHR, σ, m, rtol=.00001, order=13)
	pD, pF, pH, pR = pDFHR
  lnpH = log(pH)

	Z₀, W = gausslegendre(NLegendre)  # nodes and weights for Gauss-Legendre quadrature over [-1,1]
	Z₀ .*= z̄  # change of variables to quadrature over [-z̄, z̄]
	W  .*= z̄
	
	z̄divσ, Z₀divσ = z̄/σ[], Z₀/σ[]

	A = 0.
	B = Vector{Float64}(undef, NLegendre)
	@inbounds for k ∈ 1:NLegendre
		a = Z₀[k] / σ[]
		B[k] = log(W[k]) + logpdf(𝒩, Z₀[k] - ω) - log1mexp(lnpH + logdiffcdf(𝒩, a+z̄divσ, a-z̄divσ) * m[])
		A += exp(B[k])
	end

	function myfZcondΩ(z)
		zdivσ = z / σ[]
		b = abs(zdivσ)

		∫ = 0.
		@inbounds for k ∈ 1:NLegendre
			a = Z₀divσ[k]
			if a+b ≉ a-b
				Fₖ = -.5(zdivσ - a)^2 + (m[]-1) * logdiffcdf(𝒩, a+b, a-b)  # p_H ϕ(z;z_0,σ^2 )
				modelabsz && (Fₖ += log1pexp(-2a * b))  # log [ϕ(a-b) + ϕ(a+b)] = log[ϕ(a-b)] + log[1+exp(-2ab)]
				∫ += exp(B[k] + Fₖ)
			end
		end
		∫ *= pH / σ[] * m[]

		f_z = exp(-.5(z-ω)^2)
		modelabsz && (f_z += exp(-.5(z+ω)^2))

		∫ += f_z
		if -z̄ ≤ z ≤ z̄
			∫ *= pD
      ∫ += pR * f_z / (1 - pH * diffcdf(𝒩, zdivσ+z̄divσ, zdivσ-z̄divσ) ^ m[])
		end
		∫
	end

	endpoints = modelabsz ? [0, z̄] : [-Inf, -z̄, z̄]  # since f(z|ω) jumps at ±z̄, do quadrature separately in each range
	endpoints = [endpoints[findall(<(z), endpoints)]; z]
	quadgk(myfZcondΩ, endpoints...; rtol, order)[1] * invsqrt2π / (1 - pF * A)
end

quantFcondΩ(q, ω; kwargs...) = find_zero(z -> q - FZcondΩ(z, ω; kwargs...), (-20,20), Roots.ITP())  # ITP algorithm works well

# likelihood for a collection (vector, step range) of z's for plotting
# If truncate=true (default), returns the truncated density, i.e., conditional on publication
function fZ(z; modelabsz=false, NHermite=50, NLegendre=50, p, μ, τ, ν, pDFHR, σ, m, truncate=true)
  M = HnFmodel(z; d=length(τ), NHermite, NLegendre, modelabsz)
  ∫, G = _HnFll(M; p,μ,τ,ν,pDFHR,σ,m)
	∫ .= exp.(∫)
  truncate && (∫ ./= 1 - pDFHR[2]*G)
  ∫
end


# f(z), f(ω), f(ω|z), E[ω|z]
# inconsistency: z should be a scalar for fΩcondZ but a vector or other iterable for EΩcondZ
fΩ(ω; p, μ, τ, ν) = p'pdf.(GenT.(μ,τ,ν), ω)
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
	N::Int  # number of z's in data, # of insignificant
	k::Int  # number of z knots for interpolation
	interpolate::Bool	# interpolation resolution (points per unit interval); 0 means no interpolation
	kts::Vector{Float64}  # interpolation knots/observations in z space
	insig::BitVector  # which knots are in insignificant region
	splinetype::Interpolations.InterpolationType  # type of interpolation
	zint::Vector{Float64}  # z values mapped to cardinal knot numbering space since interpolate() is faster with cardinally spaced knots
	NHermite::Int  # number of quadrature points for integration over z₀ to compute f(z₀)
	Ω::Vector{Float64}; WHermite::Vector{Float64}; lnWpΩ²::Vector{Float64}  # quadrature nodes & weights
	NLegendre::Int  # number of quadrature points
	Z₀::Vector{Float64}; WLegendre::Vector{Float64}; lnWLegendre::Vector{Float64}  # quadrature nodes & weights
  penalty::Function
	Bdict::Dict{DataType, Vector}  # collections of pre-allocated arrays for use in likelihood computation, separate for Float64, ForwardDiff.Dual, etc.
	Edict::Dict{DataType, Vector}
	Fdict::Dict{DataType, Matrix}
	tot_hacking_dict::Dict{DataType, Vector}
	∫dict::Dict{DataType, Matrix}

	function HnFmodel(z, wt=Float64[]; d::Int, modelabsz=false, interpres=0, NHermite=50, NLegendre=50, splinetype::Interpolations.InterpolationType=BSpline(Linear()), 
                    penalty::Function=(; kwargs...)->0.)
		if iszero(interpres)
			kts = z
			zint = Float64[]
		else
			e = max(10, maximum(abs.(extrema(z)))) + .2  # interpolation knots span a bit beyond [-10,10] to avoid edge effects; symmetric start at 0 if modelabsz=true
			kts = (modelabsz ? -.2 : -e) : 1/interpres : e  # LinRange(modelabsz ? 0 : -e, e, (2-modelabsz) * ceil(Int, e * interpres) + 1)
			zint = (z .- first(kts)) .* interpres .+ 1
		end

		Ω, WHermite = gausshermite(NHermite)
		Ω .= √2 .* Ω; WHermite ./= √π  # fold in adjustment for change of variables from pdf(Normal(ω)) to exp(-x²)

		Z₀, W = gausslegendre(NLegendre)  # nodes and weights for Gauss-Legendre quadrature over [-1,1]
		Z₀ .*= z̄; W .*= z̄  # change of variables to quadrature over [-z̄, z̄]
		
		new(modelabsz, [d], z, wt/mean(wt), length(z), length(kts), interpres!=0, kts, -z̄ .≤ kts .≤ z̄, splinetype, zint, NHermite, Ω, WHermite, log.(WHermite).+.5Ω.^2, NLegendre, Z₀, W, log.(W), penalty, Dict(), Dict(), Dict(), Dict(), Dict())
	end
end

# to prevent "MethodError: ==(::ForwardDiff.Dual{ForwardDiff.Tag{var"#objective#178"{…}, Float64}, Float64, 11}, ::IrrationalConstants.Invsqrt2) is ambiguous."
import Base.==
==(a::ForwardDiff.Dual, b::IrrationalConstants.Invsqrt2) = a == Float64(b)


#
# Hack'n'file log likelihood
#

# Compute observation-level likelihood (not log likelihood) and expected number of publish/file-drawer/p-hack decision junctures (G)
function _HnFll(M::HnFmodel; p::AbstractVector{T}, μ::AbstractVector{T}, τ::AbstractVector{T}, ν::AbstractVector{T}, pDFHR::AbstractVector{T}, σ::Vector{T}, m::Vector{T}) where {T}
  pD, _, pH, pR = pDFHR
	lnpD, lnpH = log(pD), log(pH)
	lnpHσm = lnpH + log(m[] / σ[])
	mm1 = m[] - 1

	z̄divσ, zdivσ, Z₀divσ = z̄/σ[], M.kts/σ[], M.Z₀/σ[]

	is = findall(>(1e-6), p)  # nonzero mixture components
	_d = length(is)

	# pre-allocating these hampers automatic differentiation because they depend on T, which could be a Dual number
	∫ = _d<M.d[] ? Matrix{T}(undef,M.k,_d) : get!(M.∫dict, T, Matrix{T}(undef,M.k,_d))  # likelihood contributions for each z knot and each mixture component
	G = zero(T)	 # accumulator for expected number of publish/file-drawer/p-hack decision junctures
	B = get!(M.Bdict, T, Vector{T}(undef, M.NLegendre))
	F = get!(M.Fdict, T, Matrix{T}(undef, M.NLegendre, M.k))  # ϕ(z;z_0,σ^2 ) 〖ΔΦ(|z|,-|z|;z_0,σ^2 )〗^(m-1) for each z and each z₀ (Legendre integration point)
  tot_hacking = get!(M.tot_hacking_dict, T, Vector{T}(undef,M.k))

  if pH < eps()
    E = M.lnWLegendre
  else
    E = get!(M.Edict, T, Vector{T}(undef,M.NLegendre))  # w/(1-p_H  ΔΦ(z ̅,-z ̅;z_0,σ^2 ) ) for each z₀ (Legendre integration point)
    for k ∈ eachindex(E)  # for each Legendre point; pre-compute part of p-hacking contribution
      @inbounds E[k] = M.lnWLegendre[k] - log1mexp(lnpH + m[] * logdiffcdf(𝒩, Z₀divσ[k]+z̄divσ, Z₀divσ[k]-z̄divσ))  # w/(1-p_H  ΔΦ(z ̅,-z ̅;z_0,σ^2 ) )
    end
  end

	Threads.@threads for j ∈ 1:M.k
		@inbounds begin
			b = zdivσ[j]; absb = abs(b)
			M.modelabsz && (neg2b = -2b)

			M.insig[j] && (tot_hacking[j] = log(pD + (pH < eps() ? pR : pR / exp(log1mexp(lnpH + logdiffcdf(𝒩, b+z̄divσ, b-z̄divσ) * m[])))))

			l = LinearIndices(F)[1,j]  # index of top entry in this col, arrays being stored col-first
			for k ∈ eachindex(Z₀divσ)  # for each z₀ (Legendre integration point)
				a = Z₀divσ[k]
				if a+absb ≉ a-absb
					Fₖⱼ = mm1 * logdiffcdf(𝒩, a+absb, a-absb) - .5(log2π + (a-b)^2)
					M.modelabsz && (Fₖⱼ += log1pexp(neg2b * a))  # log [ϕ(a-b) + ϕ(a+b)] = log[ϕ(a-b)] + log[1+exp(-2ab)]
					F[l] = Fₖⱼ
				else
					F[l] = -floatmax()  # z->0 limit if m ≥ 1
				end
				l += 1
			end
		end
	end

	@inbounds for _i ∈ 1:_d
		i = is[_i]

		# f(z_0) for ith mixture component, integrating out ω with Gauss-Hermite quadrature
		# because this is an inner loop, economize by manually computing the log t pdf while avoiding redundant work
		τᵢ² = τ[i]^2; _τᵢ² = 1+1/τᵢ²; sqrt_τᵢ² = √_τᵢ²
		halfinv_τᵢ² = .5 / _τᵢ²
		_νᵢ = ν[i]/2 + .5
		D = (1 + τ[i]^2) * ν[i]
		Cᵢ = log(p[i]) - logbeta(ν[i]/2,.5) - .5log(D)  # contains constant factor in t pdf, in logs
		lnf_z₀_i(z₀) = logsumexp(begin  # log [∫_(-∞)^∞ ϕ(z₀;ω)t(ω;μ,τᵢ²,νᵢ)dω] sans ln Cᵢ factor
												d = (z₀ - μ[]) / sqrt_τᵢ²
												lnwpx² - halfinv_τᵢ² * (x - d / τᵢ²)^2 - log1p((x + d)^2 / D) * _νᵢ
											end
											for (x,lnwpx²) ∈ zip(M.Ω, M.lnWpΩ²))

    Gᵢ = zero(T)
    for k ∈ eachindex(E)	# for each z₀ (Legendre integration point)
      t = E[k] + lnf_z₀_i(M.Z₀[k])
      Gᵢ += exp(t)
		  B[k] = lnpHσm + t
    end
    G += exp(Cᵢ) * Gᵢ

		Threads.@threads for j ∈ 1:M.k  # for each z value/interpolation point
			@inbounds begin
				lnf_z₀ᵢⱼ = M.modelabsz ? logsumexp(lnf_z₀_i(M.kts[j]), lnf_z₀_i(-M.kts[j])) : lnf_z₀_i(M.kts[j])
				if pH < eps()  # special case of pH=0
					∫ⱼ = M.insig[j] ? lnf_z₀ᵢⱼ + tot_hacking[j] : lnf_z₀ᵢⱼ
				else
					if M.insig[j]  # component from using or reverting to initial measurement
						if pD < eps()  # special case of pD=0
							∫ⱼ = lnf_z₀ᵢⱼ + tot_hacking[j]
						else
							∫ⱼ = lnpD + logsumexp(F[k,j] + B[k] for k ∈ eachindex(B))  # p-hacking contribution, integrating out z₀
							∫ⱼ = logsumexp(∫ⱼ, lnf_z₀ᵢⱼ + tot_hacking[j])
						end
					else
						∫ⱼ = logsumexp(F[k,j] + B[k] for k ∈ eachindex(B))  # p-hacking contribution, integrating out z₀
						∫ⱼ = logsumexp(∫ⱼ, lnf_z₀ᵢⱼ)
					end
				end
				∫[j,_i] = Cᵢ + ∫ⱼ
			end
		end
	end
  logsumexp!(tot_hacking, ∫), G  # sum across mixture components, into `tot_hacking` because it's the right size and already allocated
end

function HnFll(M::HnFmodel; pDFHR, kwargs...)
	∫, G = _HnFll(M; pDFHR, kwargs...)
	M.interpolate && (∫ = interpolate!(∫, BSpline(Cubic())).(M.zint))
	ℒ = (iszero(length(M.wt)) ? ThreadsX.sum(∫) : dot(M.wt,∫)) - xlog1py(M.N, -pDFHR[2]*G) + M.penalty(; pDFHR, kwargs...)
end


# simulate hack'n'file data generating process with integer m
# returns named tuple of true z's (ω), initial measurements (z✻), and reported results
# NaN = file-drawered
# if truncate=true (the default), restricts all return results to published studies
function HnFDGP(N::Int; p::Vector{Float64}, μ::Vector{Float64}=[0.], τ::Vector{Float64}, ν::Vector{Float64}, pDFHR::Vector{Float64}, σ::Vector{Float64}, m ::Vector{Float64}, modelabsz::Bool=false, truncate::Bool=true)
	ω = Vector{Float64}(undef,N)
	z₀ = similar(ω)
	z✻ = similar(ω)
	Tμτν = GenT.(μ, τ, ν)
	Threads.@threads for j ∈ eachindex(ω)
		@inbounds begin
			i = rand(Distributions.Categorical(p))
			ω[j] = ωⱼ = rand(Tμτν[i])  # pick mixture component
			z₀[j] = ωⱼ + rand(𝒩)  # initial measurement, variance 1 around ω
		end
	end

	pD, pF, _, pR = pDFHR
	pFD  = pF + pD
  pFDR = pF + pD + pR

	Threads.@threads for i ∈ eachindex(z₀)  # for each simulated study
		@inbounds begin
			z₀ⱼ = z₀[i]
			if abs(z₀ⱼ) > z̄  # if initial result significant, publish as is
				z✻[i] = z₀ⱼ
			else
				r = rand()
				if r < pF  # file-drawer initial, insignificant result?
					z✻[i] = NaN
				elseif r<pFDR  # publish initial, insigicant result
					z✻[i] = z₀ⱼ
				else  # p-hack
					while true
						batch = rand(Normal(z₀ⱼ, σ[]), Int(m[]))  # m measurements
						zᵢ = batch[findfirst(x->abs(x)==maximum(abs.(batch)), batch)]  # most significant of batch
						if abs(zᵢ) > z̄  # if significant, publish and stop
							z✻[i] = zᵢ
							break
						else
							r = rand()
							if r < pFDR  # after halting p-hacking search, file-drawer or publish latest, insignificant result, or revert to initial measurement
								z✻[i] = r<pF ? NaN : r<pFD ? zᵢ : z₀ⱼ
								break
							end
						end
					end
				end
			end
			modelabsz && (z✻[i] = abs(z✻[i]))
		end
	end

	if truncate
		keep = @. !isnan(z✻) # && abs(z✻)<10
		ω, z₀, z✻  = ω[keep], z₀[keep], z✻[keep]
	end
	(ω=ω, z₀=z₀, z✻=z✻)
end

@kwdef struct HnFresult<:RegressionModel
	estname::String
	modelabsz::Bool
	converged::Bool
	coefdict::NamedTuple
	coefnames::Vector{String}
	coef::Vector{Float64}
	vcov::Matrix{Float64}
	k::Int
	n::Int
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
function HnFfit(z::Vector, wt::Vector=Float64[]; d=1, interpres=0, NLegendre=50, NHermite=50, from::NamedTuple=NamedTuple(), xform::NamedTuple=NamedTuple(),
									methods::Vector=[NewtonTrustRegion()], estname="", modelabsz::Bool=false, penalty::Function=(; kwargs...)->0., kwargs...)

	println("\nModeling $estname data with $d mixture component(s)")
	
	# set starting values & parameter transformes, allowing caller to override defaults
	from  = merge((p=fill(1/d,d), μ=[0.]     , τ=collect(LinRange(1,d,d)), ν=fill(1.,d), pDFHR=fill(.25,4), σ=[1.]      , m=[2.]        ),  from)
  xform = merge((p=SimplextoRⁿ, μ=identity , τ=bcast(log)              , ν=bcast(log), pDFHR=SimplextoRⁿ, σ=bcast(log), m=bcast(log1m)), xform)

	M = HnFmodel(z, wt; d, modelabsz, interpres, NLegendre, NHermite, penalty)
	
	_from = pairs(from)
	fromxform = [xform[p](v) for (p,v) ∈ _from]  # starting values in optimization parameter space

	# indexes to extract individual parameter vectors from full parameter vector
	extractor = zip(keys(_from), Iterators.accumulate((ind,f)->f isa Number ? (last(ind)+1) : last(ind)+1:last(ind)+length(f), fromxform, init=0))

	xformer(x) = (p=>inverse(xform[p])(x[e]) for (p,e) ∈ extractor)  # map primary parameters into full model space, expressed as functions of optimization parameters, e.g. exp(log(σ))
	objective(x) = -HnFll(M; xformer(x)...)
	θ = vcat(fromxform...)

	res = nothing
	for method ∈ methods
		res = Optim.optimize(objective, θ, method, Optim.Options(; merge((iterations=250, show_trace=true), kwargs)...), autodiff=:forward)
		θ = Optim.minimizer(res)
	end

	invxform = θ -> [θ[e] |> inverse(xform[p]) for (p,e) ∈ extractor]
	coefdict_maker(v) = NamedTuple(p=>inverse(xform[p])(v[e]) for (p,e) ∈ extractor)
	coefdict = coefdict_maker(θ)

	function derived_stats(; p,μ,τ,ν,pDFHR,σ,m)
		pD, pF, pH, pR = pDFHR

		f(v) = ((ω,z₀)=v; pdf(𝒩,z₀ - ω) * p'pdf.(GenT.(μ,τ,ν), ω))  # f(z₀)
		g(v) = ((_,z₀)=v; f(v) / (1 - pH * diffcdf(Normal(z₀,σ[]),z̄,-z̄)))  # f * "shots on goal"
		I₀   = hcubature(f, [-100,-z̄], [100, z̄])[1] 
		S₂₄  = hcubature(f, [-100,-4], [100,-2])[1] + hcubature(f, [-100,2], [100,4])[1]  # actually marginally significant
		G    = hcubature(g, [-100,-z̄], [100, z̄])[1] 
		Sh₂₄ = pH * hcubature(v -> ((ω,z₀)=v; g(v) * (diffcdf(Normal(z₀,σ[]),4,-4)^m[] - diffcdf(Normal(z₀,σ[]),2,-2)^m[])), [-100,-z̄], [100,z̄])[1]  # p-hacked "marginally significant"

		[
			pF*G / I₀                     # fraction of insignificant studies file-drawered
			pF*G                          # fraction of all studies file-drawered
			pR*G / I₀ + pD                # fraction of insignificant published as is
			1 - (1-pH)*G/I₀               # fraction of initially insignificant that lead to published, significant, p-hacked results
			pD * (G/I₀ - 1)               # fraction of initially insignificant that lead to published, insignificant, p-hacked results

			(I₀ - (1-pH)*G) / (1 - pF*G)  # fraction of significant results that are p-hacked
			pD * (G - I₀) / (1 - pF*G)    # fraction of insignificant results that are p-hacked
			Sh₂₄ / (Sh₂₄ + S₂₄)           # p-hacked fraction of "marginally significant" in Star Wars (2<|z|<4)
		]
	end

	Δ = ForwardDiff.jacobian(v->vcat(invxform(v)..., derived_stats(;coefdict_maker(v)...)), θ)  # Jacobian of full model parameters & derived stats wrt optimization parameters
	H = ForwardDiff.hessian(objective, θ)  # Hessian of log likelihood wrt optimization parameters
	Vxform = try pinv(H) catch _ fill(NaN, size(H)) end  # covariance matrix of optimization parameters
	vcov = Δ * Vxform * Δ'  # covariance matrix of full model parameters
	vcov[diagind(vcov)] .= max.(0, vcov[diagind(vcov)])

	# se = NamedTuple([p=> iszero(length(e)) ? zeros(length(inverse(xform[p])(θ[e]))) :
	# 											(e isa Int ? ForwardDiff.derivative : ForwardDiff.jacobian)(inverse(xform[p]), θ[e]) |>
	# 												(Δᵢ -> Δᵢ isa Number ? sqrt0(Vxform[e,e])*abs(Δᵢ) : sqrt0.(diag(Δᵢ * Vxform[e,e] * Δᵢ')))
	# 									for (p,e) ∈ extractor])

	converged = Optim.converged(res)

	t = findall(x->abs(x)>.001, coefdict[:p])  # non-trivial mixture components
	if length(t) < d
		println("Dropping mixture components with negligible weight: keeping $(length(t)) of $d components")
		coefdict = (p=coefdict.p[t], μ=coefdict.μ, τ=coefdict.τ[t], ν=coefdict.ν[t], pDFHR=coefdict.pDFHR, σ=coefdict.σ, m=coefdict.m)
		I = vcat(t, 1+d, t.+(1+d), t.+(1+2d), 2+3d:size(vcov,1))  # indexes of kept parameters in full parameter vector
		vcov = vcov[I,I]
		M.d[] = d = length(t)
	end

	one2D = first(Unicode.graphemes("₁₂₃₄"),d)
	coefnames = vcat("p".*one2D, "μ", "τ".*one2D, "ν".*one2D, "pD", "pF", "pH", "pR", "σ", "m", "frac_insig_file_drawered", "overall_file_drawer_frac", 
										 "frac_insig_pubbed_as_is", "sig_p_hacked_frac", "insig_p_hacked_frac",
										 "p_hacked_frac_of_pubbed_insig", "p_hacked_frac_of_sig", "p_hacked_frac_of_marg_sig")

	G = _HnFll(M; coefdict...)[2]
	HnFresult(; estname, modelabsz, converged, coefdict, coefnames, coef=vcat(coefdict..., derived_stats(;coefdict...)...), vcov, k=length(θ), n=size(z,1), d, ll=-Optim.minimum(res))
end


function HnFplot(z, est, wt::Vector=Float64[]; NLegendre=50, NHermite=50, zplot::StepRangeLen=-5+1e-3:.01:5, ωplot::StepRangeLen=zplot, title::String="")
	t = est.coefdict
	kwargsω = (p=t.p, μ=t.μ, τ=t.τ, ν=t.ν)
	kwargsz = (pDFHR=t.pDFHR, σ=t.σ, m=t.m)
	kwargsz0 = (pDFHR=[1.,0.,0.,0.], σ=[1.], m=[1.])  # no distortion

	f = Figure(size=(1500,900))

	# empirical distribution of z's + model fit
	Axis(f[1,1], xlabel="z", ylabel="Density", limits=(est.modelabsz ? 0 : -10, 10, nothing,nothing))
	hist!(z, normalization=:pdf, bins=floor(Int,√size(z,1)), weights=length(wt)==0 ? Makie.automatic : wt, 
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
	lines!(zplot, fZcondΩ.(zplot, ω; kwargsz..., NLegendre), label="updating from prior + research distortion", color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)
	
	# distribution of ω | z=2
	_z = 2.
	Axis(f[2,1], xlabel="True z | reported z = $_z", ylabel="Density")
	lines!(ωplot, fΩcondZ.(ωplot, _z; kwargsω..., kwargsz0..., NLegendre, NHermite), label="updating from prior")
	lines!(ωplot, fΩcondZ.(ωplot, _z; kwargsω..., kwargsz..., NLegendre, NHermite), label="updating from prior + research distortion", color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)
	
	# frequentist equal-tailed CI's as fn of z--Andrews & Kasy (2014), Figure 2
	CIs0 = Cquant.([.025 .5 .975], zplot; rtol=.0001, kwargsz0..., NLegendre)
	CIs  = Cquant.([.025 .5 .975], zplot; rtol=.0001, kwargsz..., NLegendre )
	Axis(f[1,3], xlabel="Reported z", ylabel="Point estimate and 95% CI for true z", xticks=-5:5, yticks=-6:6)
	lines!(zplot, CIs0[:,1], color=Makie.wong_colors()[1], label="No adjustment")
	lines!(zplot, CIs0[:,2], color=Makie.wong_colors()[1], linestyle=:dash)
	lines!(zplot, CIs0[:,3], color=Makie.wong_colors()[1])
	lines!(zplot, CIs[:,1], color=Makie.wong_colors()[6], label="Adjusting for research distortion")
	lines!(zplot, CIs[:,2], color=Makie.wong_colors()[6], linestyle=:dash)
	lines!(zplot, CIs[:,3], color=Makie.wong_colors()[6])
	try
		lb = linear_interpolation(CIs[:,1],zplot)(0.)  # McCrary, Christensen, and Fanelli (2016)-style z thresholds for p<.05
		ub = linear_interpolation(CIs[:,3],zplot)(0.)
		scatter!([lb;ub],[0.;0], color=Makie.wong_colors()[6])
		text!(lb, 0., text=format("{:03.2f}", lb), align=(:right, :bottom), fontsize=18)
		text!(ub, 0., text=format("{:03.2f}", ub), align=(:left, :top), fontsize=18)
	catch e
	end
	axislegend(position=:lt, framevisible = false)

	# Posterior mean of ω as fn of Z
	pplot0 = EΩcondZ(zplot; kwargsω..., kwargsz0..., NLegendre, NHermite)
	pplot  = EΩcondZ(zplot; kwargsω..., kwargsz..., NLegendre, NHermite)
	Axis(f[2,2], xlabel="Reported z", ylabel="Expected true z")
	lines!(zplot, zplot, label="As is", color=Makie.wong_colors()[3])
	lines!(zplot, pplot0, label="Updating from prior", color=Makie.wong_colors()[1])
	lines!(zplot, pplot , label="updating from prior + research distortion", color=Makie.wong_colors()[6])
	axislegend(position=:lt, framevisible = false)

	# E[ω] discount
	Axis(f[2,3], xlabel="Reported z", ylabel="Discount multiplier" #=, yticks=0:.1:1.5 , limits=(nothing,nothing,0.,nothing)=#)
	lines!(zplot[zplot.>.2], Float16.(pplot0[zplot.>.2]./zplot[zplot.>.2]), label="updating from prior")  # https://discourse.julialang.org/t/range-step-cannot-be-zero/66948/11?u=droodman
	lines!(zplot[zplot.>.2], Float16.(pplot[zplot.>.2]./zplot[zplot.>.2]), label="updating from prior + research distortion", color=Makie.wong_colors()[6])
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


#
# confirm match between model and simulation
#

p = [.7,.3]
μ = [0.7]
τ = [1.2,2.7]
ν = [20., 20.]
pD = .4
pF = .3
pH = .2
pR = .1
σ = [.2]
m = [5.]
d = length(p)
modelabsz = false
pDFHR = [pD, pF, pH, pR]
kwargs = (p=p, μ=μ, τ=τ, ν=ν, pDFHR=pDFHR, σ=σ, m=m, modelabsz=modelabsz)

n = 100_000
Random.seed!(1232)
sim = HnFDGP(n; kwargs...)

f = Figure()
Axis(f[1,1], limits=(modelabsz ? 0 : -10, 10, nothing,nothing))
hist!(sim.z✻[abs.(sim.z✻).<100], bins=10*2*100, normalization=:pdf)
zplot = (modelabsz ? 0 : -10):.01:10
lines!(zplot, fZ(zplot; NHermite=50, kwargs...), color=:orange, label="True parameters")
f|>display

penalty(; m::Vector{T}, τ::Vector{T}, σ::Vector{T}, kwargs...) where {T} = logpdf(Normal(0,5), log(m[])) + logpdf(Normal(0,5), log(σ[])) + sum(logpdf(Normal(0,5), log(τᵢ)) for τᵢ ∈ τ) 
res = HnFfit(sim.z✻; d, modelabsz, penalty, estname="simulated", extended_trace=false)  # penalized maximum likelihood
print(res.coefdict)
lines!(zplot, fZ(zplot; modelabsz, res.coefdict...)[:,1], color=:green, label="Estimated parameters")

f[0, :] = Label(f, "Simulation vs model")
axislegend(position=:lt, framevisible=false)
colsize!(f.layout, 1, Relative(1))
f |> display

#
# model real data
#

@time begin
	# penalty function for parameters that can generate singularities
  penalty(; m::Vector{T}, τ::Vector{T}, σ::Vector{T}, kwargs...) where {T} = logpdf(Normal(0,5), log(m[])) + logpdf(Normal(0,5), log(σ[])) + sum(logpdf(Normal(0,5), log(τᵢ)) for τᵢ ∈ τ) 

	# van Zwet, Schwab, and Senn (2021) data, https://osf.io/xq4b2
	df = DataFrame(CSV.File("data/van Zwet, Schwab, and Senn 2021/CochraneEffects.csv"))
	@. @subset!(df, abs(:z)<10 && :"outcome.nr"==1 && :RCT=="yes" && :"outcome.group"=="efficacy")  # vZSS used 20 not 10
	Random.seed!(29384)
	df = combine(groupby(df, :"study.id.sha1"), :z => sample => :z)  # randomly choose among primary outcomes
  results = [HnFfit(df.z; d, penalty, NLegendre=50, estname="vZZS$d") for d ∈ 1:3]
	vZSS = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	HnFplot(df.z, vZSS; title="van Zwet, Schwab, and Senn (2021) data")

	# Schuemie et al. (2013), https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fsim.5925&file=Appendix+G+Revision.xlsx
	df = DataFrame(XLSX.readtable("data/Schuemie et al. 2013/appendix g revision.xlsx", "NeatTable", first_row=2, infer_eltypes=true)...)
	@. df.z = log(df."Effect estimate") / (log(df."Upper bound of 95% CI" / df."Lower bound of 95% CI") / 2z̄)
	@. @subset!(df, abs(:z)<10)
	disallowmissing!(df, :z)
	results = [HnFfit(df.z; d, penalty, estname="Setal$d") for d ∈ 1:3]
	Setal = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	HnFplot(df.z, Setal; title="Schuemie et al. (2013) data")

	# Star Wars, doi.org/10.1257/app.20150044, openicpsr.org/openicpsr/project/113633/version/V1/view?path=/openicpsr/113633/fcr:versions/V1/brodeur_le_sangnier_zylberberg_replication/Data/Final/final_stars_supp.dta&type=file
	df = DataFrame(CSV.File("data/Brodeur et al. 2016/final_stars_supp.csv"))
	df.z = df.coefficient_num ./ df.standard_deviation_num
	@. @subset!(df, lowercase(:main)=="yes" && !ismissing(df.z) && abs(df.z)<20)
	disallowmissing!(df, :z)
	results = [HnFfit(df.z, df.weight_table; d, penalty, estname="SW$d") for d ∈ 1:3]
	SW = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	HnFplot(df.z, SW, df.weight_table; title="Brodeur et al. (2016) data")

	# Brodeur, Cook, and Heyes 2020, DOI 10.1257/aer.20190687, openicpsr.org/openicpsr/project/120246/version/V1/view?path=/openicpsr/120246/fcr:versions/V1/MM-Data.dta&type=file
	df = DataFrame(CSV.File("data/Brodeur, Cook, and Heyes 2020/MM Data.csv"))
	df.z = df.mu ./ df.sd  # .* (2*rand(Bernoulli(.5),size(df,1)).-1)
	@. @subset!(df, !ismissing(:z) && !isnan(:z) && abs(:z)<10)
	disallowmissing!(df, :z)
	hist(df.z, bins=100) |> display
	df.z .= abs.(df.z)
	results = [HnFfit(df.z; d, penalty, xform=(μ=get0,), modelabsz=true, estname="BCH$d") for d ∈ 1:3]
	BCH = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	HnFplot(df.z, BCH; title="Brodeur, Cook, and Heyes (2020) data")

	# Arel-Bundock et al. 2026
	df = DataFrame(CSV.File("data/Arel-Bundock et al. 2026/arel-bundock_briggs.csv"))
	@. @subset!(df, !ismissing.(df.z_stat) .&& abs.(df.z_stat).<10)
	results = [HnFfit(df.z_stat; d, penalty, estname="ABetal$d") for d ∈ 1:3]
	ABetal = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	@time HnFplot(df.z_stat, ABetal; title="Arel-Bundock et al. (2026) data")

	# Vivalt 2020, DOI 10.1093/jeea/jvaa019, https://figshare.com/articles/dataset/Replication_files_for_How_Much_Can_We_Generalize_from_Impact_Evaluations_/12048600/1
	df = DataFrame(CSV.File("data/Vivalt 2020/data_unstandardized.csv"))
	df.z = df.treatmentcoefficient ./ df.treatmentstandarderror
	@. @subset!(df, abs(:z)<10)
	results = [HnFfit(df.z; d, penalty, estname="V$d") for d ∈ 1:3]
	V = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	HnFplot(df.z, V; title="Vivalt (2020) data")

	# Gerber and Malhotra 2008, https://www.nowpublishers.com/article/details/supplementary-info/100.00008024_supp.rar
	df = [DataFrame(load("data/Gerber and Malhotra 2008/AJPS_Data.xls", "All Studies"))[2:end,[:x4,:x6]] ;
				DataFrame(load("data/Gerber and Malhotra 2008/APSR_Data.xls", "All Studies"))[2:end,[:x4,:x6]] ]
	@. @subset!(df, !ismissing(:x4))
	df.z = Float64.(df.x6)
	@. @subset!(df, abs.(:z)<10)
	results = [HnFfit(df.z; d, penalty, estname="GM$d") for d ∈ 1:3]
	GM = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]
	HnFplot(df.z, GM; title="Gerber & Malhotra (2008) data")

	# Georgescu and Wren 2018 ~1M sample, doi:10.1093/bioinformatics/btx811, https://github.com/agbarnett/intervals/blob/master/data/Georgescu.Wren.RData
	df = DataFrame(RData.load("data/Georgescu and Wren 2018/Georgescu.Wren.RData")["complete"])
	@. df.ci_level[ismissing(df.ci_level) || df.ci_level==.0095 || df.ci_level==.05] = .95
	@. df.z = log(df.mean) / (ifelse(ismissing(df.lower) || iszero(df.lower), log(df.upper / df.mean), log(df.upper / df.lower) / 2) / cquantile(𝒩, (1 - df.ci_level)/2))
	@. @subset!(df, !ismissing(:z) && !ismissing(:lower) && iszero(:mistake) && abs(:z)<10.)  # van Zwet & Cator Figure 1 stops at 10
	# @. @subset!(df, :source!="Abstract")
	results = [HnFfit(df.z;            d, penalty, interpres=1000          , estname="GW$d") for d ∈ 1:3]  # approximate fits by interpolating loglik over z
	results = [HnFfit(df.z; results[d].d, penalty, from=results[d].coefdict, estname="GW$d") for d ∈ 1:3]
	GW = results[argmin(isnan(t.BIC) ? Inf : t.BIC for t ∈ results)]  # BIC minimizer
	HnFplot(df.z, GW; title="Georgescu and Wren (2018) data")

	table = regtable(GW, Setal, GM, SW, BCH, ABetal, vZSS, V;
							estim_decoration = (coef,p)->coef,  # no stars
							regression_statistics = [Nobs #=, Converged, LogLikelihood, BIC=#],
							print_estimator_section = false,
							keep = ["p₁", "p₂", "p₃", "p₄", "μ", "τ₁", "τ₂", "τ₃", "τ₄", "ν₁", "ν₂", "ν₃", "ν₄", "pF", "pH", "pD", "pR", "σ", "m", "frac_insig_file_drawered", "frac_insig_pubbed_as_is", "p_hacked_frac_of_pubbed_insig", "p_hacked_frac_of_sig", "p_hacked_frac_of_marg_sig"],
							estimformat = "%0.3g",
							statisticformat = "%0.3g",
							number_regressions = false,
							file = "output/results.txt")
end