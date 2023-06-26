using Random, Distributions, Interpolations, Base.Iterators, FastGaussQuadrature, BenchmarkTools, Optim, LogExpFunctions, Plots, CSV, DataFrames, DataFramesMeta, ForwardDiff, LinearAlgebra, Roots, QuadGK, Statistics, ThreadsX

const Z̄ = 1.9599639845401

@inline diffcdf(N,b,a) = cdf(N,b) - cdf(N,a)
@inline diffpdf(N,b,a) = pdf(N,b) - pdf(N,a)

import Base.rand, Distributions.pdf, Distributions.logpdf, Distributions.cdf, Distributions.logcdf, Distributions.ccdf, Distributions.logccdf, Statistics.quantile

# type to represent distribution of maximum of m normal draws, Float64 only
struct MaxNormal <: Distribution{Univariate, Continuous}
	m::Float64  # number of draws over which to take max
	invm::Float64
	μ::Float64  # mean
	σ::Float64  # sd
	𝒩::Normal{Float64}
	MaxNormal(m=1., μ=0., σ=1.) = new(m,1/m,μ,σ,Normal(μ,σ))
end
cdf(s::MaxNormal, x::Float64) = cdf(s.𝒩,x) ^ s.m
logccdf(s::MaxNormal, x::Float64) = log1mexp(s.m * logcdf(s.𝒩,x))
ccdf(s::MaxNormal, x::Float64) = exp(logccdf(s,x))
pdf(s::MaxNormal, x::Float64) = pdf(s.𝒩,x) * s.m * cdf(s.𝒩,x)^(s.m-1)
logpdf(s::MaxNormal, x::Float64) = logpdf(s.𝒩,x) + log(s.m) + (s.m-1) * logcdf(s.𝒩,x)
quantile(s::MaxNormal, x::Float64) = quantile(s.𝒩, x^s.invm)
rand(rng::AbstractRNG, s::MaxNormal) = quantile(s.𝒩, rand(rng)^s.invm)

struct MinNormal <: Distribution{Univariate, Continuous}
	m::Float64  # number of draws over which to take max
	invm::Float64
	μ::Float64  # mean
	σ::Float64  # sd
	𝒩::Normal{Float64}
	MinNormal(m=1., μ=0., σ=1.) = new(m,1/m,μ,σ,Normal(μ,σ))
end
cdf(s::MinNormal, x::Float64) = exp(logcdf(s,x))
logcdf(s::MinNormal, x::Float64) = log1mexp(s.m * logccdf(s.𝒩,x))
ccdf(s::MinNormal, x::Float64) = ccdf(s.𝒩,x) ^ s.m
pdf(s::MinNormal, x::Float64) = pdf(s.𝒩,x) * s.m * ccdf(s.𝒩,x)^(s.m-1)
logpdf(s::MinNormal, x::Float64) = logpdf(s.𝒩,x) + log(s.m) + (s.m-1) * logccdf(s.𝒩,x)
quantile(s::MinNormal, x::Float64) = cquantile(s.𝒩, (1-x)^s.invm)
rand(rng::AbstractRNG, s::MinNormal) = cquantile(s.𝒩, rand(rng)^s.invm)

# to parameterize an n-vector of probabilities summing to 1 with an unbounded (n-1)-vector, apply logistic transform to latter, then map to squared spherical coordinates
# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates, https://math.stackexchange.com/questions/2861449/parameterizations-of-the-unit-simplex-in-mathbbr3
function RⁿtoSimplex(q::Vector{T}) where T
	_q = logistic.(q)
	p = Vector{T}(undef, length(q)+1)
	Πsin² = one(T)
	for i ∈ eachindex(_q)
		cos² = cospi(_q[i])^2
		p[i] = Πsin² * cos²
		Πsin² *= one(T) - cos²
	end
	p[end] = Πsin²
	p
end
function SimplextoRⁿ(p::Vector{T}) where T
	q = Vector{T}(undef, length(p)-1)
	sum = p[end]
	for i ∈ reverse(eachindex(q))
		sum += p[i]
		q[i] = acos(√(p[i] / sum)) / π
	end
	q .= logit.(q)
end

# unlogged likelihood for a single observation. For graphs.
function HnFl(z; p::Vector, μ::Vector, τ::Vector, pF₀, pL₀, pU₀, kL=0, kU=0, m=1, truncate=true)
	Z̄L	= min(Z̄, 1/kL-Z̄)
	Z̄U	= min(Z̄, 1/kU-Z̄)
	pD₀ = 1 - pF₀
	length(p) < length(μ) && (p = [p; 1-sum(p)])
	σ² = 1 .+ τ.^2
	𝒩  = Normal()
	𝒩μ = @. Normal(μ, √σ²)
	𝒩ω = @. NormalCanon(z + μ/τ^2, 1 + 1/τ^2)
	if abs(z) ≥ Z̄
		result = 0.
		@inbounds for (pᵢ,𝒩μᵢ,𝒩ωᵢ) ∈ zip(p,𝒩μ,𝒩ω)
			if z < 0
				result += pᵢ * pdf(𝒩μᵢ, z) * (1 + m * pL₀ * quadgk(ω -> (a = pdf(𝒩ωᵢ,ω) * ((1-kL*(Z̄+ω))*diffcdf(𝒩,Z̄L-ω,-Z̄-ω) + kL*diffpdf(𝒩,Z̄L-ω,-Z̄-ω)) * ccdf(𝒩,z-ω)^(m-1) / (1-ccdf(𝒩,-Z̄-ω)^m);
																										              isnan(a) || isinf(a) ? 0. : a), 
																                            -Inf, Inf)[1])
			else
				result += pᵢ * pdf(𝒩μᵢ, z) * (1 + m * pU₀ * quadgk(ω -> (a = pdf(𝒩ωᵢ,ω) * ((1-kU*(Z̄-ω))*diffcdf(𝒩,Z̄-ω,-Z̄U-ω) - kU*diffpdf(𝒩,Z̄-ω,-Z̄U-ω)) * cdf(𝒩,z-ω)^(m-1) / (1 - cdf(𝒩, Z̄-ω)^m);
																										              isnan(a) || isinf(a) ? 0. : a), 
																                            -Inf, Inf)[1])
			end
		end
	else
		result = dot(p, pdf.(𝒩μ, z)) * pD₀ * (1 - pL₀ * max(0, 1-kL*(Z̄+z)) - pU₀ * max(0, 1-kU*(Z̄-z)))
	end
	truncate && (result /= 1 - pF₀ * dot(p, @. diffcdf(𝒩μ,Z̄,-Z̄) - pL₀ * ((1-kL*(Z̄+μ)) * diffcdf(𝒩μ,Z̄L,-Z̄) + kL * σ² * diffpdf(𝒩μ,Z̄L,-Z̄)) - 
	                                                               pU₀ * ((1-kU*(Z̄-μ)) * diffcdf(𝒩μ,Z̄,-Z̄U) + kU * σ² * diffpdf(𝒩μ,Z̄,-Z̄U))   ))
	result
end

# object to hold pre-computed stuff for log likelihood computation
struct HnFstuff
	D::Int  # number of mixture components
	z::Vector{Float64}  # all data
	zC::Vector{Float64}  # just the central, insignificant z's
	N::Int; NC::Int; NL::Int; NU::Int  # number of z's, insigficant z's, lower significant z's, upper
	knots::LinRange  # interpolation knots in [Z̄,max]
	spline::Interpolations.InterpolationType  # type of interpolation
	zSint::NTuple{2,Vector{Float64}}  # lower- & upper-tail significant z values mapped to cardinal knot numbering space since interpolate() is faster with cardinally spaced knots
	X::Vector{Float64}; W::Vector{Float64}  # quadrature nodes & weights
	lnW::Vector{Float64}
end
# constructor
function HnFstuff(z::Vector{Float64}; D::Int, interpres::Int, quadnodes::Int)
	zC = z[abs.(z) .< Z̄]

	s = Z̄ - 3/interpres; e = max(10,maximum(z))+.1
	knots = s : 1/interpres : e  # LinRange(s, e, ceil(Int, (e - s) * interpres) + 1)
	zSint = (-z[z .≤ -Z̄] .- s) .* interpres .+ 1, (z[z .≥ Z̄] .- s) .* interpres .+ 1  # map tail z values to knot numbering 1, 2, ... for Z̄-3/interpres, Z̄-2/interpres, ...
	
	X, W = gausshermite(quadnodes)
	W ./= √π
	
	HnFstuff(D, z, zC, length(z), length(zC), length.(zSint)..., knots, BSpline(Quadratic(Free(OnGrid()))), zSint, X, W, log.(W))
end

try Base.delete_method.(methods(HnFll)) catch end
# bulk log probabilities as function of data & parameters, for estimation
function HnFll(o::HnFstuff, p::Vector{T}, μ::Vector, τ::Vector{T}, pF₀::T, pL₀::T, pU₀::T, kL::T, kU::T, m::T) where T<:Real
	τ = exp.(τ)
	kL = exp(kL)
	kU = exp(kU)
	m = exp(m); mm1 = m - one(T)
	p = RⁿtoSimplex(p)
	pF₀ = logistic(pF₀)
	pL₀ = logistic(pL₀)
	pU₀ = logistic(pU₀)
	pL₀+pU₀>1 && return(T(NaN))
	pD₀ = 1 - pF₀
	pH = [m*pL₀*exp(kL*(kL/2-Z̄)), m*pU₀*exp(kU*(kU/2-Z̄))]
	LC = fill(zero(T), o.NC)  # likelihood for insignificant obs
	LS = fill(zero(T), o.NL), fill(zero(T), o.NU)  # for significant obs, left & right tails

	σ² = 1 .+ (τ² = τ.^2)
	𝒩  = Normal()
	𝒩μ = Normal.(μ, .√σ²)

	for (pᵢ,μᵢ,τᵢ²,σᵢ²,𝒩μᵢ) ∈ zip(p,μ,τ²,σ²,𝒩μ)
		# math on integration and interpolation points, outside loops
		Ω  = o.X * √(2τᵢ² / σᵢ²)  # 1st-order component of change of variables from pdf(Normal(ω)) to exp(-x²) for Gauss-Hermite quadrature
		ΩL = -Z̄ .- Ω
		ΩU =  Z̄ .- Ω
		𝒩Ω = Normal.(Ω)

		buf = Vector{T}(undef, length(o.knots))  # pre-allocating this hampers automatic differentiation since type changes

		# lower tail
		kt1 = collect((μᵢ/τᵢ² .- o.knots) * -τᵢ²/σᵢ²)  #    -(z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points -- negated 0th-order component of change of variables for quadrature
		kt2 = collect(kt1 - o.knots                 )  # z - (z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		fill!(buf, zero(T))
		@inbounds Threads.@threads for j ∈ eachindex(o.knots)
			kt1j, kt2j = kt1[j], kt2[j]
			for (𝒩ω,ω,ωl,ωu,lnw) ∈ zip(𝒩Ω,Ω,ΩL,ΩU,o.lnW)  # quadrature integration
				buf[j] += exp(lnw - kL * (ω - kt1j) + logdiffcdf(𝒩, kt1j + ωu + kL, kt1j + ωl + kL) + mm1 * logccdf(𝒩ω, kt2j) - log1mexp(m * logccdf(𝒩, kt1j + ωl)))
			end
		end
		@. buf = pᵢ * pdf(𝒩μᵢ, -o.knots) * (one(T) + pH[1] * buf)
		LS[1] .+= interpolate!(buf, o.spline).(o.zSint[1])  # likelihoods for significant observations

		# upper tail
		kt1 .= collect((μᵢ/τᵢ² .+ o.knots) * -τᵢ²/σᵢ²)  #    -(z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		kt2 .= collect(kt1 + o.knots                 )  # z - (z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		fill!(buf, zero(T))
		@inbounds Threads.@threads for j ∈ eachindex(o.knots)
			kt1j, kt2j = kt1[j], kt2[j]
			for (𝒩ω,ω,ωl,ωu,lnw) ∈ zip(𝒩Ω,Ω,ΩL,ΩU,o.lnW)  # quadrature integration
				buf[j] += exp(lnw + kU * (ω - kt1j) + logdiffcdf(𝒩, kt1j + ωu - kU, kt1j + ωl - kU) + mm1 * logcdf(𝒩ω, kt2j) - log1mexp(m * logcdf(𝒩, kt1j + ωu)))
			end
		end
		@. buf = pᵢ * pdf(𝒩μᵢ, o.knots) * (one(T) + pH[2] * buf)
		LS[2] .+= interpolate!(buf, o.spline).(o.zSint[2])  # likelihoods for significant observations

		@. LC += pᵢ * pdf(𝒩μᵢ, o.zC)  # likelihoods for center/insignificant observations
	end
# XXX move interpolation out of above loop and interpolate log likelihood instead?
# XXX precompute Z̄ + o.zC, Z̄ - o.zC
	mapreduce(v->ThreadsX.mapreduce(log, +, v, init=zero(T)), +, (LC, LS...)) +
		o.NC * log(pD₀) + mapreduce(z->log1p(- pL₀ * exp(-kL * (Z̄ + z)) - pU₀ * exp(-kU * (Z̄ - z))), +, o.zC, init=zero(T)) - 
    xlog1py(o.N, -pF₀ * dot(p, @. diffcdf(𝒩μ,Z̄,-Z̄) - pL₀ * exp(kL*(σ²*kL/2-μ-Z̄)) * diffcdf(𝒩μ, Z̄+σ²*kL, -Z̄+σ²*kL) -
		                                                  pU₀ * exp(kU*(σ²*kU/2+μ-Z̄)) * diffcdf(𝒩μ, Z̄-σ²*kU, -Z̄-σ²*kU)))
end

# log likelihood--function of parameters only
negHnFll(o)        = v -> -HnFll(o, v[1:o.D-1], v[o.D:2*o.D-1], v[2*o.D:3*o.D-1], v[3*o.D:end]...)
negHnFll0cent(o)   = v -> -HnFll(o, v[1:o.D-1], zeros(Float64,o.D), v[o.D:2*o.D-1], v[2*o.D:end]...)  # impose μ=0
negHnFllSharedμ(o) = v -> -HnFll(o, v[1:o.D-1], fill(v[o.D],o.D), v[o.D+1:2*o.D], v[2*o.D+1:end]...) # impose shared μ

function HnFCDF(o::HnFstuff, z::T, p::Vector{T}, μ::Vector{T}, τ::Vector{T}, pD₀::T, pF::T, U) where T<:Number
	pH = 1 - pD - pF

	𝒩H = Normal.(√2τ * o.X' .+ μ)
	𝒩  = Normal.(μ, .√(1 .+ τ.^2))

	a = cdf.(𝒩H, -Z̄)
	b = ccdf.(𝒩H, Z̄)
	if z ≤ Z̄  # tails
		result = pH * p'* cdf.(𝒩, min(z,-Z̄)                          ) + pH * (p' * ( cdf.(𝒩H, min(z,-Z̄))       .* (one(T) .- a .- b) ./ (a .+ U .* b)) * o.W)
	else
		result = pH * p'*(cdf.(𝒩,       -Z̄) + cdf.(𝒩,z) - cdf.(𝒩,Z̄)) + pH * (p' * ((a + U * (cdf.(𝒩H,z) - b)) .* (one(T) .- a .- b) ./ (a .+ U .* b)) * o.W)
	end
	z > -Z̄ &&  # central bit
		(result += pD * (p' * (cdf.(𝒩, min(z,Z̄)) - cdf.(𝒩,-Z̄))))
	result / (1 - pF * (p' * (cdf.(𝒩,Z̄) - cdf.(𝒩,-Z̄))))
end

# f(z|ω)
function fZcondΩ(z, ω; pF₀, pL₀, pU₀, kL, kU, m, truncate=true)
	pD₀ = 1 - pF₀
	result = abs(z) < Z̄ ? pdf(Normal(ω),z) * pD₀ * (1 - pL₀ * exp(-kL*(Z̄+z)) - pU₀ * exp(-kU*(Z̄-z))) :
							          pdf(Normal(ω),z) + exp(z < 0 ? logpdf(MinNormal(m,ω),z) - logcdf( MinNormal(m,ω),-Z̄) + log(pL₀) + kL*(kL/2-Z̄-ω) + logdiffcdf(Normal(ω-kL),Z̄,-Z̄) :
											                                 logpdf(MaxNormal(m,ω),z) - logccdf(MaxNormal(m,ω), Z̄) + log(pU₀) + kU*(kU/2-Z̄+ω) + logdiffcdf(Normal(ω+kU),Z̄,-Z̄)  )
	truncate && (result /= (1 - pF₀ * (diffcdf(Normal(ω), Z̄,-Z̄) - pL₀ * exp(kL*(kL/2-Z̄-ω)) * diffcdf(Normal(ω-kL),Z̄,-Z̄) - pU₀ * exp(kU*(kU/2-Z̄-ω)) * diffcdf(Normal(ω+kU),Z̄,-Z̄))))
	isnan(result) || isinf(result) ? 0. : result
end

# F(z|ω)
function FZcondΩ(z, ω; pF₀, pL₀, pU₀, kL, kU, m)
	pD₀ = 1 - pF₀
	𝒩 = Normal(ω)
	D = diffcdf(Normal(ω), Z̄,-Z̄) - pL₀ * exp(kL*(kL/2-Z̄-ω)) * diffcdf(Normal(ω-kL),Z̄,-Z̄) - 
	                               pU₀ * exp(kU*(kU/2-Z̄-ω)) * diffcdf(Normal(ω+kU),Z̄,-Z̄)  # P[no p-hack]
	if z > Z̄  # tails
		𝒩max = MaxNormal(m,ω)
		result = 1 - (pU₀ * exp(logccdf(𝒩max,z) - logccdf(𝒩max,Z̄) + kU*(kU/2-Z̄+ω) + logdiffcdf(𝒩, Z̄, -Z̄)) + ccdf(𝒩,z)) / (1 - pF₀ * D)
	else
		if z < -Z̄
			𝒩min = MinNormal(m,ω)
			result =    pL₀ * exp(logcdf(𝒩min, z) - logcdf(𝒩min, -Z̄) + kL*(kL/2-Z̄-ω) + logdiffcdf(𝒩, kL+Z̄, kL-Z̄)) + cdf(𝒩,z)
		else
			result =    pL₀ * exp(                                       kL*(kL/2-Z̄-ω) + logdiffcdf(𝒩, kL+Z̄, kL-Z̄)) + cdf(𝒩,-Z̄) + 
			                pD₀ * (diffcdf(Normal(ω), z,-Z̄) - pL₀ * exp(kL*(kL/2-Z̄-ω)) * diffcdf(Normal(ω-kL),z,-Z̄) - 
											                                  pU₀ * exp(kU*(kU/2-Z̄-ω)) * diffcdf(Normal(ω+kU),z,-Z̄)  )
		end
		result /= 1 - pF₀ * D
	end
	result
end


# f(z), f(ω), f(ω|z), E[ω|z]
fZ = HnFl
fΩ(ω; p, μ, τ) = p'pdf.(Normal.(μ,τ), ω)
fΩcondZ(ω, z; p, μ, τ, kwargs...) = fZcondΩ(z, ω; kwargs..., truncate=false) * fΩ(ω; p, μ, τ) / fZ(z; p, μ, τ, kwargs..., truncate=false)
EΩcondZ(z; p, μ, τ, kwargs...) = quadgk(ω->ω * fΩcondZ(ω, z; p, μ, τ, kwargs...), -Inf, Inf)[1]

# CIs
Cquant(α, z; kwargs...) = find_zero(ω -> α - FZcondΩ(z, ω; kwargs...), (-20,20))
CI(α, z; kwargs...) = Cquant(α/2, z; kwargs...), Cquant(1-α/2, z; kwargs...)


function HnFDGP(N; p::Vector, μ::Vector, τ::Vector, pF₀, pL₀, pU₀, kL=0, kU=0, m=1, truncate=true, ω=NaN)
	isone(length(μ)) && (μ = fill(μ[], length(τ)))
	length(p) < length(μ) && (p = [p; 1-sum(p)])
	I = rand(Categorical(p), N)
	Ω = isnan(ω) ? map(i->rand(Normal(μ[i], τ[i])), I) : fill(ω,N)
	Z✻ = rand.(Normal.(Ω))
	Z = similar(Z✻)
	@inbounds Threads.@threads for i ∈ eachindex(Z✻)
		Z✻ᵢ = Z✻[i]
		if abs(Z✻ᵢ) > Z̄
			Z[i] = Z✻ᵢ  # publish significant result as is
		else
			pL = pL₀ * max(0, 1-kL*(Z̄ + Z✻ᵢ))
			pU = pU₀ * max(0, 1-kU*(Z̄ - Z✻ᵢ))
			pF = pF₀ * (1 - pL - pU)
			pD = 1 - pL - pU - pF
			r = rand()
			if r < pD
				Z[i] = Z✻ᵢ  # publish insignificant result as is
			elseif r < pD + pF
				Z[i] = NaN  # file-drawer
			elseif r < pD + pF + pL
				𝒩 = MinNormal(m,Ω[i])
				Z[i] = quantile(𝒩, rand(Uniform(0., cdf(𝒩, -Z̄))))  # hack to lower tail
			else
				𝒩 = MaxNormal(m,Ω[i])
				Z[i] = quantile(𝒩, rand(Uniform(cdf(𝒩, Z̄), 1.)))  # hack to upper tail
			end
		end
	end
	if truncate
		keep = @. !isnan(Z) && abs(Z)<10
		Ω=Ω[keep]
		Z✻=Z✻[keep]
		Z=Z[keep]
	end
	(Ω=Ω, Z✻=Z✻, Z=Z)  # named tuple with results
end


# confirm match between model and simulation
pF₀ = .3
pL₀ = .3
pU₀ = .3
kL = 10.
kU = 10.
p = [1.]
μ = [0.]
τ = [2.]
m = 5.

z = HnFDGP(3_000_000; p, μ, τ, pF₀, pL₀, pU₀, kL, kU, m, truncate=true).Z

histogram(z, normalize=:pdf, legend=false)
zplot = -10:.1:10
pplot = map(z->HnFl(z; p, μ, τ, pF₀, pL₀, pU₀, kL, kU, m, truncate=true), zplot)
plot!(zplot, pplot)
plot!(zplot, map(z->exp(-negHnFll(HnFstuff([z], D=length(μ), interpres=300, quadnodes=25))(vcat(SimplextoRⁿ(p),μ,log.(τ),logit(pF₀),logit(pL₀),logit(pU₀),log(kL),log(kU),log(m)))),zplot))

o = HnFstuff(z, D=length(μ), interpres=300, quadnodes=25)
@time res = optimize(negHnFll(o), vcat(SimplextoRⁿ(p),μ,log.(τ),logit(pF₀),logit(pL₀),logit(pU₀),log(kL),log(kU),log(m)), LBFGS(), autodiff=:forward)
θ₂ = Optim.minimizer(res)
p̂, μ̂ , τ̂ , p̂F₀, p̂L₀, p̂U₀, k̂L, k̂U, m̂ = RⁿtoSimplex(θ₂[1:o.D-1]), θ₂[o.D:2*o.D-1], exp.(θ₂[2*o.D:3*o.D-1]), logistic(θ₂[3*o.D]), logistic(θ₂[3*o.D+1]), logistic(θ₂[3*o.D+2]), exp(θ₂[3*o.D+3]), exp(θ₂[3*o.D+4]), exp(θ₂[3*o.D+5])
p̂D₀ = 1 - p̂F₀
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂F₀=p̂F₀, p̂L₀=p̂L₀, p̂U₀=p̂U₀, k̂L=k̂L, k̂U=k̂U, m̂=m̂))

plot!(zplot, map(z->HnFl(z; p=p̂, μ=μ̂ , τ=τ̂ , pF₀=p̂F₀, pL₀=p̂L₀, pU₀=p̂U₀, kL=k̂L, kU=k̂U, m=m̂), zplot))


# data prep
df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Clients & prospects\GiveWell\Noisy data\Georgescu.Wren.csv"))
@. df.cilevel[ismissing(df.cilevel) || df.cilevel==.0095 || df.cilevel==.05] = .95
@. df.z = log(df.mean) / (ifelse(ismissing(df.lower) || iszero(df.lower), log(df.upper / df.mean), log(df.upper / df.lower) / 2) / cquantile(Normal(), (1 - df.cilevel)/2.))
@. @subset!(df, !ismissing(:z) && !ismissing(:lower))
@. @subset!(df, iszero(:mistake) && abs(:z) < 10.)
@. @subset!(df, :source!="Abstract")
df.z = Float64.(df.z)
histogram(df.z, normalize=:pdf, label="Actual published effects", legend=:topleft)

# fit 2-component min/max model
p = Float64[.5,.5]
μ = [0.,0.]
τ = [1.,2.]
pD = .4
pF = .5
pH = 1 - pD - pF
u = .75
m = 3.

o = HnFstuff(df.z, D=length(μ), interpres=300, quadnodes=25)
@time res2 = optimize(negHnFll(o), vcat(SimplextoRⁿ(p),μ,log.(τ),SimplextoRⁿ([pD,pF,pH])...,logit(u),log(m)), LBFGS(), autodiff=:forward)
θ, ll = Optim.minimizer(res2), Optim.minimum(res2)
p̂, μ̂ , τ̂ , p̂D, p̂F, p̂H, û, m̂ = RⁿtoSimplex(θ[1:o.D-1]), θ[o.D:2*o.D-1], exp.(θ[2*o.D:3*o.D-1]), RⁿtoSimplex(θ[3*o.D:3*o.D+1])..., logistic(θ[3*o.D+2]), exp(θ[3*o.D+3])
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂D=p̂D, p̂F=p̂F, p̂H=p̂H, û=û, m̂=m̂))
println("Mean ω = ", p̂'μ̂)
ses2 = sqrt.(diag(pinv(ForwardDiff.hessian(negHnFll(o), θ₂))))

zplot = -10:.1:10
pplotfit = map(z->HnFl(z, p̂, μ̂ , τ̂ , p̂D, p̂F, û, m̂), zplot)
plt = histogram(df.z, normalize=:pdf, label="Actual published estimates", legend=:topleft)
plot!(zplot, [p̂[1] * pdf.(Normal(μ̂[1],τ̂[1]), zplot)+p̂[2] * pdf.(Normal(μ̂[2],τ̂[2]), zplot) p̂[1] * pdf.(Normal(μ̂[1],√(1+τ̂[1]^2)), zplot)+p̂[2] * pdf.(Normal(μ̂[2],√(1+τ̂[2]^2)), zplot) pplotfit], label=["Model: true effects" "Model: initial estimates" "Model: published estimates"])
png(plt, "fit")
plt

# distribution of z | ω=2
ω=2; zplot=-3:.01:5; plot(zplot, mapreduce(z->fZcondΩ.(z, ω, [1 p̂D], [0 p̂F], [1 û], [1 m̂]), vcat, zplot), label=["not distorted" "distorted"], xlabel="Reported z | true z = 2")
png("z cond ω=2")

# distribution of ω | z=2
z=2.; ωplot=-3:.01:5; plot(ωplot, mapreduce(ω->[fΩcondZ(ω,z,p̂,μ̂ ,τ̂ ,1.,0.,1.,1.) fΩcondZ(ω,z,p̂,μ̂ ,τ̂ ,p̂D,p̂F,û,m̂)], vcat, ωplot), label=["not distorted" "distorted"], xlabel="True z | reported z = 2")
png("ω cond z=2")

# frequentist CI's as fn of z
ωplot=-5:.01:5; plot(ωplot, ThreadsX.mapreduce(ω->[Cquant.([.025 .5 .975], ω, 1, 0, 1, 1)..., Cquant.([.025 .5 .975], ω, p̂D, p̂F, û, m̂)...]',vcat,ωplot), linecolor=[:blue :blue :blue :orange :orange :orange], linestyle=[:solid :dash :solid :solid :dash :solid], legend=false, xlabel="Reported z", ylabel="95% CI & median")
png("CI cond z")

# Bayesian posterior mean of ω as fn of Z
zplot=-5.:.1:5
pplot = mapreduce(z->[z EΩcondZ(z,p̂,μ̂ ,τ̂ ,1.,0.,1.,1.) EΩcondZ(z,p̂,μ̂ ,τ̂ ,p̂D,p̂F,û, m̂)], vcat, zplot)
plot(zplot, pplot, label=["As is" "shrinkage from informative prior" "shrinkage + adjustment for distortion"], xlabel="Reported z", ylabel="Expected true z")
png("E[ω] cond z")

plot(zplot[zplot.>.2], pplot[zplot.>.2,2:3]./zplot[zplot.>.2], label=["shrinkage from informative prior" "shrinkage + adjustment for distortion"], xlabel="Reported z", ylabel="Discount multiplier")
png("E[ω] discount")

println("Number of missing studies = ", size(df,1) * p̂F * p̂'*(@. cdf(Normal(μ,√(1+τ^2)),Z̄)-cdf(Normal(μ,√(1+τ^2)),-Z̄)))
println("Number of p-hacked studies = ", size(df,1) * (1-p̂D-p̂F) * p̂'*(@. cdf(Normal(μ,√(1+τ^2)),Z̄)-cdf(Normal(μ,√(1+τ^2)),-Z̄)))

# CDF of z conditional on fitted parameters
p = vcat([[z  p̂'cdf.(Normal.(μ̂ , .√(τ̂ .^2 .+ 1)),z) HnFCDF(o,z,p̂,μ̂ ,τ̂ ,p̂D,p̂F,û, m̂)] for z ∈ LinRange(-10:.1:10)]...)
plot(p[:,1], p[:,2:3], label=["Unfiltered" "Filtered"])


# van Zwet, Schwab, and Senn 2021 data, osf.io/xq4b2
df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Clients & prospects\GiveWell\Noisy data\CochraneEffects.csv"))
@. @subset!(df, abs(:z) < 20 && :"outcome.nr"!=1 && :RCT=="yes" && :"outcome.group"=="efficacy")
# combine(groupby(df, :"study.id.sha1"), :z => sample => :z)  # randomly choose among primary outcomes
df.z .*= rand([-1. 1.], nrow(df))
histogram(df.z, normalize=:pdf, label="Actual published effects", legend=:topleft)

p = Float64[.3,.3,.3,.1]
μ = [0.,0.,0.,0.]
τ = [.61, 1.42, 2.16, 5.64]  # vZSS Table 1
pD = .99
pF = .01
pH = 1 - pD - pF
u = .75
m = 3.

o = HnFstuff(df.z, D=length(μ), interpres=300, quadnodes=25)
@time res = optimize(negHnFll(o), vcat(SimplextoRⁿ(p),μ,log.(τ),SimplextoRⁿ([pD,pF,pH])...,logit(u),log(m)), LBFGS(), autodiff=:forward)
θ, ll = Optim.minimizer(res), Optim.minimum(res)
p̂, μ̂ , τ̂ , p̂D, p̂F, p̂H, û, m̂ = RⁿtoSimplex(θ[1:o.D-1]), θ[o.D:2*o.D-1], exp.(θ[2*o.D:3*o.D-1]), RⁿtoSimplex(θ[3*o.D:3*o.D+1])..., logistic(θ[3*o.D+2]), exp(θ[3*o.D+3])
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂D=p̂D, p̂F=p̂F, p̂H=p̂H, û=û, m̂=m̂))
println("Mean ω = ", p̂'μ̂)

ses = sqrt.(diag(pinv(ForwardDiff.hessian(negHnFll(o), θ))))

# Star Wars
df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Library\Meta-science\Brodeur et al. 2016\Data\Final\final_stars_supp.csv"))
@. @subset!(df, lowercase(:main)=="yes")
df = DataFrame(Dict(:z => df.coefficient_num ./ df.standard_deviation_num))
dropmissing!(df)
@subset!(df, abs.(:z).<20)
df.z = Float64.(df.z)
histogram(df.z, normalize=:pdf, bins=100)

p = [.3,.3,.4]
μ = 0.
τ = [1.,2.,3.]
pF₀ = .25
pL₀ = .2
pU₀ = .2
kL = 1.
kU = 1.
m = 3.

o = HnFstuff(df.z, D=length(τ), interpres=300, quadnodes=25)
@time res = optimize(negHnFllSharedμ(o), vcat(SimplextoRⁿ(p),μ,log.(τ),logit(pF₀),logit(pL₀),logit(pU₀),log(kL),log(kU),log(m)), LBFGS(), autodiff=:forward)
θ₂ = Optim.minimizer(res)
p̂, μ̂ , τ̂ , p̂F₀, p̂L₀, p̂U₀, k̂L, k̂U, m̂ = RⁿtoSimplex(θ₂[1:o.D-1]), θ₂[o.D], exp.(θ₂[o.D+1:2*o.D]), logistic(θ₂[2*o.D+1]), logistic(θ₂[2*o.D+2]), logistic(θ₂[2*o.D+3]), exp(θ₂[2*o.D+4]), exp(θ₂[2*o.D+5]), exp(θ₂[2*o.D+6])
p̂D₀ = 1 - p̂F₀
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂F₀=p̂F₀, p̂L₀=p̂L₀, p̂U₀=p̂U₀, k̂L=k̂L, k̂U=k̂U, m̂=m̂))
ses = sqrt.(diag(pinv(ForwardDiff.hessian(negHnFllSharedμ(o), θ))))
zplot = -20:.1:20
pplot = map(z->HnFl(z; p=p̂, μ=fill(μ̂ ,o.D), τ=τ̂ , pF₀=p̂F₀, pL₀=p̂L₀, pU₀=p̂U₀, kL=k̂L, kU=k̂U, m=m̂), zplot)
plot!(zplot, pplot)
