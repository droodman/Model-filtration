#
# generalized t distribution: adds μ and σ parameters
#

using Distributions

const 𝒩 = Normal()
const z̄ = quantile(𝒩, .975)  # 1.96

@inline diffcdf(N,b,a) = cdf(N,b) - cdf(N,a)

@inline Tcdf( d,x) = iszero(x) ? .5 :  cdf(d,x)  # the derivative of beta_inc() seems to be causing NaN from cdf(TDist()) when x = 0 
@inline Tccdf(d,x) = iszero(x) ? .5 : ccdf(d,x)
@inline logdiffTcdf(d,b,a) = iszero(b) ? log(.5 - cdf(d,a)) : iszero(a) ? log(cdf(d,b) - .5) : logdiffcdf(d,b,a)

struct GenT{T<:Real} <: ContinuousUnivariateDistribution
	ν::T; μ::T; σ::T

	lnσ::T
	tdist::TDist{T}  # underlying Student's t distribution

	GenT(ν::T, μ::T, σ::T) where {T<:Real} = new{T}(ν, μ, σ, log(σ), TDist{T}(ν))
end
Distributions.pdf(     d::GenT, x::Real) = pdf(     d.tdist, (x - d.μ) / d.σ) / d.σ
Distributions.logpdf(  d::GenT, x::Real) = logpdf(  d.tdist, (x - d.μ) / d.σ) - d.lnσ
Distributions.cdf(     d::GenT, x::Real) = cdf(     d.tdist, (x - d.μ) / d.σ)
Distributions.logcdf(  d::GenT, x::Real) = logcdf(  d.tdist, (x - d.μ) / d.σ)
Distributions.quantile(d::GenT, p::Real) = quantile(d.tdist, p) * d.σ + d.μ
Distributions.logdiffcdf(d::GenT, b::Real, a::Real) = logdiffcdf(d.tdist, (b - d.μ) / d.σ, (a - d.μ) / d.σ)
