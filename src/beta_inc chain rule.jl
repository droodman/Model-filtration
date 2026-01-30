# copied from Oskar Laverny's pending PR to add derivatives of the incomplete beta function to the ChainRules extension in SpecialFunctions.jl
# The latest commit in this PR is from Jan 29, 2026, so this may be added to SpecialFunctions.jl soon
# https://github.com/JuliaMath/SpecialFunctions.jl/compare/master...lrnv:SpecialFunctions.jl:chainrules-for-beta_inc-and-beta_inc_inv 
#
# only changes are the lines just below, and increasing the default error tolerance in _beta_inc_grad()

using ForwardDiff, ForwardDiffChainRules                                                  
import SpecialFunctions.beta_inc                                                          
@ForwardDiff_frule beta_inc(a::ForwardDiff.Dual, b::ForwardDiff.Dual, x::ForwardDiff.Dual)


# Incomplete beta derivatives via Boik & Robinson-Cox
#
# Reference
#   R. J. Boik and J. F. Robinson-Cox (1999).
#   "Derivatives of the incomplete beta function."
#   Journal of Statistical Software, 3(1).
#   URL: https://www.jstatsoft.org/article/view/v003i01
#
# The following implementation computes the regularized incomplete beta
# I_x(a,b) together with its partial derivatives with respect to a, b, and x
# using a continued-fraction representation of ₂F₁ and differentiating through it.
# This is an independent implementation adapted from https://github.com/arzwa/IncBetaDer.jl.

# Generic-typed helpers used by the continued-fraction evaluation of I_x(a,b)
# and its partial derivatives. These implement the scalar prefactor K(x;p,q),
# the auxiliary variable f, the continued-fraction coefficients a_n, b_n, and
# their partial derivatives w.r.t. p (≡ a) and q (≡ b). See Boik & Robinson-Cox (1999).


using SpecialFunctions, ChainRulesCore
@inline function _Kfun(logx::T, log1mx::T, p::T, q::T, logbetapq::T) where {T}
    # K(x;p,q) = x^p (1-x)^{q-1} / (p * B(p,q)) computed in log-space for stability
    # logx = log(x), log1mx = log(1-x), precomputed
    return exp(p * logx + (q - 1) * log1mx - log(p) - logbetapq)
end

@inline function _ffun(x::T, p::T, q::T) where {T}
    # f = q x / (p (1-x)) — convenience variable appearing in CF coefficients
    return q * x / (p * (1 - x))
end

@inline function _a1fun(p::T, q::T, f::T) where {T} 
    # a₁ coefficient of the continued fraction for ₂F₁ representation
    return p * f * (q - 1) / (q * (p + 1))
end

@inline function _anfun(p::T, q::T, f::T, n::Int, pfq2::T) where {T}
    # a_n coefficient (n ≥ 1) of the continued fraction for ₂F₁ in terms of p=a, q=b, f.
    # For n=1, falls back to a₁; for n≥2 uses the closed-form product from the Gauss CF.
    # pfq2 = (p * f / q)^2, precomputed for efficiency
    n == 1 && return _a1fun(p, q, f)
    pn = p + n
    p2n = pn + n
    return pfq2 * (n - 1) * (pn + q - 2) * (pn - 1) * (q - n) / ((p2n - 3) * (p2n - 2)^2 * (p2n - 1))
end

@inline function _bnfun(p::T, q::T, n::Int, pf_2q::T, pq_p2pf::T) where {T}
    # b_n coefficient (n ≥ 1) of the continued fraction. Derived for the same CF.
    # pf_2q = p * f + 2 * q, pq_p2pf = p * q * (p - 2 - p * f)
    x = 2 * n * pf_2q * (n + p - 1) + pq_p2pf
    y = q * (p + 2*n - 2) * (p + 2*n)
    return x / y
end

@inline function _dK_dp(logx::T, p::T, K::T, ψpq::T, ψp::T) where {T}
    # ∂K/∂p using digamma identities: d/dp log B(p,q) = ψ(p) - ψ(p+q)
    # logx = log(x), precomputed
    return K * (logx - inv(p) + ψpq - ψp)
end

@inline function _dK_dq(log1mx::T, K::T, ψpq::T, ψq::T) where {T}
    # ∂K/∂q using identical pattern
    # log1mx = log(1-x), precomputed
    K * (log1mx + ψpq - ψq)
end

@inline function _dK_dpdq(logx::T, log1mx::T, p::T, q::T, logbetapq::T) where {T}
    # Convenience: compute (∂K/∂p, ∂K/∂q) together with shared ψ(p+q)
    # logx = log(x), log1mx = log(1-x), precomputed
    ψ = digamma(p + q)
    Kf = _Kfun(logx, log1mx, p, q, logbetapq)
    dKdp = _dK_dp(logx, p, Kf, ψ, digamma(p))
    dKdq = _dK_dq(log1mx, Kf, ψ, digamma(q))
    return dKdp, dKdq
end

@inline function _da1_dp(p::T, q::T, f::T) where {T}
    # ∂a₁/∂p from the closed form of a₁
    return - _a1fun(p, q, f) / (p + 1)
end

@inline function _dan_dp(p::T, q::T, n::Int, an::T, da1_dp::T) where {T}
    # ∂a_n/∂p via log-derivative: d a_n = a_n * d log a_n; for n=1, uses precomputed ∂a₁/∂p
    # an is passed in from _nextapp to avoid redundant computation
    n == 1 && return da1_dp
    dlog = inv(p + q + n - 2) + inv(p + n - 1) - inv(p + 2*n - 3) - 2 * inv(p + 2*n - 2) - inv(p + 2*n - 1)
    return an * dlog
end

@inline function _da1_dq(p::T, q::T, f::T) where {T}
    # ∂a₁/∂q
    return _a1fun(p, q, f) / (q - 1)
end


@inline function _dan_dq(p::T, q::T, n::Int, pfq2::T, p2q2::T, da1_dq::T) where {T}
    # ∂a_n/∂q avoiding the removable singularity at q ≈ n for integer q.
    # For n=1, returns precomputed ∂a₁/∂q.
    # pfq2 = (p * f / q)^2, p2q2 = p + 2*q - 2, da1_dq = a1 / (q - 1)
    if n == 1
        return da1_dq
    end
    # Use the simplified closed-form of a_n that eliminates explicit q^2 via f:
    #   a_n = (x/(1-x))^2 * (n-1) * (p+n-1) * (p+q+n-2) * (q-n) / D(p,n)
    # where D(p,n) = (p+2n-3)*(p+2n-2)^2*(p+2n-1) and (x/(1-x)) = p*f/q.
    # Differentiate only the q-dependent factor G(q) = (p+q+n-2)*(q-n):
    #   dG/dq = (q-n) + (p+q+n-2) = p + 2q - 2.

    C = pfq2 * (n - 1) * (p + n - 1) /
        ((p + 2*n - 3) * (p + 2*n - 2)^2 * (p + 2*n - 1))
    return C * p2q2
end

@inline function _dbn_dp(p::T, q::T, n::Int, pf_2q::T, pq_p2pf::T, pqf::T) where {T}
    # ∂b_n/∂p via quotient rule on b_n = N/D.
    # pf_2q = p * f + 2 * q, pq_p2pf = p * q * (p - 2 - p * f), pqf = p * q * f
    A = 2 * n^2 + 2 * (p - 1) * n
    N1 = pf_2q * A
    N = N1 + pq_p2pf
    D = q * (p + 2*n - 2) * (p + 2*n)
    dN1_dp = 2 * n * pf_2q
    dN2_dp = q * (2 * p - 2) - pqf
    dN_dp = dN1_dp + dN2_dp
    dD_dp = q * (2 * p + 4 * n - 2)
    return (dN_dp * D - N * dD_dp) / (D^2)
end

@inline function _dbn_dq(p::T, q::T, n::Int, pf_2q::T, pq_p2pf::T, p_2_pf::T, p2f::T, pfq_2::T) where {T}
    # ∂b_n/∂q similarly via quotient rule
    # pf_2q = p * f + 2 * q, pq_p2pf = p * q * (p - 2 - p * f), p_2_pf = p - 2 - p * f
    # p2f = p^2 * f, pfq_2 = p * (f / q) + 2
    A = 2 * n^2 + 2 * (p - 1) * n
    N1 = pf_2q * A
    N = N1 + pq_p2pf
    D = q * (p + 2*n - 2) * (p + 2*n)
    dN1_dq = pfq_2 * A
    dN2_dq = p * p_2_pf - p2f
    dN_dq = dN1_dq + dN2_dq
    dD_dq = (p + 2*n - 2) * (p + 2*n)
    return (dN_dq * D - N * dD_dq) / (D^2)
end

@inline function _nextapp(f::T, p::T, q::T, n::Int, App::T, Ap::T, Bpp::T, Bp::T,
                         pfq2::T, pf_2q::T, pq_p2pf::T) where {T}
    # One step of the continuant recurrences:
    #   A_n = a_n A_{n-2} + b_n A_{n-1}
    #   B_n = a_n B_{n-2} + b_n B_{n-1}
    an = _anfun(p, q, f, n, pfq2)
    bn = _bnfun(p, q, n, pf_2q, pq_p2pf)
    An = an * App + bn * Ap
    Bn = an * Bpp + bn * Bp
    return An, Bn, an, bn
end

@inline function _dnextapp(an::T, bn::T, dan::T, dbn::T, Xpp::T, Xp::T, dXpp::T, dXp::T) where {T}
    # Derivative propagation for the same recurrences (X∈{A,B})
    return dan * Xpp + an * dXpp + dbn * Xp + bn * dXp
end

function _beta_inc_grad(a::T, b::T, x::T; maxapp::Int=200, minapp::Int=3, err::T=eps(T)*T(1e4)) where {T}
    # Compute I_x(a,b) and partial derivatives (∂I/∂a, ∂I/∂b, ∂I/∂x)
    # using a differentiated continued fraction with convergence control.
    oneT = one(T)
    zeroT = zero(T)

    # 1) Boundary cases for x
    isone(x) && return oneT, zeroT, zeroT, zeroT
    iszero(x) && return zeroT, zeroT, zeroT, zeroT

    # 2) Get tolerence
    ϵ = err

    logbetapq = logbeta(a,b)  # Time-consuming step; symetric in a,b

    # 3) Precompute log(x) and log(1-x) once at original x
    logx   = log(x)
    log1mx = log1p(-x)

    # 3a) Non-boundary path: precompute ∂I/∂x at original (a,b,x) via stable log form
    dx = exp((a - oneT) * logx + (b - oneT) * log1mx - logbetapq)

    # 4) Optional tail-swap for symmetry and improved CF convergence:
    #    if x > a/(a+b), evaluate at (p,q,x₀) = (b,a,1-x) and swap back at the end.
    p    = a
    q    = b
    swap = x > a / (a + b)
    if swap
        x₀      = oneT - x
        p       = b
        q       = a
        logx₀   = log1mx    # log(1-x) = log(x₀)
        log1mx₀ = logx      # log(1-(1-x)) = log(x)
    else
        x₀      = x
        logx₀   = logx
        log1mx₀ = log1mx
    end

    # 5) Initialize CF state and derivatives
    K                    = _Kfun(logx₀, log1mx₀, p, q, logbetapq)
    dK_dp_val, dK_dq_val = _dK_dpdq(logx₀, log1mx₀, p, q, logbetapq)
    f                    = _ffun(x₀, p, q)

    # 5a) Precompute loop-invariant expressions (only depend on p, q, f)
    pf      = p * f
    pfq     = pf / q                    # p * f / q
    pfq2    = pfq * pfq                 # (p * f / q)^2
    pf_2q   = pf + 2 * q                # p * f + 2 * q
    p_2_pf  = p - 2 - pf                # p - 2 - p * f
    pq_p2pf = p * q * p_2_pf            # p * q * (p - 2 - p * f)
    pqf     = p * q * f                 # for _dbn_dp
    p2f     = p * p * f                 # p^2 * f, for _dbn_dq
    pfq_2   = pfq + 2                   # p * (f / q) + 2
    p2q2    = p + 2*q - 2               # p + 2*q - 2
    a1      = _a1fun(p, q, f)           # a₁ coefficient
    da1_dp  = -a1 / (p + 1)             # ∂a₁/∂p
    da1_dq  = a1 / (q - 1)              # ∂a₁/∂q

    App                  = oneT
    Ap                   = oneT
    Bpp                  = zeroT
    Bp                   = oneT
    dApp_dp              = zeroT
    dBpp_dp              = zeroT
    dAp_dp               = zeroT
    dBp_dp               = zeroT
    dApp_dq              = zeroT
    dBpp_dq              = zeroT
    dAp_dq               = zeroT
    dBp_dq               = zeroT
    dI_dp                = T(NaN)
    dI_dq                = T(NaN)
    Ixpq                 = T(NaN)
    Ixpqn                = T(NaN)
    dI_dp_prev           = T(NaN)
    dI_dq_prev           = T(NaN)

    # 6) Main CF loop (n from 1): update continuants, scale, form current approximant Cn=A_n/B_n
    #    and its derivatives to update I and ∂I/∂(p,q). Stop on relative convergence of all.
    for n=1:maxapp

        # Update continuants.
        An, Bn, an, bn = _nextapp(f, p, q, n, App, Ap, Bpp, Bp, pfq2, pf_2q, pq_p2pf)
        dan            = _dan_dp(p, q, n, an, da1_dp)
        dbn            = _dbn_dp(p, q, n, pf_2q, pq_p2pf, pqf)
        dAn_dp         = _dnextapp(an, bn, dan, dbn, App, Ap, dApp_dp, dAp_dp)
        dBn_dp         = _dnextapp(an, bn, dan, dbn, Bpp, Bp, dBpp_dp, dBp_dp)
        dan            = _dan_dq(p, q, n, pfq2, p2q2, da1_dq)
        dbn            = _dbn_dq(p, q, n, pf_2q, pq_p2pf, p_2_pf, p2f, pfq_2)
        dAn_dq         = _dnextapp(an, bn, dan, dbn, App, Ap, dApp_dq, dAp_dq)
        dBn_dq         = _dnextapp(an, bn, dan, dbn, Bpp, Bp, dBpp_dq, dBp_dq)

        # Normalize states to control growth/underflow (scale-invariant transform)
        s = maximum((abs(An), abs(Bn), abs(Ap), abs(Bp), abs(App), abs(Bpp)))
        if isfinite(s) && s > zeroT
            invs     = inv(s)
            An      *= invs
            Bn      *= invs
            Ap      *= invs
            Bp      *= invs
            App     *= invs
            Bpp     *= invs
            dAn_dp  *= invs
            dBn_dp  *= invs
            dAn_dq  *= invs
            dBn_dq  *= invs
            dAp_dp  *= invs
            dBp_dp  *= invs
            dApp_dp *= invs
            dBpp_dp *= invs
            dAp_dq  *= invs
            dBp_dq  *= invs
            dApp_dq *= invs
            dBpp_dq *= invs
        end

        # Form current approximant Cn=A_n/B_n and its derivatives.
        # Guard against tiny/zero Bn to avoid NaNs/Inf in divisions.
        tiny   = sqrt(eps(T))

        absBn  = abs(Bn)
        sgnBn  = ifelse(Bn >= zeroT, oneT, -oneT)
        invBn  = absBn > tiny && isfinite(absBn) ? inv(Bn) : inv(sgnBn * tiny)
        Cn     = An * invBn
        invBn2 = invBn * invBn
        dI_dp  = dK_dp_val * Cn + K * (invBn * dAn_dp - (An * invBn2) * dBn_dp)
        dI_dq  = dK_dq_val * Cn + K * (invBn * dAn_dq - (An * invBn2) * dBn_dq)
        Ixpqn  = K * Cn

        # Decide convergence: 
        if n >= minapp
            # Relative convergence for I, ∂I/∂p, ∂I/∂q (guards against tiny denominators)
            denomI = max(abs(Ixpqn), abs(Ixpq), eps(T))
            denomp = max(abs(dI_dp), abs(dI_dp_prev), eps(T))
            denomq = max(abs(dI_dq), abs(dI_dq_prev), eps(T))
            rI     = abs(Ixpqn - Ixpq) / denomI
            rp     = abs(dI_dp - dI_dp_prev) / denomp
            rq     = abs(dI_dq - dI_dq_prev) / denomq
            if max(rI, rp, rq) < ϵ
                break
            end
        end
        Ixpq       = Ixpqn
        dI_dp_prev = dI_dp
        dI_dq_prev = dI_dq

        # Shift CF state for next iteration
        App        = Ap
        Bpp        = Bp
        Ap         = An
        Bp         = Bn
        dApp_dp    = dAp_dp
        dApp_dq    = dAp_dq
        dBpp_dp    = dBp_dp
        dBpp_dq    = dBp_dq
        dAp_dp     = dAn_dp
        dAp_dq     = dAn_dq
        dBp_dp     = dBn_dp
        dBp_dq     = dBn_dq
    end

    # 7) Undo tail-swap if applied; ∂I/∂x is the pdf at original (a,b,x)
    if swap
        return oneT - Ixpqn, -dI_dq, -dI_dp, dx
    else
        return Ixpqn, dI_dp, dI_dq, dx
    end
end





# Incomplete beta: beta_inc(a,b,x) -> (p, q) with q=1-p
function ChainRulesCore.frule((_, Δa, Δb, Δx), ::typeof(beta_inc), a::Number, b::Number, x::Number)
    # primal
    p, q = beta_inc(a, b, x)
    # derivatives
    _a, _b, _x = map(float, promote(a, b, x))
    _, dIa, dIb, dIx = _beta_inc_grad(_a, _b, _x)
    Δp = muladd(dIx, Δx, muladd(dIb, Δb, dIa * Δa))
    Δq = -Δp
    Tout = typeof((p, q))
    return (p, q), ChainRulesCore.Tangent{Tout}(Δp, Δq)
end

function ChainRulesCore.rrule(::typeof(beta_inc), a::Number, b::Number, x::Number)
    p, q = beta_inc(a, b, x)
    Ta = ChainRulesCore.ProjectTo(a)
    Tb = ChainRulesCore.ProjectTo(b)
    Tx = ChainRulesCore.ProjectTo(x)
    _a, _b, _x = map(float, promote(a, b, x))
    _, dIa, dIb, dIx = _beta_inc_grad(_a, _b, _x)
    function beta_inc_pullback(Δ)
        Δp, Δq = Δ
        s = Δp - Δq # because q = 1 - p
        ā = Ta(s * dIa)
        b̄ = Tb(s * dIb)
        x̄ = Tx(s * dIx)
        return ChainRulesCore.NoTangent(), ā, b̄, x̄
    end
    return (p, q), beta_inc_pullback
end
function ChainRulesCore.frule((_, Δa, Δb, Δx, Δy), ::typeof(beta_inc), a::Number, b::Number, x::Number, y::Number)
    p, q = beta_inc(a, b, x, y)
    _a, _b, _x, _y = map(float, promote(a, b, x, y))
    _, dIa, dIb, dIx = _beta_inc_grad(_a, _b, _x)
    Δp = muladd(dIx, Δx, muladd(-dIx, Δy, muladd(dIb, Δb, dIa * Δa)))
    Δq = -Δp
    Tout = typeof((p, q))
    return (p, q), ChainRulesCore.Tangent{Tout}(Δp, Δq)
end

function ChainRulesCore.rrule(::typeof(beta_inc), a::Number, b::Number, x::Number, y::Number)
    p, q = beta_inc(a, b, x, y)
    Ta = ChainRulesCore.ProjectTo(a)
    Tb = ChainRulesCore.ProjectTo(b)
    Tx = ChainRulesCore.ProjectTo(x)
    Ty = ChainRulesCore.ProjectTo(y)
    _a, _b, _x, _y = map(float, promote(a, b, x, y))
    _, dIa, dIb, dIx = _beta_inc_grad(_a, _b, _x)
    function beta_inc_pullback(Δ)
        Δp, Δq = Δ
        s = Δp - Δq
        ā = Ta(s * dIa)
        b̄ = Tb(s * dIb)
        x̄ = Tx(s * dIx)
        ȳ = Ty(-s * dIx)
        return ChainRulesCore.NoTangent(), ā, b̄, x̄, ȳ
    end
    return (p, q), beta_inc_pullback
end

# Inverse incomplete beta: beta_inc_inv(a,b,p) -> (x, 1-x)
function ChainRulesCore.frule((_, Δa, Δb, Δp), ::typeof(beta_inc_inv), a::Number, b::Number, p::Number)
    x, y = beta_inc_inv(a, b, p)
    _a, _b, _x, _p = map(float, promote(a, b, x, p))
    # Implicit differentiation at solved x: I_x(a,b) = p
    _, dIa, dIb, _ = _beta_inc_grad(_a, _b, _x)
    # ∂I/∂x at solved x via stable log-space expression
    dIx_acc = exp(muladd(_a - 1, log(_x), muladd(_b - 1, log1p(-_x), -logbeta(_a, _b))))
    inv_dIx = inv(dIx_acc)
    dx_da = -dIa * inv_dIx
    dx_db = -dIb * inv_dIx
    dx_dp = inv_dIx
    Δx = muladd(dx_dp, Δp, muladd(dx_db, Δb, dx_da * Δa))
    Δy = -Δx
    Tout = typeof((x, y))
    return (x, y), ChainRulesCore.Tangent{Tout}(Δx, Δy)
end

function ChainRulesCore.rrule(::typeof(beta_inc_inv), a::Number, b::Number, p::Number)
    x, y = beta_inc_inv(a, b, p)
    Ta = ChainRulesCore.ProjectTo(a)
    Tb = ChainRulesCore.ProjectTo(b)
    Tp = ChainRulesCore.ProjectTo(p)
    _a, _b, _x, _p = map(float, promote(a, b, x, p))
    _, dIa, dIb, _ = _beta_inc_grad(_a, _b, _x)
    # ∂I/∂x at solved x via stable log-space expression
    dIx_acc = exp(muladd(_a - 1, log(_x), muladd(_b - 1, log1p(-_x), -logbeta(_a, _b))))
    inv_dIx = inv(dIx_acc)
    dx_da = -dIa * inv_dIx
    dx_db = -dIb * inv_dIx
    dx_dp = inv_dIx
    function beta_inc_inv_pullback(Δ)
        Δx, Δy = Δ
        s = Δx - Δy
        ā = Ta(s * dx_da)
        b̄ = Tb(s * dx_db)
        p̄ = Tp(s * dx_dp)
        return ChainRulesCore.NoTangent(), ā, b̄, p̄
    end
    return (x, y), beta_inc_inv_pullback
end


aa=3.7; bb=1.2; xx=.3; @btime _beta_inc_grad($aa,$bb,$xx)

df = DataFrame(XLSX.readtable("data/Schuemie et al. 2013/appendix g revision.xlsx", "NeatTable", first_row=2, infer_eltypes=true)...)
@. df.z = log(df."Effect estimate") / (log(df."Upper bound of 95% CI" / df."Lower bound of 95% CI") / 2z̄)
@. @subset!(df, abs(:z)<20)
disallowmissing!(df, :z)

__penalty(; τ::Vector{T}, σ::Vector{T}, σₘ::Vector{T}, file_drawer_insig::T, kwargs...) where {T} = 
    logpdf(Normal(0,50), log(σ[])) + 
    logpdf(Normal(0,50), log(σₘ[])) + 
    sum(logpdf(Normal(0,5), log(τᵢ)) for τᵢ ∈ τ)
__d=1
__from  = (p=fill(1/__d,__d), μ=[0.]     , τ=collect(LinRange(1,__d,__d)), pDFR=fill(1/3,3), σ=[1.]      , ν=[5.]      , μₘ=[0.]    , σₘ=[10.]     )
__xform = (p=SimplextoRⁿ, μ=identity , τ=bcast(log)              , pDFR=SimplextoRⁿ, σ=bcast(log), ν=bcast(log), μₘ=identity, σₘ=bcast(log))
__M = HnFmodel(df.z; d=__d, penalty=__penalty)
___from = pairs(__from)
__fromxform = [__xform[p](v) for (p,v) ∈ ___from]  # starting values in optimization parameter space
__extractor = zip(keys(___from), Iterators.accumulate((ind,f)->f isa Number ? (last(ind)+1) : last(ind)+1:last(ind)+length(f), __fromxform, init=0))
__xformer(x) = (p=>inverse(__xform[p])(x[e]) for (p,e) ∈ __extractor)  # map primary parameters into full model space, expressed as functions of optimization parameters, e.g. exp(log(σ))
__objective(x) = -HnFll(__M; __xformer(x)...)
__θ = vcat(__fromxform...)
[__objective(__θ); ForwardDiff.gradient(__objective, __θ)] - [43366.60740962447, -4080.292638304726, -5061.240886317648, -2333.889966771348, 3059.97435335951, -17562.96774747974, 9314.187099419918, -566.0593935293476, -4654.085929173036]

