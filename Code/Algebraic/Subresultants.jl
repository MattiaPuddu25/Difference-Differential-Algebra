const __SUBRESULTANT_GUARD_TERM_LIMIT = 8
const __SUBRESULTANT_GUARD_MAX_VARIABLES = 4
const __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_1 = 8
const __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_2 = 8
const __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_3 = 4
const __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_4 = 2

###############################################################################
#####  Subresultant data and polynomial arithmetic
###############################################################################

# Retain the pseudo-division result and form its multiplier only on request.
mutable struct __SubresultantCofactor{U,P}
    quotient::U
    remainder::U
    exponent::Int
    multiplier::Union{Nothing,P}
end

"""
Store subresultant data for two multivariate polynomials, which are viewed as univariate
polynomials in a chosen variable.\
Regular subresultants are recorded by their index. More precisely, if the two input
polynomials have degrees `m` and `n`, and `r = min(m, n)`:

- for `0 ≤ j < r`, the entry indexed by `j`, when present, is the regular `j`-th subresultant;
the entry indexed by `0`, when computed, is the exact resultant.
- a missing entry in this range has principal subresultant coefficient equal to zero,
provided that index has already been covered by the computation; 
- the entry indexed by `r` is the input polynomial of smaller degree , or the second
input when the two inputs have equal degree.

This dictionary should not be interpreted as the full sequence including defective
subresultant polynomials. The field `computed_minimum` records the lowest degree covered
by the computation.
Missing entries at or above that degree have zero principal coefficient; entries
below it are unknown.

Stored polynomials are treated as read-only.
No global cache keeps these data after their caller releases them.
"""
struct __AlgebraicSubresultantData{R,P,U}
    ring::R
    variable_index::Int
    inputs::Tuple{U,U}
    degrees::Tuple{Int,Int}
    regular::Dict{Int,U}
    cofactors::Dict{Tuple{Int,Int},__SubresultantCofactor{U,P}}
    computed_minimum::Int
end

"""
Determine the regular subresultants of `f` and `g` with respect to the variable of index `idx`.\
The two inputs have the same multivariate polynomial ring over a field as parent, and
positive degree in the chosen variable.\
Zero inputs, constants in that variable, and specializations at which an input degree drops
are handled separately by `__algebraic_subresultant_gcd_cases`.

View `f` and `g` as univariate polynomials in the chosen variable.
Internally, reorder these univariate polynomials as `P, Q` preserving the original order if the degrees are equal
so that `degree(P) >= degree(Q)`.
The entry indexed by `deg(Q)` is `Q` itself, according to the convention used by `__AlgebraicSubresultantData`.

The lower regular subresultants are computed block by block.
A degree drop produces a defective top polynomial; Lazard normalization gives the regular base
of the same block, while the Ducos recurrence produces the top polynomial of the next block.
Zero and defective entries are not stored.

No parameter-dependent content is removed: such factors must be preserved
because their vanishing can change the specialized subresultant chain.

If the inputs were reordered, restore the signs corresponding to the original order of `f` and `g`.

With `minimum_degree > 0`, stop after computing the first regular subresultant whose index is at most `minimum_degree`.\
The field `computed_minimum` records the lowest index actually covered by the computation;
lower indices must not be interpreted as zero unless the recurrence has terminated.\
The default `minimum_degree = 0` computes the complete subresultant data.
```
"""
function __algebraic_subresultant_data(f::T, g::T, idx::Int; 
        minimum_degree::Int=0) where {T<:MPolyRingElem}
    R = parent(f)
    df, dg = degree(f, idx), degree(g, idx)

    # Regard f and g as univariate polynomials in the chosen variable.
    # Their coefficients remain in the original multivariate ring.
    S, _ = polynomial_ring(R, :__subresultant_variable; cached=false)
    F = S(AbstractAlgebra.coefficients(f, idx))
    G = S(AbstractAlgebra.coefficients(g, idx))
    U = typeof(F)
    regular = Dict{Int,U}()
    sizehint!(regular, min(df, dg) + 1)
    # Order the inputs as P, Q with degree(P) >= degree(Q); for equal degrees the original order is preserved.
    P, Q = df < dg ? (G, F) : (F, G)
    n = degree(Q)
    # Internal convention: store Q itself at index n instead of the n-th
    # subresultant, which differs from Q by a leading-coefficient factor.
    regular[n] = Q
    s = leading_coefficient(Q)^(degree(P) - n)
    previous_regular = zero(S)
    # A is the current regular base. B is the top polynomial of the next block,
    # initially obtained from the pseudo-remainder of P and Q.
    A = Q
    B = AbstractAlgebra.pseudorem(P, -Q)
    computed_minimum = 0
    while !iszero(B)
        e = degree(B)
        gap = degree(A) - e
        # Normalize the top B to the regular base C of the same block.
        C = if gap == 1
                # No degree drop: the block is regular, so top and base coincide.
                B
            elseif iszero(previous_regular)
                # First degree drop. Here s is the principal coefficient corresponding to the n-th subresultant
                divexact(leading_coefficient(B)^(gap - 1) * B, s^(gap - 1))
            else
                # Later degree drops use Lazard normalization. In the terminal
                # monomial case degree(B) = 0, use the specialized fast path.
                terminal = e == 0 ? __subresultant_terminal_monomial_normalization(B, s, gap) :
                                    nothing
                terminal === nothing ? AbstractAlgebra.subresultant_lazard(previous_regular, B) :
                                    terminal
            end
        # C is the regular subresultant at index e.
        regular[e] = C
        # A missing lower entry is unknown when the computation is truncated.
        if e <= minimum_degree
            computed_minimum = e
            break
        end
        # From the regular base C of the current block, use the
        # Ducos recurrence to construct the top polynomial of the next block.
        B = AbstractAlgebra.subresultant_ducos(A, B, C, s)
        previous_regular = C
        A = C
        s = leading_coefficient(A)
    end
    # This computation may swap the inputs to ensure degree(P) >= degree(Q):
    # restore the signs of the subresultants for the original order f, g.
    if df < dg
        for j in keys(regular)
            j < n && isodd((df - j) * (dg - j)) && (regular[j] = -regular[j])
        end
    end
    return __AlgebraicSubresultantData{typeof(R),T,U}(
        R, idx, (F, G), (df, dg), regular, Dict{Tuple{Int,Int},__SubresultantCofactor{U,T}}(), computed_minimum)
end

"""
Compute the terminal Lazard normalization directly when the relevant leading coefficients
and normalization factors are monomials in the coefficient polynomial ring, in order
to avoid generic multivariate exact division.
"""
function __subresultant_terminal_monomial_normalization(Q, s, gap::Int)
    degree(Q) == 0 || return nothing
    gap > 1 || return nothing
    length(s) == 1 || return nothing
    q = leading_coefficient(Q)
    length(q) == 1 || return nothing
    R = parent(q)
    q_exp = exponent_vector(q, 1)
    s_exp = exponent_vector(s, 1)
    quotient_exp = Int[gap * Int(q_exp[i]) - (gap - 1) * Int(s_exp[i]) for i in eachindex(q_exp)]
    # Exactness of the subresultant normalization implies nonnegative exponents in the monomial case.
    q_coeff = leading_coefficient(q)
    s_coeff = leading_coefficient(s)
    quotient_coeff = q_coeff^gap / s_coeff^(gap - 1)
    L = R(quotient_coeff)
    for i in eachindex(quotient_exp)
        e = quotient_exp[i]
        iszero(e) || (L *= gen(R, i)^e)
    end
    return parent(Q)(L)
end

"""
Return the regular subresultant indexed by `j`, or zero if no regular subresultant is stored at that index.\
For `j = min(data.degrees...)`, return the input polynomial stored at that index according to
the convention of `__AlgebraicSubresultantData`.\
If the subresultant indexed by `j` is defective, return zero even when the corresponding defective
subresultant is a nonzero polynomial of degree smaller than `j`.

Require `j >= computed_minimum`; callers must not query uncomputed degrees as if they were zero.
"""
function __algebraic_subresultant(data::__AlgebraicSubresultantData, j::Int)
    haskey(data.regular, j) || return zero(data.ring)
    return __subresultant_lift(data, data.regular[j])
end

"""
Reconstruct a polynomial in the original ring, without exposing a stored coefficient.
"""
function __subresultant_lift(data::__AlgebraicSubresultantData, p::PolyRingElem)
    return deepcopy(evaluate(p, gen(data.ring, data.variable_index)))
end

"""
Return the principal subresultant coefficient indexed by `j`.\
If the regular subresultant indexed by `j` is stored, return its coefficient of degree `j` in
the chosen variable. If no regular subresultant is stored at that index, return zero.
Require `j >= computed_minimum`, as guaranteed by the internal callers.

Provided that the degrees of both inputs are preserved under specialization, the smallest index
with nonzero principal subresultant coefficient is the degree of the specialized GCD.
On a branch of the Thomas decomposition, nonvanishing of a parameter-dependent coefficient must be
guaranteed by adding a new inequation.
"""
function __algebraic_principal_subresultant(data::__AlgebraicSubresultantData, j::Int)
    haskey(data.regular, j) || return zero(data.ring)
    return deepcopy(coeff(data.regular[j], j))
end

"""
Return the resultant in the original input order, including its sign and coefficient factors.\
Require complete data covering degree zero.
If the inputs have a nonconstant GCD over the fraction field of the parameter ring, return zero.
"""
__algebraic_subresultant_resultant(data::__AlgebraicSubresultantData) = __algebraic_subresultant(data, 0)

"""
For the input selected by `polynomial`, compute a pseudo-division certificate
with respect to the regular subresultant indexed by `j`.\
Return `(divisor, quotient, remainder, multiplier)` satisfying

`multiplier * input = quotient * divisor + remainder`.

If `j = 0`, use one as divisor.

The pseudo-division is computed and cached when requested; with `quotient_only=true`,
return only an independent copy of the quotient, without constructing the multiplier.
Cache that multiplier only if a complete certificate is requested.

In the Thomas decomposition, on a branch where `j` is the GCD degree and both input degrees
are preserved, the remainder specializes to zero and the multiplier is nonzero.
Hence the quotient represents `input/GCD` up to a nonzero factor.

The remainder need not be the zero polynomial in the original parameter ring:
it may vanish only modulo the equations defining the branch.
"""
function __algebraic_subresultant_cofactor(data::__AlgebraicSubresultantData, j::Int; polynomial::Int=1, quotient_only::Bool=false)
    iszero(j) && quotient_only && return __subresultant_lift(data, data.inputs[polynomial])
    divisor = j == 0 ? one(parent(data.inputs[1])) :
                    data.regular[j]
    certificate = get!(data.cofactors, (j, polynomial)) do
        q, r, exponent = __subresultant_pseudo_division(data.inputs[polynomial], divisor)
        __SubresultantCofactor{typeof(q),elem_type(data.ring)}(q, r, exponent, nothing)
    end
    quotient_only && return __subresultant_lift(data, certificate.quotient)
    if certificate.multiplier === nothing
        certificate.multiplier = leading_coefficient(divisor)^certificate.exponent
    end
    return (divisor=__subresultant_lift(data, divisor),
            quotient=__subresultant_lift(data, certificate.quotient),
            remainder=__subresultant_lift(data, certificate.remainder),
            multiplier=deepcopy(certificate.multiplier))
end

###############################################################################
#####  Parameter splitting and GCD cases
###############################################################################

"""
When the coefficients of the polynomials input only depend on lower variables,
some specialization may drop both their degrees and their gcd.
We then split the computation of the subresultant in branches that are determined
by equations and inequations on such coefficients. This struct records the data
corresponding to one of these branches.
The fields `equations` and `inequations` record the conditions defining the branch,
while `left` and `right` represent the two input polynomials, taking into account
the possible degree drop in `variable_index`.
The field `gcd` records a representative of the greatest common divisor of `left`
and `right` on the branch, whose degree in the variable indexed by `variable_index`
is stored in `degree`.
"""
struct __SubresultantGCDCase{P<:MPolyRingElem}
    equations::Vector{P}
    inequations::Vector{P}
    left::P
    right::P
    gcd::P
    degree::Int
    variable_index::Int
end

"""
Compute the possible GCD cases of `f` and `g` with respect to the variable
of index `idx`.
Since specialization of the lower variables may change both the degrees of the
inputs and the degree of their GCD, the computation is split into branches
defined by equations and inequations:
- First split according to the leading coefficients of the inputs, in order to
account for possible degree drops (including the cases in which one or both
inputs specialize to zero)
- For each resulting pair of fixed input degrees, use the principal subresultant
coefficients to determine the possible GCD degrees: all coefficients at smaller
indices vanish, while the coefficient at the GCD index is nonzero.
The collecting method returns a `Vector{__SubresultantGCDCase}`. The callback
method calls `visit(case)` for each generated branch and returns `nothing`.

    ```jldoctest
    julia> R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b]);
    julia> cases = __algebraic_subresultant_gcd_cases(x^2-a, x-b, 1);
    julia> # The cases are b^2-a ≠ 0, with GCD 1, and b^2-a = 0, with GCD x-b
    julia> special = only(filter(c -> c.degree == 1, cases));
    julia> special.gcd == x-b
    true
    julia> certificate = __algebraic_subresultant_cofactor(special);
    julia> certificate.quotient == x+b
    true
    julia> certificate.remainder == b^2-a
    true
    ```
"""
function __algebraic_subresultant_gcd_cases(f::P, g::P, idx::Int; kwargs...) where {P<:MPolyRingElem}
    result = __SubresultantGCDCase{P}[]
    __algebraic_subresultant_gcd_cases(c -> push!(result, c), f, g, idx; kwargs...)
    return result
end

function __algebraic_subresultant_gcd_cases(visit::F, f::P, g::P, idx::Int;
            equations=P[], inequations=P[]) where {F,P<:MPolyRingElem}
    R = parent(f)
    context = __subresultant_context(R, equations, inequations)
    context === nothing && return nothing
    E, I = context
    __subresultant_degree_cases(f, idx, E, I) do left, E_left, I_left
        __subresultant_degree_cases(g, idx, E_left, I_left) do right, E_both, I_both
            __subresultant_fixed_degree_cases(visit, left, right, idx, E_both, I_both)
        end
    end
    return nothing
end

"""
Construct the normalized equations and inequations used in the subresultant case splitting.\
Assume the caller supplies conditions in `R` independent of the elimination variable.
Normalize each condition by a nonzero constant from the coefficient field and discard conditions
that are already implied by the current equations and inequations.\
Return `nothing` if an immediate contradiction is detected.
"""
function __subresultant_context(R, equations, inequations)
    P = elem_type(R)
    E, I = P[], P[]
    for (nonzero, polynomials) in ((false, equations), (true, inequations)), p in polynomials
        context = __subresultant_add_condition(E, I, deepcopy(p), nonzero)
        context === nothing && return nothing
        E, I = context
    end
    return E, I
end

"""
Add the condition `p ≠ 0` if `nonzero` is true, and `p = 0` otherwise.\
Normalize `p` by its leading coefficient: if the condition is already determined
by `E` and `I`, leave the context unchanged or return `nothing` when it contradicts
the current context; otherwise add the normalized polynomial to the appropriate vector.
"""
function __subresultant_add_condition(E::Vector{P}, I::Vector{P}, p::P, nonzero::Bool) where {P<:MPolyRingElem}
    if is_constant(p)
        return iszero(p) == nonzero ? nothing : (E, I)
    end
    h = divexact(p, leading_coefficient(p))
    status = __subresultant_condition_status(h, E, I)
    status === nothing || return status == nonzero ? (E, I) : nothing
    return nonzero ? (E, [I; h]) : ([E; h], I)
end

"""
Try to detect from the current equations and inequations whether `h` is zero or nonzero.\
Return `false` if `h = 0` is implied, `true` if `h ≠ 0` is implied, and
`nothing` if neither condition can be established easily.

The test uses constant values, conditions already present in `E` or `I`,
divisibility relations, and, when `reduce=true`, reduction of `h` with respect
to the current conditions.
"""
function __subresultant_condition_status(h::P, E::Vector{P}, I::Vector{P}; reduce::Bool=true) where {P<:MPolyRingElem}
    is_constant(h) && return !iszero(h)
    h in E && return false
    h in I && return true
    any(e -> !iszero(e) && first(Oscar.divides(h, e)), E) && return false
    any(q -> !iszero(q) && first(Oscar.divides(q, h)), I) && return true
    reduced = reduce ? __subresultant_reduce_condition(h, E, I) : h
    is_constant(reduced) && return !iszero(reduced)
    __subresultant_guard_nonzero(reduced, I) === true && return true
    return nothing
end

"""
Split the current equations and inequations into branches on which the degree of `p`
in the variable of index `idx` is fixed.\
Write `p = c_d*x^d + ... + c_0` and consider successively the cases

c_d != 0
c_d = 0, c_(d-1) != 0
...
c_d = ... = c_0 = 0

For each branch, call `visit(current, E_branch, I_branch)`, where `current` is
obtained from `p` by removing the leading terms whose coefficients are forced
to vanish on that branch.
"""
function __subresultant_degree_cases(visit::F, p::P, idx::Int, E::Vector{P}, I::Vector{P}) where {F,P<:MPolyRingElem}
    current = p
    x = gen(parent(p), idx)
    while !iszero(current)
        d = degree(current, idx)
        h = leading_coefficient(current, idx)
        nonzero, zero_case = __subresultant_condition_cases(E, I, h)
        nonzero === nothing || visit(current, nonzero...)
        zero_case === nothing && return nothing
        E, I = zero_case
        current -= h*x^d
    end
    visit(current, E, I)
    return nothing
end

"""
Construct the two complementary branches defined by `p != 0` and `p = 0`.
Normalize `p` by its leading coefficient and first try to certify its status from the
current equations and inequations. If its status is already known, return only the compatible
branch. Otherwise, add the normalized polynomial to the inequations in the nonzero branch and
to the equations in the zero branch.

Return `(nonzero_case, zero_case)`, where each case is either a pair `(E_branch, I_branch)` or `nothing`.
"""
function __subresultant_condition_cases(E::Vector{P}, I::Vector{P}, h::P) where {P<:MPolyRingElem}
    normalized = is_constant(h) ? h : divexact(h, leading_coefficient(h))
    status = __subresultant_condition_status(normalized, E, I)
    status === true && return (E, I), nothing
    status === false && return nothing, (E, I)
    return (E, [I; normalized]), ([E; normalized], I)
end

"""
Compute the possible GCD cases of `f` and `g` on a branch where,
under every specialization satisfying `E = 0` and `I ≠ 0`, their degrees in the
variable of index `idx` remain unchanged.

Distinguish the possible GCD degrees using the principal subresultant coefficients;
see Mishra, Algorithmic Algebra, §7.7.1, Lemma 7.7.8 and Corollary 7.7.9.

For each resulting branch, call `visit` with the corresponding `__SubresultantGCDCase`;
if `data` is supplied, reuse the given subresultant data instead of recomputing it.
"""
function __subresultant_fixed_degree_cases(visit::F, f::P, g::P, idx::Int, E::Vector{P}, I::Vector{P}; data=nothing) where {F,P<:MPolyRingElem}
    # Degree specialization can turn a nonconstant input into a constant or zero.
    f_zero = iszero(f)
    g_zero = iszero(g)
    # Handle branches where at least one input vanishes identically.
    if f_zero || g_zero
        if f_zero && g_zero
            value = zero(parent(f))
        else
            d = f_zero ? g : f
            value = degree(d, idx) == 0 ? one(parent(f)) : d
        end
        __subresultant_visit_case(visit, E, I, f, g, value, idx)
        return nothing
    end
    # If one input is a nonzero constant, the GCD is represented by one
    if min(degree(f, idx), degree(g, idx)) == 0
        __subresultant_visit_case(visit, E, I, f, g, one(parent(f)), idx)
        return nothing
    end
    # Compute the subresultant data only if they were not supplied by the caller
    data = data === nothing ? __algebraic_subresultant_data(f, g, idx) : data
    # Split according to the first nonzero principal subresultant coefficient
    for j in 0:min(data.degrees...)
        h = __algebraic_principal_subresultant(data, j)
        iszero(h) && continue
        nonzero, zero_case = __subresultant_condition_cases(E, I, h)
        # On the branch h ≠ 0, the GCD has degree j
        if nonzero !== nothing
            d = j == 0 ? one(parent(f)) : __algebraic_subresultant(data, j)
            __subresultant_visit_case(visit, nonzero..., f, g, d, idx)
        end
        # Continue only along the complementary branch h = 0
        zero_case === nothing && break
        E, I = zero_case
    end
    return nothing
end

"""
Construct the `__SubresultantGCDCase` defined by `E`, `I`, the input representatives
`f` and `g`, and the GCD representative `d`, and pass it to `visit`. Copy the data before
constructing the case, so that subsequent branches cannot modify it.
"""
function __subresultant_visit_case(visit, E, I, f::P, g::P, d::P, idx::Int) where {P<:MPolyRingElem}
    e, i, left, right, divisor = deepcopy((E, I, f, g, d))
    visit(__SubresultantGCDCase{P}(e, i, left, right, divisor, degree(d, idx), idx))
    return nothing
end

"""
Let `case` be a `__SubresultantGCDCase` obtained from `__algebraic_subresultant_gcd_cases`.
Compute a pseudo-division certificate with respect to `case.gcd` for `case.left` if
`polynomial=1`, or for `case.right` if `polynomial=2`.

Return `(divisor, quotient, remainder, multiplier)` satisfying

`multiplier`* `f` = `quotient` * `divisor` + `remainder`

where `f` is the input selected by `polynomial`. On the branch defined by `case.equations`
and `case.inequations`, the remainder vanishes and the multiplier is nonzero.\
If `case.gcd` is one, return the corresponding exact certificate directly; if
`quotient_only=true`, return only an independent copy of the quotient.
The cofactor is undefined for the case in which both inputs vanish and `case.gcd` is zero.
"""
function __algebraic_subresultant_cofactor(case::__SubresultantGCDCase;
        polynomial::Int=1, quotient_only::Bool=false)
    f = polynomial == 1 ? case.left : case.right
    R = parent(f)
    if isone(case.gcd)
        quotient = deepcopy(f)
        quotient_only && return quotient
        return (divisor=one(R), quotient=quotient, remainder=zero(R), multiplier=one(R))
    end
    S, _ = polynomial_ring(R, :__subresultant_variable; cached=false)
    F = iszero(f) ? zero(S) : S(AbstractAlgebra.coefficients(f, case.variable_index))
    G = S(AbstractAlgebra.coefficients(case.gcd, case.variable_index))
    q, r, multiplier = __subresultant_pseudo_certificate(F, G; compute_multiplier=!quotient_only)
    x = gen(R, case.variable_index)
    quotient = deepcopy(evaluate(q, x))
    quotient_only && return quotient
    return (divisor=deepcopy(case.gcd), quotient=quotient, remainder=deepcopy(evaluate(r, x)), multiplier=deepcopy(multiplier))
end

###############################################################################
#####  SQUARE-FREE AND LCM CASES
###############################################################################

"""
Store a `polynomial` replacement together with the `equations` and `inequations`
defining the branch on which the replacement is valid.
"""
struct __SubresultantReplacement{P<:MPolyRingElem}
    equations::Vector{P}
    inequations::Vector{P}
    polynomial::P
end

"""
Compute square-free replacements of `p` with respect to the variable of index `idx`,
splitting according to the conditions on its coefficients.

On each branch, remove from `p` the leading terms whose coefficients are forced to vanish.
If the resulting polynomial `f` has positive degree in the variable of index `idx`, compute the GCD of `f` and its separant:
a pseudo-cofactor of `f` with respect to this GCD is then used as the square-free replacement; on
the branch, it differs from the square-free part of `f` only by a factor independent of the chosen variable and nonzero
under the branch conditions.

The `polynomial` stored in each `__SubresultantReplacement` has the same roots as the corresponding
specialization of `p`. If `p` specializes identically to zero, store zero; if it specializes to a
nonzero polynomial independent of the chosen variable, store one.

If `data` is supplied, reuse the subresultant data of `p` and its derivative only on branches where
the degree of `p` remains unchanged.

The method without `visit` returns all resulting cases as a vector of `__SubresultantReplacement`.
The callback method instead passes each case to `visit` as soon as it is produced and returns `nothing`.

    ```jldoctest
    julia> R, (x, a) = polynomial_ring(QQ, [:x, :a]);
    julia> cases = __algebraic_subresultant_squarefree_cases(x^2-a, 1);
    julia> length(cases)
    2
    julia> regular = only(filter(c -> c.polynomial == x^2-a, cases));
    julia> regular.inequations == [a]
    true
    julia> special = only(filter(c -> c.polynomial == x, cases));
    julia> special.equations == [a]
    true
    ```
"""
function __algebraic_subresultant_squarefree_cases(p::P, idx::Int; kwargs...) where {P<:MPolyRingElem}
    result = __SubresultantReplacement{P}[]
    __algebraic_subresultant_squarefree_cases(c -> push!(result, c), p, idx; kwargs...)
    return result
end

function __algebraic_subresultant_squarefree_cases(
    visit::F, p::P, idx::Int; equations=P[], inequations=P[], data=nothing
) where {F,P<:MPolyRingElem}
    R = parent(p)
    context = __subresultant_context(R, equations, inequations)
    context === nothing && return nothing
    E, I = context
    __subresultant_degree_cases(p, idx, E, I) do f, Ef, If
        if degree(f, idx) <= 0
            value = iszero(f) ? zero(R) : one(R)
            visit(__SubresultantReplacement{P}(deepcopy(Ef), deepcopy(If), value))
        else
            # In characteristic zero, preserving deg(f) also preserves deg(f').
            reusable = f == p ? data : nothing
            __subresultant_fixed_degree_cases(f, derivative(f, idx), idx, Ef, If; data=reusable) do c
                q = reusable === nothing ? __algebraic_subresultant_cofactor(c; quotient_only=true) :
                    __algebraic_subresultant_cofactor(reusable, c.degree; quotient_only=true)
                visit(__SubresultantReplacement{P}(c.equations, c.inequations, q))
            end
        end
    end
    return nothing
end

"""
Compute LCM replacements of `f` and `g` with respect to the variable of index
`idx`. On each branch determined by possible degree drops of the inputs, compute
their GCD. If one input is zero, use zero as the replacement. Otherwise, multiply
a pseudo-cofactor of the first input with respect to the GCD by the second input.
The result differs from the LCM only by a factor independent of the chosen
variable and nonzero on the branch.

Consequently, the roots of the stored `polynomial` are the union of the roots of the
two specialized input polynomials. Root multiplicities are those of the LCM, namely
the larger of the two multiplicities in the inputs; in particular, the replacement
need not be square-free.

The method without `visit` returns all resulting cases as a vector of `__SubresultantReplacement`.
The callback method passes each case to `visit` as soon as it is produced and returns `nothing`.
"""
function __algebraic_subresultant_lcm_cases(f::P, g::P, idx::Int; kwargs...) where {P<:MPolyRingElem}
    result = __SubresultantReplacement{P}[]
    __algebraic_subresultant_lcm_cases(c -> push!(result, c), f, g, idx; kwargs...)
    return result
end

function __algebraic_subresultant_lcm_cases(
    visit::F, f::P, g::P, idx::Int; kwargs...
) where {F,P<:MPolyRingElem}
    __algebraic_subresultant_gcd_cases(f, g, idx; kwargs...) do c
        value = if iszero(c.left) || iszero(c.right)
            zero(parent(f))
        else
            __algebraic_subresultant_cofactor(c; quotient_only=true) * c.right
        end
        visit(__SubresultantReplacement{P}(c.equations, c.inequations, value))
    end
    return nothing
end

###############################################################################
#####  SUBRESULTANT OPTIMIZATIONS
###############################################################################
"""
Compute a pseudo-division certificate for `f` with respect to `g`, used only for cofactor certificates.
Return `(q, r, m)` such that

    m*f = q*g + r

with `degree(r) < degree(g)`.\
At each reduction step, multiply by the leading coefficient of `g` when it is not one,
avoiding exact-divisibility tests in the coefficient ring.
With `compute_multiplier=false`, return `nothing` in place of the multiplier without constructing its power.
"""
function __subresultant_pseudo_certificate(f::U, g::U; compute_multiplier::Bool=true) where {U<:PolyRingElem}
    quotient, remainder, exponent = __subresultant_pseudo_division(f, g)
    multiplier = compute_multiplier ? leading_coefficient(g)^exponent : nothing
    return quotient, remainder, multiplier
end

"""
Try to certify that `p` is nonzero using the inequations in `I`, remove from `p` factors
whose nonvanishing is certified by the guards in `I`; several guards may jointly certify
a product; for example, `a != 0` and `b != 0` certify `a*b != 0`.

Return `true` if all nonconstant factors of `p` can be removed, and `nothing` otherwise.
In particular, `nothing` gives no information about whether `p` vanishes.

This optimization is restricted to small inputs, so it remains only a sufficient nonvanishing test.
"""
function __subresultant_guard_nonzero(p::P, I::AbstractVector{P}) where {P<:MPolyRingElem}
    is_constant(p) && return iszero(p) ? nothing : true
    # Reject large inputs before allocating the filtered guard vector.
    length(p) > __SUBRESULTANT_GUARD_TERM_LIMIT && return nothing
    guards = P[]
    for q in I
        is_constant(q) && continue
        length(q) > __SUBRESULTANT_GUARD_TERM_LIMIT && return nothing
        push!(guards, q)
    end
    isempty(guards) && return nothing
    reduced = __algebraic_strip_guard_factors(p, guards;
                    max_variables=__SUBRESULTANT_GUARD_MAX_VARIABLES)
    return is_constant(reduced) ? true : nothing
end

"""
Reduce the polynomial `p` using the equations in `E` whenever this operation
can be certified to preserve the conditions of the system, subtracting a multiple
of an equation or using a nonvanishing initial certified by `I` and by
equations with strictly lower leaders.
If such a certificate cannot be obtained, skip that reduction.\
By construction, the returned polynomial has the same vanishing and nonvanishing
behavior as `p` under the given conditions. However, a remainder that is nonzero
as a polynomial may still vanish under every specialization satisfying `E = 0` and
`I != 0`, and therefore does not by itself certify that `p` is nonzero.

If `reduction_data` is provided, use it to store and reuse the information computed from
`E` during repeated reductions, as long as `E` remains unchanged.
"""
function __subresultant_reduce_condition(p::P, E::AbstractVector{P}, I::AbstractVector{P};
        reduction_data=nothing) where {P<:MPolyRingElem}
    is_constant(p) && return p
    isempty(E) && return p
    reducers = if reduction_data === nothing
            __algebraic_reduction_data(E)
        else
            reduction_data[] === nothing && (reduction_data[] = __algebraic_reduction_data(E))
            reduction_data[]
        end
    return __subresultant_reduce_prepared(p, reducers, I)
end

"""
Precompute the data from `E` used during repeated reductions; for each nonconstant equation,
store the polynomial, the index of its leader, and its degree in that variable.
Compute and cache its initial only when a reduction needs it.

Using this helper, since the same equations may be used several times during the reduction
process and the recursive reduction of initials, we compute their leaders and corresponding
degrees only once.\
Sort these data by leader index so that the reduction algorithm can process
the equations in the required order and reuse the same data in recursive calls.
"""
function __algebraic_reduction_data(E::AbstractVector{P}) where {P<:MPolyRingElem}
    reducers = __AlgebraicReductionData{P}[]
    sizehint!(reducers, length(E))
    for f in E
        is_constant(f) && continue
        ds = degrees(f)
        k = something(findfirst(>(0), ds))
        push!(reducers, __AlgebraicReductionData{P}(f, k, ds[k], nothing))
    end
    sort!(reducers; by=r -> r.variable_index, alg=Base.Sort.MergeSort)
    return reducers
end

"""
Reduce `p` using the equations represented by `reducers`, starting from `first_reducer`.\
Return the polynomial obtained after all certified reductions.
"""
function __subresultant_reduce_prepared(p::P, reducers, I::AbstractVector{P}, first_reducer::Int=1) where {P<:MPolyRingElem}
    is_constant(p) && return p
    q = p
    for i in first_reducer:lastindex(reducers)
        reducer = reducers[i]
        f, k, df = reducer.polynomial, reducer.variable_index, reducer.degree
        dq = degree(q, k)
        dq < df && continue
        h = __algebraic_reducer_initial!(reducer)
        scalar_initial = is_constant(h)
        safe_initial = scalar_initial ? true : nothing
        while !iszero(q) && dq >= df
            a = leading_coefficient(q, k)
            exact, b = scalar_initial ? (true, divexact(a, h)) : Oscar.divides(a, h)
            if !exact
                if safe_initial === nothing
                    safe_initial = __subresultant_guard_nonzero(h, I)
                    if safe_initial !== true
                        lower_start = i + 1
                        while lower_start <= lastindex(reducers) && reducers[lower_start].variable_index == k
                            lower_start += 1
                        end
                        if lower_start <= lastindex(reducers)
                            reduced_initial = __subresultant_reduce_prepared(h, reducers, I, lower_start)
                            if reduced_initial != h
                                safe_initial = __subresultant_guard_nonzero(reduced_initial, I)
                            end
                        end
                    end
                end
                (safe_initial === true) || break
            end
            shift = dq - df
            shifted = iszero(shift) ? f : gen(parent(p), k)^shift*f
            q = exact ? q - b*shifted : h*q - a*shifted
            dq = degree(q, k)
        end
        is_constant(q) && return q
    end
    return q
end

"""
Process one branch of the subresultant GCD computation.

First determine whether the degrees of `f` and `g` change under specialization `E = 0`, `I != 0`.
If a leading coefficient is certified nonzero, the corresponding degree is preserved; if it is certified
zero, remove the leading term and continue with the next coefficient; if its status cannot be certified,
return `kind=:split` together with that coefficient as the new splitting condition.

Once the degrees of both polynomial inputs are determined, consider the principal subresultant
coefficients starting from `first_index`. Coefficients certified zero are skipped; at the first coefficient
certified nonzero, return `kind=:gcd` and the corresponding `__SubresultantGCDCase`.
If its status cannot be certified, return `kind=:split`: `case` represents the branch where the coefficient is
nonzero, while `condition` defines the complementary branch where it is zero.\
Return `kind=:inconsistent` if all possible principal subresultant coefficients are certified zero.

If a degree drop changes one of the two polynomial inputs, restart from index zero and discard any
supplied subresultant data; otherwise, `first_index` may be used to skip principal subresultant
coefficients already known to vanish, and `data` may be reused.
"""
function __subresultant_gcd_step(f::P, g::P, idx::Int, E::Vector{P}, I::Vector{P};
        data=nothing, first_index::Int=0) where {P<:MPolyRingElem}
    reduction_data = Ref{Union{Nothing,Vector{__AlgebraicReductionData{P}}}}(nothing)
    left, condition = __subresultant_degree_step(f, idx, E, I; reduction_data)
    condition === nothing || return (kind=:split, condition=condition, case=nothing)
    right, condition = __subresultant_degree_step(g, idx, E, I; reduction_data)
    condition === nothing || return (kind=:split, condition=condition, case=nothing)
    if iszero(left) || iszero(right)
        d = iszero(left) ? right : left
        !iszero(d) && degree(d, idx) == 0 && (d = one(d))
        c = __SubresultantGCDCase{P}(E, I, left, right, d, degree(d, idx), idx)
        return (kind=:gcd, case=c)
    elseif min(degree(left, idx), degree(right, idx)) == 0
        c = __SubresultantGCDCase{P}(E, I, left, right, one(f), 0, idx)
        return (kind=:gcd, case=c)
    end
    # first_index > 0 is valid only for the original pair.
    # A degree drop changes the pair, so lower PSCs must be considered again.
    unchanged = left == f && right == g
    effective_first_index = unchanged ? first_index : 0
    reusable = unchanged ? data : nothing
    sequence = if reusable === nothing
        __algebraic_subresultant_data(left, right, idx; minimum_degree=effective_first_index)
        else
            reusable
        end
    maxj = min(sequence.degrees...)
    for j in effective_first_index:maxj
        h = __algebraic_principal_subresultant(sequence, j)
        iszero(h) && continue
        reduced = __subresultant_reduce_condition(h, E, I; reduction_data)
        status = __subresultant_condition_status(reduced, E, I; reduce=false)
        status === false && continue
        d = iszero(j) ? one(f) :
            __algebraic_subresultant(sequence, j)
        guards = status === true ? I : [I; reduced]
        c = __SubresultantGCDCase{P}(E, guards, left, right, d, j, idx)
        return status === true ? (kind=:gcd, case=c) :
                                (kind=:split, condition=reduced, case=c)
    end
    return (kind=:inconsistent,)
end

"""
Determine the degree of `p` in the variable of index `idx` under the specializations
satisfying `E = 0` and `I ≠ 0`, stopping at the first leading coefficient whose vanishing
cannot be decided.

Discard leading terms whose coefficients are certified zero. Return the first
remaining polynomial together with `nothing` if its leading coefficient is
certified nonzero, or with that coefficient if its status is undecided.

Return zero and `nothing` if all coefficients are certified zero.
"""
function __subresultant_degree_step(p::P, idx::Int, E::Vector{P}, I::Vector{P};
        reduction_data=nothing) where {P<:MPolyRingElem}
    current = p
    x = gen(parent(p), idx)
    while !iszero(current)
        d = degree(current, idx)
        h = leading_coefficient(current, idx)
        reduced = __subresultant_reduce_condition(h, E, I; reduction_data)
        status = __subresultant_condition_status(reduced, E, I; reduce=false)

        status === true && return current, nothing
        status === nothing && return current, reduced
        current -= h*x^d
    end
    return current, nothing
end

"""
Return from `f` factors whose non-vanishing is certified by the guards in `I`.

If `max_variables = nothing`, remove with full multiplicity every irreducible
factor of `f` that divides at least one polynomial in `I`. Equivalently, if Q = prod(I),
the returned polynomial generates the saturated ideal `(f) : Q^∞`.

If `max_variables` is an integer, instead, restrict factor cancellation to GCD computations
involving at most `max_variables` active variables and sufficiently few terms
(for the moment, respectively fixed to 8 and 2).
A skipped GCD leaves the corresponding factors in `f`, so the result need not generate the
full saturation; nevertheless, on a branch where every polynomial in `I` is nonzero, the
returned polynomial has the same vanishing behavior as `f`.
"""
function __algebraic_strip_guard_factors(f, I;
        max_variables::Union{Nothing,Int}=nothing)
    f_degrees = nothing
    for q in I
        q_degrees = max_variables === nothing ? nothing :
            degrees(q isa ActionPolyRingElem ? Oscar.data(q) : q)
        while !is_constant(f)
            if max_variables !== nothing
                if f_degrees === nothing
                    f_degrees = degrees(f isa ActionPolyRingElem ? Oscar.data(f) : f)
                end
                active = 0
                common = false
                for i in eachindex(f_degrees, q_degrees)
                    in_f = f_degrees[i] > 0
                    in_q = q_degrees[i] > 0
                    common = common || (in_f && in_q)
                    if in_f || in_q
                        active += 1
                        active > max_variables && break
                    end
                end
                common || break
                active > max_variables && break
                limit = __guard_gcd_term_limit(active)
                limit == 0 && break
                max(length(f), length(q)) > limit && break
            end
            d = gcd(f, q)
            is_constant(d) && break
            # Remove the first certified common factor.
            f = divexact(f, d)
            f_degrees = nothing
            # If the same factor occurs with higher multiplicity in f, exact
            # divisibility is much cheaper than recomputing gcd(f, q).
            while !is_constant(f)
                exact, quotient = Oscar.divides(f, d)
                exact || break
                f = quotient
                f_degrees = nothing
            end
        end
        is_constant(f) && break
    end
    return f
end

"""
Return the maximum number of terms allowed in a guard-factor GCD computation
with `active` active variables, or zero if the computation should be skipped.
"""
function __guard_gcd_term_limit(active::Int)
    active == 1 && return __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_1
    active == 2 && return __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_2
    active == 3 && return __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_3
    active == 4 && return __SUBRESULTANT_GUARD_GCD_TERM_LIMIT_VAR_4
    return 0
end

###############################################################################
#####  SUBRESULTANTS OPTIMIZATIONS
###############################################################################

"""Borrow a reducer and cache its fixed leader data and, lazily, its initial. """
mutable struct __AlgebraicReductionData{P}
    polynomial::P
    variable_index::Int
    degree::Int
    initial::Union{Nothing,P}
end

"""Compute the reducer initial once. Reducer and cached polynomial are treated as read-only. """
function __algebraic_reducer_initial!(data::__AlgebraicReductionData)
    if data.initial === nothing
        data.initial = leading_coefficient(data.polynomial, data.variable_index)
    end
    return data.initial
end

"""Compute the quotient, remainder and exponent of the fixed divisor initial, without forming its power."""
function __subresultant_pseudo_division(f::U, g::U) where {U<:PolyRingElem}
    S = parent(f)
    isone(g) && return f, zero(S), 0
    dg = degree(g)
    h = leading_coefficient(g)
    monic = isone(h)
    quotient, remainder = zero(S), f
    exponent = 0
    while !iszero(remainder) && degree(remainder) >= dg
        a = leading_coefficient(remainder)
        shift = degree(remainder) - dg
        term = AbstractAlgebra.shift_left(S(a), shift)
        quotient, remainder = __polynomial_pseudo_step(quotient, remainder, g, h, term, monic)
        monic || (exponent += 1)
    end
    return quotient, remainder, exponent
end

"""Apply one pseudo-division step; use `quotient=nothing` when no quotient is required. """
function __polynomial_pseudo_step(quotient, remainder, divisor, initial, term, monic::Bool)
    if quotient !== nothing
        quotient = monic ? quotient + term : initial * quotient + term
    end
    remainder = monic ? remainder - term * divisor : initial * remainder - term * divisor
    return quotient, remainder
end