const __DIFFERENTIAL_ZERO_INSPECTION_MINIMUM_TERMS = 32
const __DIFFERENTIAL_ZERO_INSPECTION_CACHE_LIMIT = 32
const __DIFFERENTIAL_MAX_TERMS_FACTORIZATION = 32
const __LARGE_REDUCTION = 32
const __PENDING_LARGE_REDUCTION = 24

""" Estimate whether a nonlinear pseudo-step is large enough to justify coefficient-level shortcuts. """
function __action_large_nonlinear_reduction(remainder, reducer, remainder_initial, reducer_initial)
    estimated_pair_products = length(remainder) * max(1, length(reducer_initial)) +
        length(remainder_initial) * length(reducer)
    return (
        length(remainder) >= 5000 ||
        length(reducer) >= 1000 ||
        length(remainder_initial) >= 2000 ||
        estimated_pair_products >= 250_000
    )
end

###############################################################################
#####  Early simplification and contradiction checks
###############################################################################

# Extract variables forced nonzero by monomial factors of the retained guards.
# No factorization, coefficient GCD or product of guards is needed,
# even when the target polynomial is very large.
function __algebraic_nonzero_monomial_variables(guards)
    indices = BitSet()
    for q in guards
        is_constant(q) && continue
        raw = q isa ActionPolyRingElem ? AbstractAlgebra.data(Oscar.data(q)) : q
        shift, _ = AbstractAlgebra.deflation(raw)
        for i in eachindex(shift)
            iszero(shift[i]) || push!(indices, i)
        end
    end
    return indices
end

# Cancel all powers of these nonzero variables from a constraint in one pass.
function __algebraic_strip_monomial_variables(p, indices::BitSet)
    (isempty(indices) || is_constant(p)) && return p
    raw = p isa ActionPolyRingElem ? AbstractAlgebra.data(Oscar.data(p)) : p
    shift, _ = AbstractAlgebra.deflation(raw)
    for i in eachindex(shift)
        i in indices || (shift[i] = 0)
    end
    all(iszero, shift) && return p
    reduced = AbstractAlgebra.deflate(raw, shift, ones(Int, length(shift)))
    if p isa ActionPolyRingElem
        stored = Oscar.data(p)
        return parent(p)(typeof(stored)(reduced, parent(stored)))
    end
    return reduced
end

function __differential_pending_inequation_contradiction(state)
    isempty(state.Q_inequations) && return false
    data = __differential_janet_data!(state)
    changed = false
    for i in reverse(eachindex(state.Q_inequations))
        q = state.Q_inequations[i]
        is_zero(q) && return true
        reduced = if data === nothing || is_constant(q)
                q
            else
                __normal_form_cached(q, data, state.normal_form_cache, state.normal_form_certificate_cache).nf
            end
        is_zero(reduced) && return true
        if is_constant(reduced)
            deleteat!(state.Q_inequations, i)
        elseif reduced != q
            state.Q_inequations[i] = reduced
            changed = true
        end
    end
    changed && unique!(state.Q_inequations)
    return false
end

"""
Remove factors of short pending equations that divide a short processed or pending inequation,
with full multiplicity, using the existing gcd-based cancellation and without computing factorizations.
Keep every inequation and leave the processed equation set unchanged.

If `p = d*r` and `d` divides a guard `q ≠ 0`, then `d` is nonzero on every
solution of the branch, so `p = 0` is equivalent to `r = 0`.
"""
function __differential_cancel_pending_guard_factors!(
    state;
    maximum_equation_terms::Int=__DIFFERENTIAL_MAX_TERMS_FACTORIZATION,
    maximum_guard_terms::Int=__DIFFERENTIAL_MAX_TERMS_FACTORIZATION
)
    coefficient_ring(state.ring) isa Field || return state
    isempty(state.Q_equations) && return state
    P = eltype(state.T_equations)
    guards = P[]
    for q in Iterators.flatten((state.T_inequations, state.Q_inequations))
        is_constant(q) && continue
        length(q) <= maximum_guard_terms || continue
        push!(guards, q)
    end
    unique!(guards)
    isempty(guards) && return state
    for i in eachindex(state.Q_equations)
        p = state.Q_equations[i]
        is_constant(p) && continue
        length(p) <= maximum_equation_terms || continue
        state.Q_equations[i] = __algebraic_strip_guard_factors(p, guards)
    end
    return state
end

"""
Look for contradictions before performing algebraic splitting.
Use equations a*v + b = 0, where v is the leader, b contains only lower
variables, and a is a nonzero element of the coefficient field.
Substitute v = -b/a and the corresponding derivations into
the constraints being checked. For example, u = 0 forces u_x = 0 and
contradicts u_x ≠ 0, even when both constraints are still in S_Q.
These substitutions require no extra nonvanishing assumptions, since a is invertible.
An initial depending on the jets may vanish on a part of the solution set that the
algebraic splitting has not yet separated.
"""
function __differential_state_has_safe_contradiction(state;
        scan_equations::Bool=true)
    coefficient_ring(state.ring) isa Field || return false
    E = [state.T_equations; state.Q_equations]
    I = [state.T_inequations; state.Q_inequations]
    any(p -> is_constant(p) && !is_zero(p), E) && return true
    any(is_zero, I) && return true
    # Equations skipped here must remain eligible for a later full check.
    scan_equations || empty!(state.safe_prechecked_equations)
    if !scan_equations && all(is_constant, I)
        empty!(state.safe_prechecked_inequations)
        empty!(state.safe_precheck_reducers)
        return false
    end
    E = unique!(filter!(!is_zero, E))
    if isempty(E)
        empty!(state.safe_prechecked_equations)
        empty!(state.safe_prechecked_inequations)
        union!(state.safe_prechecked_inequations, I)
        empty!(state.safe_precheck_reducers)
        return false
    end
    P = eltype(state.T_equations)
    reducers = P[]
    leaders = P[]
    source_indices = scan_equations ? Int[] : nothing
    positions = scan_equations ? zeros(Int, length(E)) : nothing
    sizehint!(reducers, length(E))
    sizehint!(leaders, length(E))
    scan_equations && sizehint!(source_indices, length(E))
    for (i, f) in pairs(E)
        v = leader(f)
        scan_equations && push!(leaders, v)
        variable_index = var_index(Oscar.data(v))
        degree(Oscar.data(f), variable_index) == 1 || continue
        h = state.ring(coeff(Oscar.data(f), [variable_index], [1]))
        is_constant(h) || continue
        is_zero(h) && continue
        reducer = isone(h) ?
            f :
            divexact(f, h)
        push!(reducers, reducer)
        if scan_equations
            push!(source_indices, i)
            positions[i] = length(reducers)
        else
            push!(leaders, v)
        end
    end
    current_reducers = Set(reducers)
    if isempty(reducers)
        empty!(state.safe_prechecked_equations)
        scan_equations && union!(state.safe_prechecked_equations, E)
        empty!(state.safe_prechecked_inequations)
        union!(state.safe_prechecked_inequations, I)
        empty!(state.safe_precheck_reducers)
        return false
    end
    new_reducer = any(reducer -> !(reducer in state.safe_precheck_reducers), current_reducers)
    equation_targets = if !scan_equations
            nothing
        elseif new_reducer
            trues(length(E))
        else
            [!(f in state.safe_prechecked_equations) for f in E]
        end
    inequation_targets = if new_reducer
            trues(length(I))
        else
            [!(q in state.safe_prechecked_inequations) for q in I]
        end
    if (!scan_equations || !any(equation_targets)) && !any(inequation_targets)
        empty!(state.safe_precheck_reducers)
        union!(state.safe_precheck_reducers, current_reducers)
        return false
    end
    # Prepare the equations used for substitution. If equations are also checked,
    # compute multiplicative directions using all equations in E, then keep the
    # directions belonging to the chosen reducers. Allow substitution at an
    # equation's leader only in those directions (janet_head=true).
    # For inequations, any derivative of a chosen reducer gives a valid substitution,
    # so no restriction from the full set of Janet directions is needed.
    # __differential_splitting uses this second case (scan_equations=false):
    # it checks inequations here and reduces equations when they are selected from S_Q.
    data = __differential_safe_substitution_data(reducers, leaders, source_indices)
    if scan_equations
        for i in eachindex(E)
            equation_targets[i] || continue
            reduced = __differential_reduce_unit_linear(E[i], data; skip=positions[i], janet_head=true)
            is_constant(reduced) && !is_zero(reduced) && return true
        end
    end
    for i in eachindex(I)
        inequation_targets[i] || continue
        q = I[i]
        is_constant(q) && continue
        reduced = __differential_reduce_unit_linear(q, data)
        is_zero(reduced) && return true
    end
    empty!(state.safe_prechecked_equations)
    scan_equations && union!(state.safe_prechecked_equations, E)
    empty!(state.safe_prechecked_inequations)
    union!(state.safe_prechecked_inequations, I)
    empty!(state.safe_precheck_reducers)
    union!(state.safe_precheck_reducers, current_reducers)
    return false
end

# Reuse the leaders collected while choosing reducers. For equations, derive Janet
# masks from the full equation set; for inequations, omit the unused masks entirely.
# Keep only the data needed for substitution, without polynomial-index dictionaries.
function __differential_safe_substitution_data(reducers::Vector{P}, leaders::Vector{P}, source_indices) where {P<:DifferentialPolyRingElem}
    shifts = Vector{Vector{Int}}(undef, length(leaders))
    by_symbol = [Int[] for _ in 1:n_action_indeterminates(parent(first(reducers)))]
    for (i, v) in pairs(leaders)
        jet = __jet_data(v)
        shifts[i] = jet.shift
        push!(by_symbol[jet.symbol], i)
    end
    masks = if source_indices === nothing
        nothing
    else
        full_masks = __action_multiplicative_masks((shifts=shifts, by_symbol=by_symbol))
        # Convert full-equation indices to reducer indices after computing the masks.
        symbols = zeros(Int, length(leaders))
        for (symbol, indices) in pairs(by_symbol), i in indices
            symbols[i] = symbol
        end
        foreach(empty!, by_symbol)
        for (i, source) in pairs(source_indices)
            push!(by_symbol[symbols[source]], i)
        end
        shifts = shifts[source_indices]
        full_masks[source_indices]
    end
    return (equations=reducers, shifts=shifts, by_symbol=by_symbol,
        multiplicative_variables=masks,
        prolongations=Dict{Tuple{Int,Tuple{Vararg{Int}}},P}())
end

"""
Substitute derivatives of monic equations that are linear in their leaders, from the fixed
`data.equations` into `p`. Exclude the equation at index `skip` when reducing an
equation set itself. Use ordinary differential divisibility in tails and inequations.
Set `janet_head=true` for equations and supply the Janet masks
of the full current equation set: only an admissible derivative may eliminate
an equation's leader. This retains prolongations needed for Janet completion,
even when they are redundant as differential equations.

For a monic equation with leader `v`, each derivative has the corresponding
derivative of `v` as leader, again with coefficient one. That jet is replaced by
the negative tail using (univariate) Horner evaluation. Every replacement contains
only smaller jets. Cache prolongations only while the equation set is unchanged.
"""
function __differential_reduce_unit_linear(
    p::P,
    data;
    skip::Int=0,
    janet_head::Bool=false
) where {P<:DifferentialPolyRingElem}
    while !is_constant(p)
        changed = false
        head = janet_head ? leader(p) : nothing
        for v in vars(p)
            jet = __jet_data(v)
            for i in data.by_symbol[jet.symbol]
                i == skip && continue
                alpha = jet.shift - data.shifts[i]
                __nonnegative_shift(alpha) || continue
                if janet_head && v == head
                    __admissible_shift(alpha, data.multiplicative_variables[i]) || continue
                end
                prolongation = get!(data.prolongations, (i, Tuple(alpha))) do
                    apply_action(data.equations[i], alpha)
                end
                replacement =v -prolongation
                p = __differential_substitute_jet(p,v,replacement)
                changed = true
                break
            end
            changed && break
        end
        changed || return p
    end
    return p
end

# State arguments use the fields of __DifferentialThomasState, defined in the
# decomposition file loaded after this one. Read polynomial types from its vectors.
function __differential_substitute_jet(p::P, v::P, value::P) where {P<:DifferentialPolyRingElem}
    coefficients = __action_univariate_coefficients_by_terms(p, v)
    result = pop!(coefficients)
    while !isempty(coefficients)
        result = pop!(coefficients) + result * value
    end
    return result
end

###############################################################################
#####  Janet zero reduction
###############################################################################

struct __PassivityCertificate{P<:ActionPolyRingElem}
    reductions::Vector{
        Tuple{P,Vector{Int}}
    }
end

"""
Keep the tail and the optional monic tail of one fixed leader-linear reducer.
Compute divisibility of the tail by the initial only when a reduction of
degree at least two needs it. Share these data within one zero inspection,
including its recursive initial checks; never reuse them with another basis.
"""
mutable struct __PreparedActionLinearReducer{P<:ActionPolyRingElem}
    tail::P
    monic_tail::Union{Nothing,P}
    monic_checked::Bool
end

"""
Try to certify that `p` reduces to zero on the current Thomas branch.
Return `false` when the shortcut is inconclusive or below the size threshold.
First remove terms annihilated by monic jet equations inside their Janet
cones, then share fixed linear-reducer tails across the recursive zero test.

Optionally cache positive proofs using only the equations and reductive
shifts actually used. Validate every reused proof against the current Janet
data; preserve independent polynomial snapshots and bound the cache by
`__DIFFERENTIAL_ZERO_INSPECTION_CACHE_LIMIT`.
Do not interpret a zero-inspection certificate as a normal-form multiplier identity.
"""
function __differential_fast_zero_inspection(
    p::P,
    data::__ActionJanetData{P};
    minimum_terms::Int=__DIFFERENTIAL_ZERO_INSPECTION_MINIMUM_TERMS,
    certificate_cache::Union{Nothing,Dict{P,__PassivityCertificate{P}}}=nothing
) where {P<:DifferentialPolyRingElem}
    length(p) >= minimum_terms || return false
    __require_same_parent(p, data.equations)
    if certificate_cache !== nothing
        certificate = get(certificate_cache, p, nothing)
        if certificate !== nothing
            __passivity_certificate_valid(certificate, data) && return true
            delete!(certificate_cache, p)
        end
    end
    reductions = certificate_cache === nothing ? nothing : Tuple{P,Vector{Int}}[]
    reduced = __differential_drop_zero_jet_terms(p, data, reductions)
    success = is_zero(reduced)
    if !success
        prepared_linear_reducers = Dict{
            Tuple{Int,Tuple{Vararg{Int}}},__PreparedActionLinearReducer{P},
        }()
        inspection = __janet_zero_reduce_with_reductions(reduced, data, reductions; prepared_linear_reducers)
        success = inspection.certificate_safe && is_zero(inspection.nf)
    end
    if success && certificate_cache !== nothing
        # Store detached proofs, not fake (zero, one) normal-form identities.
        # Bound retained large inputs; eviction only causes recomputation.
        unique!(reductions)
        key, certificate = __passivity_certificate_snapshot(p, reductions)
        length(certificate_cache) >= __DIFFERENTIAL_ZERO_INSPECTION_CACHE_LIMIT && empty!(certificate_cache)
        certificate_cache[key] = certificate
    end
    return success
end

function __passivity_certificate_valid(
    certificate::__PassivityCertificate{P},
    data::__ActionJanetData{P}
) where {P<:ActionPolyRingElem}
    for (equation, shift) in certificate.reductions
        index = get(data.indices, equation, 0)
        iszero(index) && return false
        __admissible_shift(shift, data.multiplicative_variables[index]) || return false
    end
    return true
end

"""
Remove terms containing a jet certified zero by an admissible prolongation
of an equation that is exactly one jet variable. Preserve the input and
record the equations and shifts used when `reductions` is supplied.
Assume the caller has validated the common parent of `p` and `data.equations`.

Use only the current Janet cones. In particular, do not remove a
non-multiplicative prolongation merely because it is a differential
consequence: that equation may still be required for Janet completion.
Leave nonlinear equations and nontrivial polynomial initials to the general
reducer. This helper is an internal prefilter for a zero test, not a public
normal form with a pseudo-reduction multiplier.
"""
function __differential_drop_zero_jet_terms(
    p::P,
    data::__ActionJanetData{P},
    reductions::Union{Nothing,Vector{Tuple{P,Vector{Int}}}}=nothing
) where {P<:DifferentialPolyRingElem}
    sources = Int[i for i in eachindex(data.equations) if data.equations[i] == data.leaders[i]]
    isempty(sources) && return p
    zero_indices = Int[]
    for v in vars(p)
        jet = __jet_data(v)
        for i in sources
            data.symbols[i] == jet.symbol || continue
            alpha = jet.shift - data.shifts[i]
            __nonnegative_shift(alpha) || continue
            __admissible_shift(alpha, data.multiplicative_variables[i]) || continue
            push!(zero_indices, var_index(Oscar.data(v)))
            reductions === nothing || push!(reductions, (data.equations[i], alpha))
            break
        end
    end
    isempty(zero_indices) && return p
    R = parent(p)
    universal_ring = base_ring(R)
    n_variables = ngens(universal_ring)
    context = MPolyBuildCtx(universal_ring)
    raw = Oscar.data(p)
    for (c, exponent) in zip(coefficients(raw), Oscar.AbstractAlgebra.exponent_vectors(raw))
        any(i -> i <= length(exponent) && !iszero(exponent[i]), zero_indices) && continue
        __action_pad_exponent_vector!(exponent, n_variables)
        push_term!(context, c, exponent)
    end
    return R(finish(context))
end

"""
Specialized zero test for pending differential constraints.
Record reduction provenance, reuse linear/nonlinear fast paths, and recursively
test lower initials when needed to remove a leading term.
An optional prepared_linear_reducers dictionary belongs to this fixed basis
and is shared with recursive calls, without extending the lifetime of data.
"""
function __janet_zero_reduce_with_reductions(
    p::P,
    data::__ActionJanetData{P},
    reductions::Union{Nothing,Vector{Tuple{P,Vector{Int}}}};
    prepared_linear_reducers=nothing
) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, data.equations)
    remainder = p
    while !is_constant(remainder)
        reduction = __is_janet_reducible_univariate_detailed(remainder, data)
        if reduction !== false
            if reductions !== nothing
                push!(reductions, (reduction.polynomial, collect(Int, reduction.alpha)))
            end
            v = reduction.variable
            reducer = reduction.prolongation
            # Janet reducibility of two single monomials gives exact zero.
            if length(remainder) == 1 && length(reducer) == 1
                remainder = R(0)
                continue
            end
            reducer_degree, reducer_initial = __action_reduction_univariate_data(reduction, data)
            if reducer_degree == 1
                prepared = if prepared_linear_reducers === nothing
                    nothing
                else
                    get!(prepared_linear_reducers, reduction.cache_key) do
                        tail = R(coeff(Oscar.data(reducer), [reduction.variable_index], [0]))
                        __PreparedActionLinearReducer{P}(tail, nothing, false)
                    end
                end
                remainder = __action_linear_pseudoreduce(remainder, reducer, v; reducer_initial, prepared,)
                continue
            end
            degree_difference = degree(Oscar.data(remainder), var_index(Oscar.data(v))) - reducer_degree
            remainder_initial = __action_univariate_initial(remainder, v)
            large_reduction = __action_large_nonlinear_reduction(
                remainder, reducer, remainder_initial, reducer_initial
            )
            # A lower initial certified zero may be dropped before an expensive
            # pseudo-step. Record the differential reductions used to certify it.
            if large_reduction && length(remainder_initial) >= __LARGE_REDUCTION
                initial_precheck = __janet_zero_reduce_with_reductions(
                    remainder_initial, data, reductions;
                    prepared_linear_reducers,
                )
                if is_zero(initial_precheck.nf)
                    remainder = __action_drop_univariate_initial_by_terms(remainder, v)
                    continue
                end
            end
            if large_reduction && length(reducer) <= 64
                remainder = __action_coefficientwise_pseudo(remainder, reducer, v).remainder
                continue
            end
            shifted_reducer = degree_difference == 0 ? reducer : v^degree_difference*reducer
            fast_pseudo = __action_try_monomial_fraction_mul_submul(
                reducer_initial, remainder, remainder_initial, shifted_reducer,
            )
            remainder = fast_pseudo === nothing ?
                reducer_initial*remainder - remainder_initial*shifted_reducer : fast_pseudo
            continue
        end
        # For the zero-test, reduce only the initial of an irreducible leader.
        v = leader(remainder)
        remainder_initial = __action_univariate_initial(remainder, v)
        reduced_initial = __janet_zero_reduce_with_reductions(
            remainder_initial, data, reductions;
            prepared_linear_reducers
        )
        !is_zero(reduced_initial.nf) && return (nf=remainder, certificate_safe=true)
        remainder = __action_drop_univariate_initial_by_terms(remainder, v)
    end
    return (nf=remainder, certificate_safe=true)
end

# Build one nonlinear pseudo-step coefficient-by-coefficient to avoid materializing its cancelling head term.
function __action_coefficientwise_pseudo(remainder::P, reducer::P, v::P) where {P<:ActionPolyRingElem}
    R = __require_same_parent(remainder, reducer, v)
    remainder_coefficients = __action_univariate_coefficients_by_terms(remainder, v)
    reducer_coefficients = __action_univariate_coefficients_by_terms(reducer, v)
    n = length(remainder_coefficients) - 1
    m = length(reducer_coefficients) - 1
    n >= 1 || return (remainder=remainder,)
    degree_difference = n - m
    remainder_initial = remainder_coefficients[end]
    reducer_initial = reducer_coefficients[end]
    monic = is_one(reducer_initial)
    output = Vector{P}(undef, n)
    for power in 0:(n - 1)
        left_coefficient = remainder_coefficients[power + 1]
        reducer_power = power - degree_difference
        if 0 <= reducer_power < m
            right_coefficient = reducer_coefficients[reducer_power + 1]
            result = __action_try_monomial_fraction_mul_submul(
                reducer_initial, left_coefficient, remainder_initial, right_coefficient,
            )
            if result === nothing
                result = reducer_initial*left_coefficient - remainder_initial*right_coefficient
            end
        elseif is_zero(left_coefficient)
            result = R(0)
        elseif monic
            result = left_coefficient
        else
            result = reducer_initial*left_coefficient
        end
        output[power + 1] = result
    end
    return (remainder=__action_rebuild_univariate_by_terms(output, v),)
end

# Delete the highest coefficient in one term scan; used after an initial is certified zero.
function __action_drop_univariate_initial_by_terms(p::P, v::P) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, v)
    universal_ring = base_ring(R)
    n_variables = ngens(universal_ring)
    iszero(p) && return R(0)
    variable_index = var_index(Oscar.data(v))
    raw = Oscar.data(p)
    top_degree = degree(raw, variable_index)
    context = MPolyBuildCtx(universal_ring)
    for (c, exponent) in zip(coefficients(raw), Oscar.AbstractAlgebra.exponent_vectors(raw))
        power = variable_index <= length(exponent) ? exponent[variable_index] : 0
        power == top_degree && continue
        __action_pad_exponent_vector!(exponent, n_variables)
        push_term!(context, c, exponent)
    end
    return R(finish(context))
end

# Dispatch a complete linear pseudo-reduction to monic/small-degree/general fast paths.
# The optional initial is read-only metadata from the current Janet basis.
# Supply prepared data only for this same reducer, leader and initial.
function __action_linear_pseudoreduce(
    p::P,
    reducer::P,
    v::P;
    reducer_initial::Union{Nothing,P}=nothing,
    prepared::Union{Nothing,__PreparedActionLinearReducer{P}}=nothing
) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, reducer, v)
    variable_index = var_index(Oscar.data(v))
    p_degree = iszero(p) ? -1 : degree(Oscar.data(p), variable_index)
    h = reducer_initial === nothing ? __action_univariate_initial(reducer, v) : reducer_initial
    reducer_tail = prepared === nothing ?
        R(coeff(Oscar.data(reducer), [variable_index], [0])) : prepared.tail
    # For p=a*v+c and reducer=h*v+b, preserve the original scale:
    # return c-(a/h)*b if h divides a, and h*c-a*b otherwise.
    if p_degree == 1
        p_initial = __action_univariate_initial(p, v)
        p_tail = R(coeff(Oscar.data(p), [variable_index], [0]))
        division = isone(h) ? (divides=true, quotient=p_initial) : __checked_divides(h, p_initial)
        if division.divides
            return __action_fast_mul_submul(R(1), p_tail, division.quotient, reducer_tail)
        end
        return __action_fast_mul_submul(h, p_tail, p_initial, reducer_tail)
    end
    # If the reducer is h*v, the monic branch reducer is v: evaluate at v=0.
    # Keep the degree-one path above separate because it may retain a factor h.
    is_zero(reducer_tail) && return R(coeff(Oscar.data(p), [variable_index], [0]))
    coefficients = __action_univariate_coefficients_by_terms(p, R, variable_index, p_degree)
    length(coefficients) == 1 && return coefficients[1]
    # On a branch where the initial h is nonzero, replace h*(v+b/h) by its monic
    # factor when possible, avoiding further exact divisions and pseudo scaling.
    monic_tail = if prepared !== nothing && prepared.monic_checked
        prepared.monic_tail
    else
        division = isone(h) ?
            (divides=true, quotient=reducer_tail) : __checked_divides(h, reducer_tail)
        tail = division.divides ? division.quotient : nothing
        if prepared !== nothing
            prepared.monic_tail = tail
            prepared.monic_checked = true
        end
        tail
    end
    if monic_tail !== nothing
        return __action_linear_monic_horner(coefficients, monic_tail)
    end
    if p_degree in (2, 3)
        return __action_linear_small_degree_pseudoreduce(coefficients, h, reducer_tail, p_degree)
    end
    scale = R(1)
    current = pop!(coefficients)
    while !isempty(coefficients)
        result = __action_linear_pseudo_step(h, reducer_tail, scale, current, pop!(coefficients))
        current, scale = result.current, result.scale
    end
    return current
end

function __action_fast_mul_submul(a::P, b::P, c::P, d::P) where {P<:ActionPolyRingElem}
    # Avoid expanding and flattening a product that is already known to vanish.
    if is_zero(c) || is_zero(d)
        return is_one(a) ? b : a*b
    elseif is_zero(a) || is_zero(b)
        return -(c*d)
    end
    result = __action_try_monomial_fraction_mul_submul(a, b, c, d)
    result !== nothing && return result
    return (a * b - c * d)
end

# Completely reduce by a monic linear reducer using Horner evaluation in the leader.
function __action_linear_monic_horner(coefficients::Vector{P}, monic_tail::P) where {P<:ActionPolyRingElem}
    unit = one(parent(monic_tail))
    current = pop!(coefficients)
    while !isempty(coefficients)
        current = __action_fast_mul_submul(unit, pop!(coefficients), current, monic_tail)
    end
    return current
end

# Completely reduce degree-2/3 inputs by a linear reducer with specialized small-degree kernels.
function __action_linear_small_degree_pseudoreduce(
    coefficients::Vector{P},
    h::P,
    reducer_tail::P,
    degree::Int
) where {P<:ActionPolyRingElem}
    R = parent(h)
    scale = R(1)
    current = coefficients[end]
    # Keep the exact-division-or-pseudo decision local to each coefficient transition.
    for k in degree:-1:1
        result = __action_linear_pseudo_step(h, reducer_tail, scale, current, coefficients[k])
        current, scale = result.current, result.scale
    end
    return current
end

# Execute one linear pseudo-reduction step with exact division when available.
# Share this transition between the small-degree and general loops.
@inline function __action_linear_pseudo_step(
    h::P,
    reducer_tail::P,
    scale::P,
    current::P,
    next_coefficient::P
) where {P<:ActionPolyRingElem}
    is_zero(current) && return (current=scale*next_coefficient, scale=scale)
    division = __checked_divides(h, current)
    if division.divides
        return (current=__action_fast_mul_submul(scale, next_coefficient, division.quotient, reducer_tail), scale=scale)
    end
    new_scale = h*scale
    return (current=__action_fast_mul_submul(new_scale, next_coefficient, current, reducer_tail), scale=new_scale)
end

###############################################################################
#####  Univariate coefficient extraction and reconstruction
###############################################################################

# Rebuild an action polynomial from coefficient polynomials using one raw multivariate builder.
function __action_rebuild_univariate_by_terms(coefficients::Vector{P}, v::P) where {P<:ActionPolyRingElem}
    R = parent(v)
    universal_ring = base_ring(R)
    n_variables = ngens(universal_ring)
    variable_index = var_index(Oscar.data(v))
    context = MPolyBuildCtx(universal_ring)
    for (offset, coefficient) in enumerate(coefficients)
        raw = Oscar.data(coefficient)
        for (c, exponent) in zip(Oscar.AbstractAlgebra.coefficients(raw), Oscar.AbstractAlgebra.exponent_vectors(raw))
            __action_pad_exponent_vector!(exponent, n_variables)
            exponent[variable_index] += offset - 1
            push_term!(context, c, exponent)
        end
    end
    return R(finish(context))
end

###############################################################################
#####  Speeding up arithmetic
###############################################################################

function __action_try_monomial_fraction_mul_submul(a::P, b::P, c::P, d::P) where {P<:ActionPolyRingElem}
    # The unit-denominator implementation avoids fraction flattening in the
    # polynomial-coefficient case, so try it first.
    unit_result = __action_try_unit_denominator_mul_submul(a, b, c, d)
    unit_result !== nothing && return unit_result
    R = __require_same_parent(a, b, c, d)
    universal_ring = base_ring(R)
    coefficient_field = coefficient_ring(universal_ring)
    numerator_one = try
        numerator(one(coefficient_field))
    catch
        return nothing
    end
    # This kernel flattens fractions of multivariate polynomials over a field;
    # in particular, QQ numerators live in ZZ, not a polynomial ring.
    # Unsupported coefficient representations must use ordinary arithmetic.
    numerator_one isa MPolyRingElem || return nothing
    numerator_ring = parent(numerator_one)
    coefficient_ring(numerator_ring) isa Field || return nothing
    n_coefficient_variables = ngens(numerator_ring)
    n_action_variables = ngens(universal_ring)
    temporary_ring, _ = polynomial_ring(
        coefficient_ring(numerator_ring),
        [Symbol("__thomas_flat_$(i)") for i in 1:(n_coefficient_variables + n_action_variables)])
    flat_a = __action_flatten_monomial_fractions(a, temporary_ring, n_action_variables, n_coefficient_variables)
    flat_a === nothing && return nothing
    flat_b = __action_flatten_monomial_fractions(b, temporary_ring, n_action_variables, n_coefficient_variables)
    flat_b === nothing && return nothing
    flat_c = __action_flatten_monomial_fractions(c, temporary_ring, n_action_variables, n_coefficient_variables)
    flat_c === nothing && return nothing
    flat_d = __action_flatten_monomial_fractions(d, temporary_ring, n_action_variables, n_coefficient_variables)
    flat_d === nothing && return nothing
    denominator_ab = flat_a.denominator_exponent + flat_b.denominator_exponent
    denominator_cd = flat_c.denominator_exponent + flat_d.denominator_exponent
    result_denominator = map(max, denominator_ab, denominator_cd)
    shift_ab = result_denominator - denominator_ab
    shift_cd = result_denominator - denominator_cd
    flat_ab = flat_a.polynomial * flat_b.polynomial
    flat_cd = flat_c.polynomial * flat_d.polynomial
    if any(x -> !iszero(x), shift_ab)
        exponent = vcat(shift_ab, zeros(Int, n_action_variables))
        flat_ab = __qqmpoly_monomial(temporary_ring, exponent) * flat_ab
    end
    if any(x -> !iszero(x), shift_cd)
        exponent = vcat(shift_cd, zeros(Int, n_action_variables))
        flat_cd = __qqmpoly_monomial(temporary_ring, exponent) * flat_cd
    end
    flat_numerator = flat_ab - flat_cd
    return __action_from_flat_monomial_fraction(
        R, flat_numerator, result_denominator, numerator_ring,
        n_action_variables, n_coefficient_variables)
end

# Construct one QQ multivariate monomial used by the flattened fraction arithmetic kernel.
function __qqmpoly_monomial(ring, exponent::Vector{Int})
    context = MPolyBuildCtx(ring)
    push_term!(context, one(coefficient_ring(ring)), exponent)
    return finish(context)
end

# Materialize one exponent vector at the requested ambient length for raw-ring arithmetic.
function __padded_exponent_vector(p, n_variables::Int)
    length(p) == 1 || return nothing
    exponent = exponent_vector(p, 1)
    __action_pad_exponent_vector!(exponent, n_variables)
    return (term=p, exponent=exponent)
end

# Reconstruct an action polynomial from the flattened monomial-fraction representation.
function __action_from_flat_monomial_fraction(
    R, flat_numerator, denominator_exponent::Vector{Int}, numerator_ring,
    n_action_variables::Int, n_coefficient_variables::Int
)
    iszero(flat_numerator) && return R(0)
    universal_ring = base_ring(R)
    coefficient_field = coefficient_ring(universal_ring)
    # Cancel a monomial factor common to the ENTIRE flattened numerator.
    # This is cheap and reduces the common denominator before constructing
    # individual FracFieldElem coefficients.
    cancellation = copy(denominator_exponent)
    first_term = true
    for term in terms(flat_numerator)
        exponent = exponent_vector(term, 1)
        if first_term
            @inbounds for i in 1:n_coefficient_variables
                cancellation[i] = min(cancellation[i], i <= length(exponent) ? exponent[i] : 0)
            end
            first_term = false
        else
            @inbounds for i in 1:n_coefficient_variables
                cancellation[i] = min(cancellation[i], i <= length(exponent) ? exponent[i] : 0)
            end
        end
    end
    reduced_denominator_exponent = denominator_exponent - cancellation
    numerator_contexts = Dict{Tuple{Vararg{Int}}, Any, }()
    for term in terms(flat_numerator)
        exponent = exponent_vector(term, 1)
        coefficient_exponent = zeros(Int, n_coefficient_variables)
        @inbounds for i in 1:n_coefficient_variables
            coefficient_exponent[i] = (i <= length(exponent) ? exponent[i] : 0) - cancellation[i]
        end
        action_exponent = ntuple(
            i -> (n_coefficient_variables + i <= length(exponent) ? exponent[n_coefficient_variables + i] : 0),
            n_action_variables)
        context = get!(numerator_contexts, action_exponent) do
            MPolyBuildCtx(numerator_ring)
        end
        push_term!(context, coeff(term, 1), coefficient_exponent)
    end
    denominator_is_one = all(iszero, reduced_denominator_exponent)
    denominator_field = if denominator_is_one
        nothing
    else
        coefficient_field(__qqmpoly_monomial(numerator_ring, reduced_denominator_exponent))
    end
    result_context = MPolyBuildCtx(universal_ring)
    for (action_exponent, numerator_context) in numerator_contexts
        numerator_coefficient = finish(numerator_context)
        field_coefficient = coefficient_field(numerator_coefficient)
        if !denominator_is_one
            field_coefficient = field_coefficient / denominator_field
        end
        iszero(field_coefficient) && continue
        push_term!(result_context, field_coefficient, collect(action_exponent))
    end
    return R(finish(result_context))
end

# Flatten monomial-fraction coefficients into a common raw polynomial representation.
function __action_flatten_monomial_fractions(
    p::P,
    temporary_ring,
    n_action_variables::Int,
    n_coefficient_variables::Int
) where {P<:ActionPolyRingElem}
    raw_data = __action_monomial_fraction_raw_data(p, n_action_variables, n_coefficient_variables)
    raw_data === nothing && return nothing
    context = MPolyBuildCtx(temporary_ring)
    denominator_exponent = raw_data.denominator_exponent
    for (scalar, numerator_exponent, term_denominator_exponent, action_exponent) in raw_data.raw
        full_exponent = zeros(Int, n_coefficient_variables + n_action_variables)
        @inbounds for i in 1:n_coefficient_variables
            full_exponent[i] = numerator_exponent[i] + denominator_exponent[i] - term_denominator_exponent[i]
        end
        @inbounds for i in 1:n_action_variables
            full_exponent[n_coefficient_variables + i] = action_exponent[i]
        end
        push_term!(context, scalar, full_exponent)
    end
    return (polynomial=finish(context), denominator_exponent=denominator_exponent)
end

# Extract raw numerator/denominator data when an action polynomial is a monomial fraction.
function __action_monomial_fraction_raw_data(
    p::P,
    n_action_variables::Int,
    n_coefficient_variables::Int
) where {P<:ActionPolyRingElem}
    raw = Any[]
    sizehint!(raw, length(p))
    common_denominator_exponent = zeros(Int, n_coefficient_variables)
    polynomial = Oscar.data(p)
    for (coefficient, action_exponent) in zip(
        coefficients(polynomial), Oscar.AbstractAlgebra.exponent_vectors(polynomial)
)
        numerator_coefficient = try
            numerator(coefficient)
        catch
            return nothing
        end
        denominator_coefficient = try
            denominator(coefficient)
        catch
            return nothing
        end
        numerator_data = __padded_exponent_vector(numerator_coefficient, n_coefficient_variables)
        numerator_data === nothing && return nothing
        denominator_data = __padded_exponent_vector(denominator_coefficient, n_coefficient_variables)
        denominator_data === nothing && return nothing
        numerator_scalar = coeff(numerator_data.term, 1)
        denominator_scalar = coeff(denominator_data.term, 1)
        __action_pad_exponent_vector!(action_exponent, n_action_variables)
        @inbounds for i in 1:n_coefficient_variables
            common_denominator_exponent[i] = max(common_denominator_exponent[i], denominator_data.exponent[i])
        end
        push!(
            raw,
            (numerator_scalar / denominator_scalar, numerator_data.exponent, denominator_data.exponent, action_exponent)
)
    end
    return (raw=raw, denominator_exponent=common_denominator_exponent)
end

# Attempt a*b-c*d arithmetic without constructing both full intermediate products.
function __action_try_unit_denominator_mul_submul(a::P, b::P, c::P, d::P) where {P<:ActionPolyRingElem}
    R = __require_same_parent(a, b, c, d)
    universal_ring = base_ring(R)
    coefficient_field = coefficient_ring(universal_ring)
    numerator_one = try
        numerator(one(coefficient_field))
    catch
        return nothing
    end
    numerator_type = typeof(numerator_one)
    n_variables = ngens(universal_ring)
    a_terms = __action_unit_denominator_term_data(a, n_variables, numerator_type)
    a_terms === nothing && return nothing
    b_terms = __action_unit_denominator_term_data(b, n_variables, numerator_type)
    b_terms === nothing && return nothing
    c_terms = __action_unit_denominator_term_data(c, n_variables, numerator_type)
    c_terms === nothing && return nothing
    d_terms = __action_unit_denominator_term_data(d, n_variables, numerator_type)
    d_terms === nothing && return nothing
    accumulator = Dict{Tuple{Vararg{Int}}, numerator_type, }()
    if !isempty(a_terms) && !isempty(b_terms)
        __action_accumulate_unit_denominator_product!(accumulator, a_terms, b_terms, n_variables, 1)
    end
    if !isempty(c_terms) && !isempty(d_terms)
        __action_accumulate_unit_denominator_product!(accumulator, c_terms, d_terms, n_variables, -1)
    end
    context = MPolyBuildCtx(universal_ring)
    for (exponent, numerator_coefficient) in accumulator
        push_term!(context, coefficient_field(numerator_coefficient), collect(exponent))
    end
    return R(finish(context))
end

# Detect the raw-term representation used by the unit-denominator arithmetic fast path.
function __action_unit_denominator_term_data(p::P, n_variables::Int, ::Type{N}) where {P<:ActionPolyRingElem,N}
    term_data = Vector{Tuple{ N, Vector{Int}, } }()
    sizehint!(term_data, length(p))
    raw = Oscar.data(p)
    for (coefficient, exponent) in zip(coefficients(raw), Oscar.AbstractAlgebra.exponent_vectors(raw))
        denominator_coefficient = try
            denominator(coefficient)
        catch
            return nothing
        end
        isone(denominator_coefficient) || return nothing
        numerator_coefficient = numerator(coefficient)
        numerator_coefficient isa N || return nothing
        __action_pad_exponent_vector!(exponent, n_variables)
        push!(term_data, (numerator_coefficient, exponent))
    end
    return term_data
end

# Accumulate a product directly into a raw builder for the unit-denominator fast path.
function __action_accumulate_unit_denominator_product!(
    accumulator,
    left_terms,
    right_terms,
    n_variables::Int,
    sign::Int
)
    for (left_coefficient, left_exponent) in left_terms
        for (right_coefficient, right_exponent) in right_terms
            product_coefficient = left_coefficient * right_coefficient
            sign == -1 && (product_coefficient = -product_coefficient)
            iszero(product_coefficient) && continue
            exponent = ntuple(i -> left_exponent[i] + right_exponent[i], n_variables)
            previous = get(accumulator, exponent, nothing)
            if previous !== nothing
                new_coefficient = previous + product_coefficient
                if iszero(new_coefficient)
                    delete!(accumulator, exponent)
                else
                    accumulator[exponent] = new_coefficient
                end
            else
                accumulator[exponent] = product_coefficient
            end
        end
    end
    return accumulator
end



###############################################################################
##### Compact optional caches and pending filters
###############################################################################

# Copy each distinct polynomial only once inside a detached zero proof.
# Multiple shifts of one equation share its snapshot, never the mutable input.
function __passivity_certificate_snapshot(p::P, reductions::Vector{Tuple{P,Vector{Int}}}) where {P<:ActionPolyRingElem}
    if length(reductions) < 2
        return deepcopy(p), __PassivityCertificate{P}(deepcopy(reductions))
    end
    snapshots = IdDict{P,P}()
    snapshot(f) = get!(() -> deepcopy(f), snapshots, f)
    key = snapshot(p)
    proof = __PassivityCertificate{P}([(snapshot(f), copy(alpha)) for (f, alpha) in reductions])
    return key, proof
end

# Detect changes in optional short-reducer sets without storing their polynomials.
# Ignore order; a hash collision can only skip an optional simplification.
function __differential_constraint_fingerprint(polynomials)
    combined, total = UInt(0), UInt(0)
    for p in polynomials
        h = hash(p)
        combined = xor(combined, h)
        total += h
    end
    return hash((length(polynomials), combined, total))
end

# Skip optional queue reductions when no relevant jet or cancellable variable occurs.
# This inexpensive test ignores Janet masks; actual reductions still check them.
function __differential_pending_may_simplify(p, source_jets, nonzero_variables)
    isempty(source_jets) && isempty(nonzero_variables) && return false
    for v in vars(p; sorted=false)
        var_index(Oscar.data(v)) in nonzero_variables && return true
        jet = __jet_data(v)
        for source in source_jets
            source.symbol == jet.symbol || continue
            all(i -> source.shift[i] <= jet.shift[i], eachindex(jet.shift)) && return true
        end
    end
    return false
end