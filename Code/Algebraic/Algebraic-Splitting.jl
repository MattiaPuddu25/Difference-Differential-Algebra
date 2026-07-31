###############################################################################
#####  Algebraic Splitting
###############################################################################

__algebraic_parent(p) = parent(p)
__algebraic_gcd(p, q) = gcd(p, q)
__algebraic_one(p) = one(p)
__algebraic_zero(p) = zero(p)

@doc raw"""
    algebraic_splitting(
        E::AbstractVector{<:MPolyRingElem},
        I::AbstractVector{<:MPolyRingElem};
        max_steps::Integer=1000,
        clean_output::Bool=false,
        factorization::Bool=true,
    )
    algebraic_splitting(
        E::AbstractVector{<:MPolyRingElem};
        max_steps::Integer=1000,
        clean_output::Bool=false,
        factorization::Bool=true,
    )

Split the finite algebraic system `E` = 0, `I` != 0 into quasi-simple algebraic branches.

When only `E` is supplied, the guard list `I` is taken to be empty.

The algorithm repairs repeated leaders and initials which may vanish, and it discards inconsistent branches.
During branch cleanup, equations and guards are replaced by equivalent reduced forms when lower equations or
    already nonzero guards make this possible.

A completely inconsistent input returns an empty vector of branches.\
The empty branch ([], []) is reserved for systems that are always verified.

# Keyword arguments

- `max_steps`: maximum number of branch-processing steps.  If this bound is exceeded,
    the function throws an error instead of continuing indefinitely.

- `clean_output`: if `true`, the final branches are printed with print\_splitting\_result.
    To avoid repeated outputs, a semicolon should be written before launching the command.
    If `false`, no extra formatted output is printed.
    In both cases, the returned vector of branches is unchanged.

- `factorization`: if `true`, equations are split into disjoint branches defined by their
    nonconstant factors and the necessary complementary guards.
    Guards are replaced by the simultaneous nonvanishing of their nonconstant factors.
    If `false`, the algorithm only applies the ordinary algebraic splitting repairs.

# Examples

```jldoctest
julia> R, (x, c, b, a) = polynomial_ring(QQ, [:x, :c, :b, :a]);
julia> p = a*x^2 + b*x + c;
julia> algebraic_splitting([p])
3-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
 ([x^2*a + x*b + c], [a])
 ([x*b + c, a], [b])
 ([c, a, b], [])
julia> algebraic_splitting([p]; clean_output=true);
Branch 1
  E = QQMPolyRingElem[x^2*a + x*b + c]
  I = QQMPolyRingElem[a]
Branch 2
  E = QQMPolyRingElem[x*b + c, a]
  I = QQMPolyRingElem[b]
Branch 3
  E = QQMPolyRingElem[c, a, b]
  I = QQMPolyRingElem[]


julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
julia> algebraic_splitting([(x^2 - 1)*y])
2-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
 ([x^2 - 1], [y])
 ([y], [])
julia> algebraic_splitting([(x^2 - 1)*y]; factorization=true)
3-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
 ([x + 1], [])
 ([x - 1], [])
 ([y], [x^2 - 1])
```
"""
function algebraic_splitting(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}};
    max_steps::Integer=1000,
    clean_output::Bool=false,
    factorization::Bool=true,
) where T
    E_vec, I_vec = collect(E), collect(I)
    result = __algebraic_splitting_loop(
        E_vec,
        I_vec;
        max_steps=Int(max_steps),
        factorize=factorization,
    )
    final_result = __algebraic_finalize_result(result)
    clean_output && print_splitting_result(final_result)
    return final_result
end

function algebraic_splitting(
    E::AbstractVector{<:MPolyRingElem{T}};
    max_steps::Integer=1000,
    clean_output::Bool=false,
    factorization::Bool=true,
) where T
    E_vec = collect(E)
    return algebraic_splitting(
        E_vec,
        empty(E_vec);
        max_steps,
        clean_output,
        factorization,
    )
end

@doc raw"""
    print_splitting_result(result; io=stdout)

Print a splitting result one branch per block, avoiding the full Julia container type display for long vectors of branches.
The input may come from either algebraic or difference splitting.
Guards are displayed in factored form when this is possible.

# Keyword arguments

- `io`: output stream used for printing.
    The default is `stdout`; pass another stream to redirect the formatted display.
"""
function print_splitting_result(
    result::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}};
    io::IO=stdout,
)
    if isempty(result)
        println(io, "[]")
        return nothing
    end
    for (i, (E, I)) in enumerate(result)
        println(io, "Branch $i")
        print(io, "  E = ")
        __splitting_print_polynomial_vector(io, E)
        println(io)
        print(io, "  I = ")
        __splitting_print_polynomial_vector(io, I; factorized=true)
        println(io)
    end
    return nothing
end

function __splitting_print_polynomial_vector(
    io::IO,
    polys::AbstractVector;
    factorized::Bool=false,
)
    print(io, string(eltype(polys)), "[")
    for (i, p) in enumerate(polys)
        i > 1 && print(io, ", ")
        print(io, factorized ? __splitting_factor_display(p) : string(p))
    end
    print(io, "]")
    return nothing
end

function __splitting_factor_display(
    p::Union{MPolyRingElem,DifferencePolyRingElem},
)
    p_clean = try
        __algebraic_remove_constant_factor(p)
    catch
        p
    end

    factor_data = Tuple{Bool,String,String}[]
    try
        for term in collect(factor(p_clean))
            f = try
                __algebraic_remove_constant_factor(first(term))
            catch
                first(term)
            end
            is_constant(f) && continue
            base = string(f)
            exponent = last(term)
            atomic = __splitting_factor_display_is_atomic(base)
            displayed_base = atomic ? base : "($base)"
            displayed = exponent == 1 ? displayed_base : displayed_base * "^" * string(exponent)
            push!(factor_data, (atomic, base, displayed))
        end
    catch
        return string(p_clean)
    end

    isempty(factor_data) && return string(p_clean)
    length(factor_data) == 1 && return factor_data[1][3]
    sort!(factor_data; by=t -> (t[1] ? 0 : 1, t[2]))
    return join((t[3] for t in factor_data), "*")
end

function __splitting_factor_display_is_atomic(s::AbstractString)
    occursin(" + ", s) && return false
    occursin(" - ", s) && return false
    return true
end

###############################################################################
#####  Internal loop
###############################################################################

function __algebraic_splitting_loop(
    E::Vector{T},
    I::Vector{T};
    max_steps::Integer,
    check_discriminants::Bool=false,
    factorize::Bool=true,
) where {T<:MPolyRingElem}
    first = __algebraic_cleanup(E, I)
    first === nothing && return Tuple{Vector{T},Vector{T}}[]

    queue = Tuple{Vector{T},Vector{T}}[first]
    result = Tuple{Vector{T},Vector{T}}[]
    seen = Set{String}()
    steps = 0
    head = 1

    while head <= length(queue)
        steps += 1
        steps > max_steps && error("The algebraic splitting did not terminate within max_steps=$max_steps")

        E_current, I_current = queue[head]
        head += 1
        # Used to recognize the system under consideration
        key = __algebraic_key(E_current, I_current)
        key in seen && continue
        push!(seen, key)

        # Termination check for the system under consideration
        if __algebraic_target_reached(E_current, I_current; check_discriminants, factorize)
            push!(result, (__algebraic_copy_system(E_current, I_current)...,))
            continue
        end

        # Check conditions in the definition
        next_systems = __algebraic_step(E_current, I_current; check_discriminants, factorize)
        target = if check_discriminants && factorize
            "factorized quasi-simple with regular discriminants"
        elseif check_discriminants
            "quasi-simple with regular discriminants"
        elseif factorize
            "factorized quasi-simple"
        else
            "quasi-simple"
        end
        next_systems === nothing && error("No algebraic splitting step was applicable although the branch is not $target")
        append!(queue, next_systems)
    end

    return result
end

function __algebraic_step(
    E::Vector{T},
    I::Vector{T};
    check_discriminants::Bool=false,
    factorize::Bool=true,
) where {T<:MPolyRingElem}
    simplification = __algebraic_inequation_simplification(E, I)
    if simplification !== nothing
        simplification.kind === :inconsistent && return Tuple{Vector{T},Vector{T}}[]
        return __algebraic_normalized([(E, T[I[j] for j in eachindex(I) if j != simplification.index])])
    end

    if factorize
        factorization = __algebraic_factorization_issue(E, I)
        factorization !== nothing && return __algebraic_factorization_split(E, I, factorization)
    end

    duplicate = __algebraic_leader_issue(E, I)
    if duplicate !== nothing
        duplicate.kind === :equations &&
            return __algebraic_same_leader_equations(E, I, duplicate.first_index, duplicate.second_index)
        duplicate.kind === :inequations &&
            return __algebraic_same_leader_inequations(E, I, duplicate.first_index, duplicate.second_index)
        return __algebraic_same_leader_equation_inequation(E, I, duplicate.first_index, duplicate.second_index)
    end

    initial = __algebraic_initial_issue(E, I)
    if initial !== nothing
        initial.kind === :equation && return __algebraic_initial_equation(E, I, initial.index)
        return __algebraic_initial_inequation(E, I, initial.index)
    end
    if check_discriminants
        discriminant = __algebraic_discriminant_issue(E, I)
        discriminant !== nothing && return __algebraic_discriminant_split(E, I, discriminant)
    end
    return nothing
end

###############################################################################
#####  Branch handling
###############################################################################

function __algebraic_cleanup(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    E_clean = T[]
    I_clean = T[]

    for f in E
        is_zero(f) && continue
        is_constant(f) && return nothing
        push!(E_clean, __algebraic_remove_constant_factor(f))
    end

    for q in I
        is_zero(q) && return nothing
        is_constant(q) && continue
        push!(I_clean, __algebraic_remove_constant_factor(q))
    end

    unique!(E_clean)
    unique!(I_clean)

    reduced = __algebraic_autoreduce_branch(E_clean, I_clean)
    reduced === nothing && return nothing
    E_clean, I_clean = reduced

    unique!(E_clean)
    unique!(I_clean)
    any(f -> f in I_clean, E_clean) && return nothing
    return E_clean, I_clean
end

function __algebraic_autoreduce_branch(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    changed = true
    seen = Set{String}()

    while changed
        key = __algebraic_key(E, I)
        key in seen && break
        push!(seen, key)
        changed = false

        for i in eachindex(E)
            f = E[i]
            E_context = T[E[j] for j in eachindex(E) if j != i]
            E_lower = T[g for g in E_context if !is_constant(g) && poly_leader(g) < poly_leader(f)]
            reduced = __algebraic_reduce_equation(f, E_lower, I)

            if is_zero(reduced)
                deleteat!(E, i)
                changed = true
                break
            elseif is_constant(reduced)
                return nothing
            elseif reduced != f
                E[i] = reduced
                changed = true
                break
            end
        end
        changed && continue

        for i in eachindex(I)
            q = I[i]
            I_context = T[I[j] for j in eachindex(I) if j != i]
            E_context = __algebraic_guard_reduction_equations(E, q)
            reduced = __algebraic_reduce_guard(q, E_context, I_context)

            if is_zero(reduced)
                return nothing
            elseif is_constant(reduced)
                deleteat!(I, i)
                changed = true
                break
            elseif reduced != q
                I[i] = reduced
                changed = true
                break
            end
        end
    end

    return E, I
end

function __algebraic_guard_reduction_equations(E::Vector{T}, q::T) where {T<:MPolyRingElem}
    is_constant(q) && return T[]
    v = poly_leader(q)
    return T[f for f in E if !is_constant(f) && poly_leader(f) <= v]
end

function __algebraic_reduce_equation(
    f::T,
    E_context::Vector{T},
    I_context::Vector{T},
) where {T<:MPolyRingElem}
    reduced = __algebraic_normal_form_mod_equations(f, E_context)
    reduced = __algebraic_remove_constant_factor(reduced)
    (is_zero(reduced) || is_constant(reduced)) && return reduced
    return __algebraic_cancel_forced_nonzero_factors(reduced, E_context, I_context)
end

function __algebraic_reduce_guard(
    q::T,
    E_context::Vector{T},
    I_context::Vector{T},
) where {T<:MPolyRingElem}
    reduced = __algebraic_normal_form_mod_equations(q, E_context)
    reduced = __algebraic_remove_constant_factor(reduced)
    (is_zero(reduced) || is_constant(reduced)) && return reduced
    return __algebraic_cancel_forced_nonzero_factors(reduced, E_context, I_context)
end

function __algebraic_cancel_forced_nonzero_factors(
    p::T,
    E_context::Vector{T},
    I_context::Vector{T},
) where {T<:MPolyRingElem}
    factors = __algebraic_nonconstant_factors(p)
    remaining = T[]
    changed = false

    for f in factors
        if __algebraic_factor_is_forced_nonzero(f, E_context, I_context)
            changed = true
        else
            push!(remaining, f)
        end
    end

    changed || return p
    isempty(remaining) && return __algebraic_one(p)
    return __algebraic_remove_constant_factor(prod(remaining))
end

function __algebraic_factor_is_forced_nonzero(
    p::T,
    E_context::Vector{T},
    I_context::Vector{T},
) where {T<:MPolyRingElem}
    return !__compatible_zero_algebraic_system(p, E_context, I_context)
end

function __algebraic_remove_constant_factor(p::T) where {T<:MPolyRingElem}
    is_zero(p) && return p
    q = p + __algebraic_zero(p)
    c = try
        content(q)
    catch
        return __algebraic_normalize_sign(q)
    end
    try
        if !is_zero(c) && !isone(c)
            q = divexact(q, c)
        end
    catch
        return __algebraic_normalize_sign(q)
    end
    return __algebraic_normalize_sign(q)
end

function __algebraic_normalize_sign(p::T) where {T<:MPolyRingElem}
    is_constant(p) && return p
    lc = try
        leading_coefficient(p)
    catch
        return p
    end
    try
        lc < zero(lc) ? -p : p
    catch
        p
    end
end

function __algebraic_normalized(systems::Vector{Tuple{Vector{T},Vector{T}}}) where {T<:MPolyRingElem}
    out = Tuple{Vector{T},Vector{T}}[]
    for (E, I) in systems
        clean = __algebraic_cleanup(E, I)
        clean === nothing || push!(out, clean)
    end
    return out
end

function __algebraic_finalize_result(
    result::Vector{Tuple{Vector{T},Vector{T}}},
) where {T<:MPolyRingElem}
    if isempty(result)
        println("INCONSISTENT SYSTEM")
    elseif length(result) == 1 && isempty(result[1][1]) && isempty(result[1][2])
        println("SYSTEM ALWAYS VERIFIED")
    end
    return result
end

function __algebraic_copy_system(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    return T[f + parent(f)(0) for f in E], T[q + parent(q)(0) for q in I]
end

function __algebraic_key(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    return "E=" * join(sort(string.(E)), ";") * "|I=" * join(sort(string.(I)), ";")
end

###############################################################################
#####  Quasi-simple checks and algebraic tests
###############################################################################

function __algebraic_target_reached(
    E::Vector{T},
    I::Vector{T};
    check_discriminants::Bool,
    factorize::Bool=true,
) where {T<:MPolyRingElem}
    if isempty(I)
        is_quasi_simple(E) || return false
    elseif isempty(E)
        is_quasi_simple_ineq(I) || return false
    else
        is_quasi_simple(E, I) || return false
    end
    factorize && __algebraic_factorization_issue(E, I) !== nothing && return false
    check_discriminants || return true
    return __algebraic_discriminant_issue(E, I) === nothing
end

function __algebraic_vanishes_on_system(p::T, E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    is_zero(p) && return true
    is_constant(p) && return false
    R = __algebraic_parent(p)
    J = ideal(R, E)
    is_one(J) && return true
    K = J
    if !isempty(I)
        Q = prod(I)
        is_zero(Q) && return true
        K = saturation(J, ideal(R, [Q]))
        is_one(K) && return true
    end
    return radical_membership(p, K)
end

function __algebraic_inequation_simplification(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    for i in eachindex(I)
        q = I[i]
        I_without_q = T[I[j] for j in eachindex(I) if j != i]
        E_partial, I_partial = __algebraic_initial_partial_system(E, I_without_q, q)
        !__compatible_zero_algebraic_system(q, E_partial, I_partial) &&
            return (kind=:remove_inequation, index=i)
        __algebraic_vanishes_on_system(q, E_partial, I_partial) &&
            return (kind=:inconsistent, index=i)
    end
    return nothing
end

function __algebraic_factorization_issue(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    for i in eachindex(E)
        factors = __algebraic_nonconstant_factors(E[i])
        __algebraic_nontrivial_factorization(E[i], factors) &&
            return (kind=:equation, index=i, factors=factors)
    end

    for i in eachindex(I)
        factors = __algebraic_nonconstant_factors(I[i])
        grouped_factors = __algebraic_group_factors_by_leader(factors)
        __algebraic_nontrivial_factorization(I[i], grouped_factors) &&
            return (kind=:inequation, index=i, factors=grouped_factors)
    end

    return nothing
end

function __algebraic_nonconstant_factors(p::T) where {T<:MPolyRingElem}
    p = __algebraic_remove_constant_factor(p)
    factors = T[]
    try
        for factor_pair in collect(factor(p))
            f = __algebraic_remove_constant_factor(first(factor_pair))
            is_constant(f) || push!(factors, f)
        end
    catch
        return T[p]
    end
    unique!(factors)
    return isempty(factors) ? T[p] : factors
end

function __algebraic_nontrivial_factorization(p::T, factors::Vector{T}) where {T<:MPolyRingElem}
    p = __algebraic_remove_constant_factor(p)
    return length(factors) > 1 || (length(factors) == 1 && first(factors) != p)
end

function __algebraic_group_factors_by_leader(factors::Vector{T}) where {T<:MPolyRingElem}
    grouped = T[]
    leaders = T[]
    for f in factors
        leader = poly_leader(f)
        index = findfirst(==(leader), leaders)
        if index === nothing
            push!(leaders, leader)
            push!(grouped, f)
        else
            grouped[index] = __algebraic_remove_constant_factor(grouped[index] * f)
        end
    end
    return grouped
end

###############################################################################
#####  Leaders and initials
###############################################################################

# Find leader duplicates
function __algebraic_leader_issue(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    E_leaders = [poly_leader(f) for f in E]
    I_leaders = [poly_leader(q) for q in I]
    for i in eachindex(E), j in (i + 1):length(E)
        E_leaders[i] == E_leaders[j] && return (kind=:equations, first_index=i, second_index=j)
    end
    for i in eachindex(I), j in (i + 1):length(I)
        I_leaders[i] == I_leaders[j] && return (kind=:inequations, first_index=i, second_index=j)
    end
    for i in eachindex(E), j in eachindex(I)
        E_leaders[i] == I_leaders[j] && return (kind=:equation_inequation, first_index=i, second_index=j)
    end
    return nothing
end

# Find initial condition failure
function __algebraic_initial_issue(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    for i in eachindex(E)
        h = poly_initial(E[i])
        E_partial, I_partial = __algebraic_initial_partial_system(E, I, h)
        __compatible_zero_algebraic_system(h, E_partial, I_partial) && return (kind=:equation, index=i)
    end
    for i in eachindex(I)
        h = poly_initial(I[i])
        E_partial, I_partial = __algebraic_initial_partial_system(E, I, h)
        __compatible_zero_algebraic_system(h, E_partial, I_partial) && return (kind=:inequation, index=i)
    end
    return nothing
end

function __algebraic_initial_partial_system(E::Vector{T}, I::Vector{T}, h::T) where {T<:MPolyRingElem}
    is_constant(h) && return T[], T[]
    v = poly_leader(h)
    return T[f for f in E if !is_constant(f) && poly_leader(f) <= v],
           T[q for q in I if !is_constant(q) && poly_leader(q) <= v]
end

function __algebraic_squarefree_part(p::T) where {T<:MPolyRingElem}
    is_constant(p) && return p
    v = poly_leader(p)
    idx = var_index(v)
    dp = derivative(p, idx)
    is_zero(dp) && return p
    g = __algebraic_gcd(p, dp)
    is_constant(g) && return p
    division = __checked_divides(g, p)
    division.divides && division.quotient !== nothing && return division.quotient
    return divexact(p, g)
end

function __algebraic_lower_equations(E::Vector{T}, v::T) where {T<:MPolyRingElem}
    return T[f for f in E if !is_constant(f) && poly_leader(f) < v]
end

function __algebraic_normal_form_mod_equations(p::T, E_lower::Vector{T}) where {T<:MPolyRingElem}
    isempty(E_lower) && return p
    return Oscar.normal_form(p, ideal(__algebraic_parent(p), E_lower))
end

function __algebraic_tail(p::T) where {T<:MPolyRingElem}
    is_constant(p) && return __algebraic_zero(p)
    v = poly_leader(p)
    return p - poly_initial(p) * v^degree(p, var_index(v))
end

###############################################################################
#####  Splitting steps
###############################################################################

function __algebraic_factorization_split(
    E::Vector{T},
    I::Vector{T},
    issue::NamedTuple,
) where {T<:MPolyRingElem}
    i = Int(issue.index)
    factors = collect(T, issue.factors)

    if issue.kind === :equation
        factors = __algebraic_order_factors_for_disjoint_split(factors)
        systems = Tuple{Vector{T},Vector{T}}[]
        previous_factors = T[]
        for f in factors
            E_new = copy(E)
            E_new[i] = f
            push!(systems, (E_new, [I; previous_factors]))
            push!(previous_factors, f)
        end
        return __algebraic_normalized(systems)
    elseif issue.kind === :inequation
        I_new = T[I[j] for j in eachindex(I) if j != i]
        append!(I_new, factors)
        return __algebraic_normalized([(E, I_new)])
    end

    error("Unknown factorization issue kind: $(issue.kind)")
end

function __algebraic_order_factors_for_disjoint_split(factors::Vector{T}) where {T<:MPolyRingElem}
    ordered = copy(factors)
    sort!(
        ordered;
        by = f -> begin
            v = poly_leader(f)
            idx = var_index(v)
            (idx, -degree(f, idx), string(f))
        end,
    )
    return ordered
end

function __algebraic_same_leader_equations(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    i, j = Int(i), Int(j)
    f, g = E[i], E[j]
    v = poly_leader(f)
    reduce_index, divisor_index = i, j
    if degree(f, var_index(v)) < degree(g, var_index(v))
        reduce_index, divisor_index = j, i
        f, g = g, f
    end
    h = poly_initial(g)
    _, r = __algebraic_pseudo_divrem(f, g)

    E_nonzero = copy(E)
    E_nonzero[reduce_index] = r
    E_zero = copy(E)
    E_zero[divisor_index] = __algebraic_tail(g)
    push!(E_zero, h)
    return __algebraic_normalized([(E_nonzero, [I; h]), (E_zero, I)])
end

function __algebraic_same_leader_inequations(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    i, j = Int(i), Int(j)
    q = I[i] * I[j]
    I_new = T[]
    for k in eachindex(I)
        k == i ? push!(I_new, q) : (k != j && push!(I_new, I[k]))
    end
    return __algebraic_normalized([(E, I_new)])
end

function __algebraic_same_leader_equation_inequation(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    i, j = Int(i), Int(j)
    f, q = E[i], I[j]
    v = poly_leader(f)
    E_lower = __algebraic_lower_equations(E, v)
    f_reduced = __algebraic_normal_form_mod_equations(f, E_lower)
    q_reduced = __algebraic_normal_form_mod_equations(q, E_lower)
    if f_reduced != f || q_reduced != q
        E_reduced = copy(E)
        I_reduced = copy(I)
        E_reduced[i] = f_reduced
        I_reduced[j] = q_reduced
        return __algebraic_normalized([(E_reduced, I_reduced)])
    end

    I_without_q = T[I[k] for k in eachindex(I) if k != j]

    __algebraic_vanishes_on_system(q, E, I_without_q) && return Tuple{Vector{T},Vector{T}}[]
    !__compatible_zero_algebraic_system(q, E, I_without_q) && return __algebraic_normalized([(E, I_without_q)])

    if degree(q, var_index(v)) >= degree(f, var_index(v))
        h = poly_initial(f)
        _, r = __algebraic_pseudo_divrem(q, f)
        I_nonzero = copy(I)
        I_nonzero[j] = r
        E_zero = copy(E)
        E_zero[i] = __algebraic_tail(f)
        push!(E_zero, h)
        return __algebraic_normalized([(E, [I_nonzero; h]), (E_zero, I)])
    end

    hq = poly_initial(q)
    quotient, remainder = __algebraic_pseudo_divrem(f, q)
    if __algebraic_vanishes_on_system(remainder, T[E[k] for k in eachindex(E) if k != i], [I; hq])
        E_nonzero = copy(E)
        E_nonzero[i] = quotient
        E_zero = [E; hq]
        I_zero = copy(I)
        I_zero[j] = __algebraic_tail(q)
        return __algebraic_normalized([(E_nonzero, [I; hq]), (E_zero, I_zero)])
    end

    division = __checked_divides(q, f)
    if division.divides
        E_new = copy(E)
        E_new[i] = division.quotient
        return __algebraic_normalized([(E_new, I)])
    end

    common = __algebraic_common_factor_equation_inequation(E, I, i, j)
    common !== nothing && return common
    res = __algebraic_resultant_equation_inequation(E, I, i, j)
    res !== nothing && return res
    return __algebraic_power_remainder_equation_inequation(E, I, i, j)
end

function __algebraic_initial_equation(E::Vector{T}, I::Vector{T}, i::Integer) where {T<:MPolyRingElem}
    i = Int(i)
    p = E[i]
    h = poly_initial(p)
    E_zero = copy(E)
    E_zero[i] = __algebraic_tail(p)
    push!(E_zero, h)
    return __algebraic_normalized([(E, [I; h]), (E_zero, I)])
end

function __algebraic_initial_inequation(E::Vector{T}, I::Vector{T}, i::Integer) where {T<:MPolyRingElem}
    i = Int(i)
    q = I[i]
    h = poly_initial(q)
    E_zero = [E; h]
    I_zero = copy(I)
    I_zero[i] = __algebraic_tail(q)
    return __algebraic_normalized([(E, [I; h]), (E_zero, I_zero)])
end

###############################################################################
#####  Algebraic operations
###############################################################################

function __algebraic_pseudo_divrem(p::T, g::T) where {T<:MPolyRingElem}
    R = __algebraic_parent(p)
    is_constant(p) && return R(0), p
    v = poly_leader(g)
    idx = var_index(v)
    dg = degree(g, idx)
    hg = poly_initial(g)
    q = R(0)
    r = p
    while !is_constant(r) && poly_leader(r) == v && degree(r, idx) >= dg
        dr = degree(r, idx)
        hr = poly_initial(r)
        t = hr * v^(dr - dg)
        q = hg * q + t
        r = hg * r - t * g
    end
    return q, r
end

function __algebraic_pseudo_divrem_with_multiplier(p::T, g::T) where {T<:MPolyRingElem}
    R = __algebraic_parent(p)
    is_constant(p) && return (quotient=R(0), remainder=p, multiplier=R(1))
    v = poly_leader(g)
    idx = var_index(v)
    dg = degree(g, idx)
    hg = poly_initial(g)
    q = R(0)
    r = p
    c = R(1)
    while !is_constant(r) && poly_leader(r) == v && degree(r, idx) >= dg
        dr = degree(r, idx)
        hr = poly_initial(r)
        t = hr * v^(dr - dg)
        q = hg * q + t
        r = hg * r - t * g
        c *= hg
    end
    return (quotient=q, remainder=r, multiplier=c)
end

function __algebraic_common_factor_equation_inequation(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    f, q = E[Int(i)], I[Int(j)]
    d = try
        gcd(f, q)
    catch
        return nothing
    end
    is_constant(d) && return nothing
    f_division = __checked_divides(d, f)
    q_division = __checked_divides(d, q)
    (!f_division.divides || !q_division.divides ||
     f_division.quotient === nothing || q_division.quotient === nothing) && return nothing
    E_new = copy(E)
    E_new[Int(i)] = f_division.quotient
    I_new = T[I[k] for k in eachindex(I) if k != Int(j)]
    push!(I_new, d)
    push!(I_new, q_division.quotient)
    return __algebraic_normalized([(E_new, I_new)])
end

function __algebraic_resultant_equation_inequation(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    f, q = E[Int(i)], I[Int(j)]
    v = poly_leader(f)
    I_without_q = T[I[k] for k in eachindex(I) if k != Int(j)]
    res = try
        resultant(f, q, var_index(v))
    catch
        return nothing
    end
    is_zero(res) && return nothing
    res = __algebraic_squarefree_part(res)
    is_constant(res) && return __algebraic_normalized([(E, I_without_q)])
    __algebraic_vanishes_on_system(res, E, I_without_q) && return nothing
    !__compatible_zero_algebraic_system(res, E, I_without_q) && return __algebraic_normalized([(E, I_without_q)])
    return __algebraic_normalized([(E, [I_without_q; res]), ([E; res], I)])
end

function __algebraic_power_remainder_equation_inequation(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    f, q = E[Int(i)], I[Int(j)]
    v = poly_leader(f)
    idx = var_index(v)
    dq = degree(q, idx)
    dq <= 0 && return nothing
    exponent = max(1, cld(degree(f, idx), dq))
    division = __algebraic_pseudo_divrem_with_multiplier(q^exponent, f)
    I_without_q = T[I[k] for k in eachindex(I) if k != Int(j)]
    systems = Tuple{Vector{T},Vector{T}}[]
    if !is_constant(division.multiplier)
        is_zero(division.remainder) ||
            push!(systems, (E, [I_without_q; division.multiplier; division.remainder]))
        push!(systems, ([E; division.multiplier], I))
    elseif !is_zero(division.remainder)
        push!(systems, (E, [I_without_q; division.remainder]))
    end
    return __algebraic_normalized(systems)
end

###############################################################################
#####  Branch Intersections and Unions
###############################################################################

###############################################################################
#####  Branch intersections
###############################################################################

@doc raw"""
    branch_intersection(branch_1, branch_2)

Return `true` if two algebraic or difference branches have a common solution,
and `false` if they are disjoint.

Each branch is a pair `(E, I)` and represents the system `E = 0`, `I != 0`.\
For algebraic branches the check uses the criterion

  V(E\_1 ∪ E_2) ∩ D(\prod(I_1 ∪ I_2)) ≠ ∅

over an algebraic closure, computed by saturation.

For difference branches, the finite set of occurring difference variables is first converted to an ordinary
polynomial ring and the same algebraic check is applied.

# Examples

```jldoctest
julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
julia> branch_intersection(([x], empty([x])), ([x, y], empty([x])))
true
julia> branch_intersection(([x], [y]), ([x, y], empty([x])))
false


julia> branches = algebraic_splitting([(x^2 - 1)*y]; factorization=true);
julia> branch_intersection(branches[1], branches[2])
false
julia> branch_intersection(branches[1], branches[3])
false
```
"""
function branch_intersection(
    branch_1::Tuple{<:AbstractVector{<:MPolyRingElem{T}},<:AbstractVector{<:MPolyRingElem{T}}},
    branch_2::Tuple{<:AbstractVector{<:MPolyRingElem{T}},<:AbstractVector{<:MPolyRingElem{T}}},
) where T
    E1, I1 = branch_1
    E2, I2 = branch_2

    # A common solution must satisfy both equation sets and every guard from
    # both branches; the helper tests this locally closed set by saturation.
    return __algebraic_branch_intersection(E1, I1, E2, I2)
end

function branch_intersection(
    branch_1::Tuple{<:AbstractVector{<:DifferencePolyRingElem{T}},<:AbstractVector{<:DifferencePolyRingElem{T}}},
    branch_2::Tuple{<:AbstractVector{<:DifferencePolyRingElem{T}},<:AbstractVector{<:DifferencePolyRingElem{T}}},
) where T
    E1, I1 = branch_1
    E2, I2 = branch_2

    # Convert all four lists together, so equal transforms in the two branches
    # become equal variables in the same ordinary polynomial ring.
    system = [collect(E1); collect(I1); collect(E2); collect(I2)]

    # Two empty branches both represent the whole space.
    isempty(system) && return true

    converted, _, _ = conversion_difference_algebraic(system, system; export_variables=false)

    # Split the converted vector back into the equations and guards of each branch.
    nE1 = length(E1)
    nI1 = length(I1)
    nE2 = length(E2)
    E1_alg = converted[1:nE1]
    I1_alg = converted[(nE1 + 1):(nE1 + nI1)]
    E2_alg = converted[(nE1 + nI1 + 1):(nE1 + nI1 + nE2)]
    I2_alg = converted[(nE1 + nI1 + nE2 + 1):end]

    # Apply the algebraic intersection criterion to the finite conversion.
    return branch_intersection((E1_alg, I1_alg), (E2_alg, I2_alg))
end

function __algebraic_branch_intersection(E1, I1, E2, I2)
    all_inputs = [collect(E1); collect(I1); collect(E2); collect(I2)]
    isempty(all_inputs) && return true

    first_polynomial = first(all_inputs)
    __splitting_check_input_polynomial(nothing, first_polynomial)
    for p in Iterators.drop(all_inputs, 1)
        __splitting_check_input_polynomial(first_polynomial, p)
    end
    P = typeof(first_polynomial)
    E = P[collect(E1); collect(E2)]
    I = P[collect(I1); collect(I2)]

    # Simplify the combined locally closed system and reject immediate
    # contradictions such as a zero guard or a nonzero constant equation.
    clean = __algebraic_cleanup(E, I)
    clean === nothing && return false

    E_clean, I_clean = clean
    all_polys = [E_clean; I_clean]
    isempty(all_polys) && return true

    # The common zero set of the combined equations is empty exactly when its
    # ideal is the unit ideal.
    R = parent(first(all_polys))
    J = ideal(R, E_clean)
    is_one(J) && return false
    isempty(I_clean) && return true

    # Requiring every guard to be nonzero is equivalent, over a field, to
    # requiring their product h to be nonzero.
    h = R(1)
    for q in I_clean
        h *= q
    end
    is_zero(h) && return false

    # Saturating J by h removes components contained in h = 0. A nonunit
    # saturated ideal therefore means that a common point with h ≠ 0 remains.
    K = saturation(J, ideal(R, [h]))
    return !is_one(K)
end

###############################################################################
#####  Branch unions and coverage
###############################################################################

@doc raw"""
    splitting_covers_system(
        E::AbstractVector{<:MPolyRingElem},
        I::AbstractVector{<:MPolyRingElem},
        branches;
        return_remainder::Bool=false,
        max_branches::Integer=10000,
    )
    splitting_covers_system(
        E::AbstractVector{<:MPolyRingElem},
        branches;
        return_remainder::Bool=false,
        max_branches::Integer=10000,
    )

Check whether the locally closed algebraic system `E` = 0, `I` != 0 is exactly the union of
    the locally closed systems in `branches`.

The number of branches is arbitrary.
The test subtracts the branches one at a time from the initial system and also checks that no branch contributes
    points outside the initial system.
Empty constructible pieces are detected by saturation.

The returned value is a boolean.
If `return_remainder` is `true`, the function returns a named tuple `(covers, missing, extra)`,
    where `missing` is the part of the initial system not covered by the branches and `extra` is the part of the branches outside the initial system.

# Keyword arguments

- `return_remainder`: if `true`, return the diagnostic named tuple `(covers, missing, extra)`
instead of only the boolean value.

- `max_branches`: maximum number of intermediate constructible pieces retained during the subtraction process.
If this bound is exceeded, the function throws an error.

# Examples

```jldoctest
julia> R, (x, y) = polynomial_ring(QQ, [:x, :y])
julia> branches = algebraic_splitting([(x^2 - 1)*y]; factorization=true)
julia> splitting_covers_system([(x^2 - 1)*y], branches)
true
julia> splitting_covers_system([(x^2 - 1)*y], branches[1:2])
false
julia> splitting_covers_system([(x^2 - 1)*y], branches[1:2]; return_remainder=true).missing
1-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
 ([y], [x + 1, x - 1])


julia> branch_with_extra_points = [([x], empty([x])), ([y], empty([x]))]
julia> splitting_covers_system([x], branch_with_extra_points)
false
```
"""
function splitting_covers_system(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}},
    branches::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}};
    return_remainder::Bool=false,
    max_branches::Integer=10000,
) where T
    E_vec = collect(E)
    I_vec = collect(I)

    # Collect the input system and every branch into vectors of elements of the same polynomial ring.
    P = __splitting_input_element_type(E_vec, I_vec, branches)

    # Check that the initial system is covered by the branches, and no branch adds new points.
    return __algebraic_splitting_covers_system(
        P[p for p in E_vec],
        P[p for p in I_vec],
        __splitting_collect_branch_vector(P, branches);
        return_remainder,
        max_branches=Int(max_branches),
    )
end

function splitting_covers_system(
    E::AbstractVector{<:MPolyRingElem{T}},
    branches::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}};
    return_remainder::Bool=false,
    max_branches::Integer=10000,
) where T
    E_vec = collect(E)
    return splitting_covers_system(
        E_vec,
        empty(E_vec),
        branches;
        return_remainder,
        max_branches,
    )
end

function splitting_covers_system(
    E::AbstractVector{<:DifferencePolyRingElem{T}},
    I::AbstractVector{<:DifferencePolyRingElem{T}},
    branches::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}};
    return_remainder::Bool=false,
    max_branches::Integer=10000,
) where T
    raw_E = collect(E)
    raw_I = collect(I)

    # Collect all inputs into vectors of elements of the same difference polynomial ring.
    P = __splitting_input_element_type(raw_E, raw_I, branches)
    E_vec = P[p for p in raw_E]
    I_vec = P[p for p in raw_I]
    branch_vec = __splitting_collect_branch_vector(P, branches)

    # Every occurring transform is represented by the same variable in one algebraic ring.
    system = [E_vec; I_vec]
    for (F, G) in branch_vec
        append!(system, F)
        append!(system, G)
    end

    # With no polynomial at all, the initial system is the whole space. It is
    # covered exactly when the branch list contains the whole-space branch.
    if isempty(system)
        covers = any(branch -> isempty(branch[1]) && isempty(branch[2]), branch_vec)
        result = (covers=covers, missing=covers ? [] : [(E_vec, I_vec)], extra=[])
        return return_remainder ? result : covers
    end

    # Perform the comparison in the finite polynomial ring of the occurring transforms.
    converted, _, _ = conversion_difference_algebraic(system, system; export_variables=false)

    # Recover the converted input system from the converted vector.
    cursor = 1
    E_alg = converted[cursor:(cursor + length(E_vec) - 1)]
    cursor += length(E_vec)
    I_alg = converted[cursor:(cursor + length(I_vec) - 1)]
    cursor += length(I_vec)

    # Recover each converted branch, preserving its equation/guard boundary.
    branches_alg = Tuple{Vector{eltype(converted)},Vector{eltype(converted)}}[]
    for (F, G) in branch_vec
        F_alg = converted[cursor:(cursor + length(F) - 1)]
        cursor += length(F)
        G_alg = converted[cursor:(cursor + length(G) - 1)]
        cursor += length(G)
        push!(branches_alg, (F_alg, G_alg))
    end

    # Reuse the algebraic coverage test after the common finite conversion.
    return splitting_covers_system(
        E_alg,
        I_alg,
        branches_alg;
        return_remainder,
        max_branches,
    )
end

function splitting_covers_system(
    E::AbstractVector{<:DifferencePolyRingElem{T}},
    branches::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}};
    return_remainder::Bool=false,
    max_branches::Integer=10000,
) where T
    E_vec = collect(E)
    return splitting_covers_system(
        E_vec,
        empty(E_vec),
        branches;
        return_remainder,
        max_branches,
    )
end

function __splitting_input_element_type(
    E::AbstractVector{<:Union{MPolyRingElem,DifferencePolyRingElem}},
    I::AbstractVector{<:Union{MPolyRingElem,DifferencePolyRingElem}},
    branches::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}},
)
    first_polynomial = nothing
    for polys in (E, I)
        for p in polys
            first_polynomial = __splitting_check_input_polynomial(first_polynomial, p)
        end
    end
    for (F, G) in branches
        for polys in (F, G)
            for p in polys
                first_polynomial = __splitting_check_input_polynomial(first_polynomial, p)
            end
        end
    end
    first_polynomial === nothing || return typeof(first_polynomial)

    P = eltype(E)
    P <: Union{MPolyRingElem,DifferencePolyRingElem} && return P
    P = eltype(I)
    P <: Union{MPolyRingElem,DifferencePolyRingElem} && return P
    error("At least one input vector must contain a polynomial")
end

function __splitting_check_input_polynomial(first_polynomial, p)
    p isa Union{MPolyRingElem,DifferencePolyRingElem} ||
        error("Every equation and guard must be a polynomial")
    first_polynomial === nothing && return p
    first_polynomial isa Union{MPolyRingElem,DifferencePolyRingElem} ||
        error("Every equation and guard must be a polynomial")
    typeof(p) === typeof(first_polynomial) ||
        error("All equations and guards must have the same polynomial element type")
    __algebraic_parent(p) === __algebraic_parent(first_polynomial) ||
        error("All equations and guards must belong to the same polynomial ring")
    return first_polynomial
end

function __splitting_collect_branch_vector(
    P::Type{T},
    branches::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}},
) where {T<:Union{MPolyRingElem,DifferencePolyRingElem}}
    return Tuple{Vector{T},Vector{T}}[
        (T[p for p in F], T[q for q in G])
        for (F, G) in branches
    ]
end

function __algebraic_splitting_covers_system(
    E::Vector{T},
    I::Vector{T},
    branches::Vector{Tuple{Vector{T},Vector{T}}};
    return_remainder::Bool,
    max_branches::Integer,
) where {T<:MPolyRingElem}
    max_branches < 0 &&
        error("max_branches must be nonnegative")

    # Normalize the input and discard constructible pieces proved empty by saturation.
    initial = __algebraic_normalized_nonempty([(E, I)])
    branches = __algebraic_normalized_nonempty(branches)

    # What remains after removing every output branch consists of lost solutions.
    missing = __algebraic_subtract_branch_list(
        initial,
        branches;
        max_branches,
    )

    # Conversely, subtract the initial system from every branch to detect
    # solutions introduced by the purported splitting.
    extra = Tuple{Vector{T},Vector{T}}[]
    for branch in branches
        append!(extra, __algebraic_subtract_branch_list(
            Tuple{Vector{T},Vector{T}}[branch],
            initial;
            max_branches,
        ))
        extra = __algebraic_deduplicate_branch_list(extra)
        __algebraic_check_remainder_bound(extra, max_branches)
    end

    # Equality holds precisely when both relative complements are empty.
    covers = isempty(missing) && isempty(extra)
    result = (covers=covers, missing=missing, extra=extra)
    return return_remainder ? result : covers
end

function __algebraic_subtract_branch_list(
    sources::Vector{Tuple{Vector{T},Vector{T}}},
    subtractors::Vector{Tuple{Vector{T},Vector{T}}};
    max_branches::Integer,
) where {T<:MPolyRingElem}
    current = __algebraic_normalized_nonempty(sources)
    __algebraic_check_remainder_bound(current, max_branches)

    for subtractor in subtractors
        next_current = Tuple{Vector{T},Vector{T}}[]
        for source in current
            append!(next_current, __algebraic_subtract_branch(source, subtractor))
            __algebraic_check_remainder_bound(next_current, max_branches)
        end
        current = __algebraic_deduplicate_branch_list(next_current)
        __algebraic_check_remainder_bound(current, max_branches)
        isempty(current) && break
    end
    return current
end

function __algebraic_subtract_branch(
    source::Tuple{Vector{T},Vector{T}},
    subtractor::Tuple{Vector{T},Vector{T}},
) where {T<:MPolyRingElem}
    current = __algebraic_normalized_nonempty(Tuple{Vector{T},Vector{T}}[source])
    isempty(current) && return Tuple{Vector{T},Vector{T}}[]

    F, G = subtractor
    outside = Tuple{Vector{T},Vector{T}}[]

    for f in F
        next_current = Tuple{Vector{T},Vector{T}}[]
        for (E, I) in current
            append!(outside, __algebraic_normalized_nonempty([(E, [I; f])]))
            append!(next_current, __algebraic_normalized_nonempty([([E; f], I)]))
        end
        current = __algebraic_deduplicate_branch_list(next_current)
        isempty(current) && return __algebraic_deduplicate_branch_list(outside)
    end

    for q in G
        next_current = Tuple{Vector{T},Vector{T}}[]
        for (E, I) in current
            append!(outside, __algebraic_normalized_nonempty([([E; q], I)]))
            append!(next_current, __algebraic_normalized_nonempty([(E, [I; q])]))
        end
        current = __algebraic_deduplicate_branch_list(next_current)
        isempty(current) && break
    end

    return __algebraic_deduplicate_branch_list(outside)
end

function __algebraic_normalized_nonempty(
    systems::Vector{Tuple{Vector{T},Vector{T}}},
) where {T<:MPolyRingElem}
    out = Tuple{Vector{T},Vector{T}}[]
    for branch in __algebraic_normalized(systems)
        __algebraic_branch_has_solution(branch...) && push!(out, branch)
    end
    return __algebraic_deduplicate_branch_list(out)
end

function __algebraic_branch_has_solution(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    isempty(E) && isempty(I) && return true
    return __algebraic_branch_intersection(E, I, T[], T[])
end

function __algebraic_deduplicate_branch_list(
    branches::Vector{Tuple{Vector{T},Vector{T}}},
) where {T<:MPolyRingElem}
    out = Tuple{Vector{T},Vector{T}}[]
    seen = Set{String}()
    for (E, I) in branches
        key = __algebraic_key(E, I)
        key in seen && continue
        push!(seen, key)
        push!(out, (E, I))
    end
    return out
end

function __algebraic_check_remainder_bound(
    branches::Vector{Tuple{Vector{T},Vector{T}}},
    max_branches::Integer,
) where {T<:MPolyRingElem}
    length(branches) <= max_branches && return nothing
    error("The coverage check produced more than $max_branches intermediate constructible pieces")
end

###############################################################################
#####  Discriminant Refinement
###############################################################################

@doc raw"""
    algebraic_splitting_discriminant(
        E::AbstractVector{<:MPolyRingElem},
        I::AbstractVector{<:MPolyRingElem};
        max_steps::Integer=1000,
        clean_output::Bool=false,
        factorization::Bool=true,
    )
    algebraic_splitting_discriminant(
        E::AbstractVector{<:MPolyRingElem};
        max_steps::Integer=1000,
        clean_output::Bool=false,
        factorization::Bool=true,
    )

Split the finite algebraic system `E` = 0, `I` != 0 into branches which are quasi-simple and satisfy the additional discriminant condition.

This is an optional refinement, kept separate from the main algebraic splitting workflow.
For every polynomial `r` in `E` or in `I`, let v = ld(r), d = deg\_v(r), and

disc(r) = (-1)^(d(d-1)/2) * res(r, dr/dv, v) / init(r).

The extra condition requires disc(r) = 0 to have no solution in the partial system made of the equations and guards whose leaders are
at most ld(disc(r)).

When only `E` is supplied, the guard list `I` is taken to be empty.

# Keyword arguments

- `max_steps`: maximum number of branch-processing steps.  If this bound is exceeded,
    the function throws an error instead of continuing indefinitely.

- `clean_output`: if `true`, the final branches are printed with print\_splitting\_result.
    To avoid repeated outputs, a semicolon should be written before launching the command.
    If `false`, no extra formatted output is printed. In both cases, the returned vector of branches is unchanged.

- `factorization`: if `true`, equations are split into disjoint branches defined by their nonconstant factors
    and the necessary complementary guards.  Guards are replaced by the simultaneous nonvanishing
    of their nonconstant factors.
    If `false`, only the discriminant refinement and the ordinary algebraic splitting repairs are used.

# Examples

```jldoctest
julia> R, (x, a) = polynomial_ring(QQ, [:x, :a]);
julia> algebraic_splitting_discriminant([x^2 - a])
2-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
 ([x^2 - a], [a])
 ([x, a], [])

julia> R, (x, c, b, a) = polynomial_ring(QQ, [:x, :c, :b, :a]);
julia> p = a*x^2 + b*x + c;
julia> algebraic_splitting_discriminant([p])
4-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
 ([x^2*a + x*b + c], [a, -4*c*a + b^2])
 ([2*x*a + b, -4*c*a + b^2], [a])
 ([x*b + c, a], [b])
 ([c, a, b], [])


julia> R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z]);
julia> steiner = x^2*y^2 + x^2*z^2 + y^2*z^2 - x*y*z;
julia> algebraic_splitting_discriminant([steiner]; factorization=true, clean_output=true);
Branch 1
  E = QQMPolyRingElem[x^2*y^2 + x^2*z^2 - x*y*z + y^2*z^2]
  I = QQMPolyRingElem[y*(4*y^2 + 4*z^2 - 1)*(y^2 + z^2), z*(2*z + 1)*(2*z - 1)]
Branch 2
  E = QQMPolyRingElem[-x + 2*y*z, 4*y^2 + 4*z^2 - 1]
  I = QQMPolyRingElem[(2*z + 1)*(2*z - 1)]
Branch 3
  E = QQMPolyRingElem[x*y + z^3, y^2 + z^2]
  I = QQMPolyRingElem[z]
Branch 4
  E = QQMPolyRingElem[4*x^2*y^2 + x^2 + 2*x*y + y^2, 2*z + 1]
  I = QQMPolyRingElem[y*(4*y^2 + 1)]
Branch 5
  E = QQMPolyRingElem[4*x^2*y^2 + x^2 - 2*x*y + y^2, 2*z - 1]
  I = QQMPolyRingElem[y*(4*y^2 + 1)]
Branch 6
  E = QQMPolyRingElem[x, z]
  I = QQMPolyRingElem[y*(2*y + 1)*(2*y - 1)]
Branch 7
  E = QQMPolyRingElem[x, y, 2*z + 1]
  I = QQMPolyRingElem[]
Branch 8
  E = QQMPolyRingElem[x, y, 2*z - 1]
  I = QQMPolyRingElem[]
Branch 9
  E = QQMPolyRingElem[x, y]
  I = QQMPolyRingElem[z*(2*z + 1)*(2*z - 1)]
Branch 10
  E = QQMPolyRingElem[z, y]
  I = QQMPolyRingElem[]


julia> R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z])
julia> E = [x*(x-1)-y^2+z]
julia> I = [x*y*z, x^2-y^2]
julia> algebraic_splitting_discriminant(E,I; factorization=false, clean_output=true);
Branch 1
  E = QQMPolyRingElem[2*x - 1, 4*y^2 - 4*z + 1]
  I = QQMPolyRingElem[(2*z - 1)*(4*z - 1)]
Branch 2
  E = QQMPolyRingElem[x^2 - x - y^2 + z]
  I = QQMPolyRingElem[y*(4*y^2 - 4*z + 1)*(y + z)*(y - z)*(y^2 - z), z*(2*z - 1)*(4*z - 1)*(z - 1)]
Branch 3
  E = QQMPolyRingElem[x*z - x + y^2 - 2*z + 1, y^4 - y^2*z^2 - y^2*z + z^3]
  I = QQMPolyRingElem[z*(2*z - 1)*(z - 1)]
Branch 4
  E = QQMPolyRingElem[x - 1, 2*y^2 - 1, 2*z - 1]
  I = QQMPolyRingElem[]
julia> algebraic_splitting_discriminant(E,I; factorization=true, clean_output=true);
Branch 1
  E = QQMPolyRingElem[x - 1, y^2 - z]
  I = QQMPolyRingElem[z*(z - 1)]
Branch 2
  E = QQMPolyRingElem[x + z - 1, y + z]
  I = QQMPolyRingElem[z*(2*z - 1)*(z - 1)]
Branch 3
  E = QQMPolyRingElem[x + z - 1, y - z]
  I = QQMPolyRingElem[z*(2*z - 1)*(z - 1)]
Branch 4
  E = QQMPolyRingElem[x^2 - x - y^2 + z]
  I = QQMPolyRingElem[y*(4*y^2 - 4*z + 1)*(y + z)*(y - z)*(y^2 - z), z*(2*z - 1)*(4*z - 1)*(z - 1)]
Branch 5
  E = QQMPolyRingElem[2*x^2 - 2*x - 2*y^2 + 1, 2*z - 1]
  I = QQMPolyRingElem[y*(2*y + 1)*(2*y - 1)*(2*y^2 - 1)]
Branch 6
  E = QQMPolyRingElem[x^2 - x - y^2 + 1, z - 1]
  I = QQMPolyRingElem[y*(4*y^2 - 3)*(y + 1)*(y - 1)]
Branch 7
  E = QQMPolyRingElem[2*x - 1, 4*y^2 - 4*z + 1]
  I = QQMPolyRingElem[z*(2*z - 1)*(4*z - 1)]
Branch 8
  E = QQMPolyRingElem[2*x + 2*y - 1, 4*z - 1]
  I = QQMPolyRingElem[y*(2*y + 1)*(2*y - 1)*(4*y + 1)*(4*y - 1)]
Branch 9
  E = QQMPolyRingElem[2*x - 2*y - 1, 4*z - 1]
  I = QQMPolyRingElem[y*(2*y + 1)*(2*y - 1)*(4*y + 1)*(4*y - 1)]
```
"""
function algebraic_splitting_discriminant(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}};
    max_steps::Integer=1000,
    clean_output::Bool=false,
    factorization::Bool=true,
) where T
    E_vec, I_vec = collect(E), collect(I)
    result = __algebraic_splitting_loop(
        E_vec,
        I_vec;
        max_steps=Int(max_steps),
        check_discriminants=true,
        factorize=factorization,
    )
    final_result = __algebraic_finalize_result(result)
    clean_output && print_splitting_result(final_result)
    return final_result
end

function algebraic_splitting_discriminant(
    E::AbstractVector{<:MPolyRingElem{T}};
    max_steps::Integer=1000,
    clean_output::Bool=false,
    factorization::Bool=true,
) where T
    E_vec = collect(E)
    return algebraic_splitting_discriminant(
        E_vec,
        empty(E_vec);
        max_steps,
        clean_output,
        factorization,
    )
end

function __algebraic_discriminant(p::T) where {T<:MPolyRingElem}
    is_constant(p) && return __algebraic_one(p)
    v = poly_leader(p)
    idx = var_index(v)
    d = degree(p, idx)
    d <= 1 && return __algebraic_one(p)

    dp = derivative(p, idx)
    is_zero(dp) && return __algebraic_zero(p)
    res = resultant(p, dp, idx)
    signed_res = isodd(div(d * (d - 1), 2)) ? -res : res
    h = poly_initial(p)
    division = __checked_divides(h, signed_res)
    division.divides && division.quotient !== nothing && return division.quotient
    return divexact(signed_res, h)
end

function __algebraic_specialized_squarefree_part(
    p::T,
    E_lower::Vector{T},
) where {T<:MPolyRingElem}
    reduced = if isempty(E_lower)
        p
    else
        Oscar.normal_form(p, ideal(__algebraic_parent(p), E_lower))
    end
    return __algebraic_squarefree_part(reduced)
end

function __algebraic_discriminant_zero_replacement(
    p::T,
    E_lower::Vector{T},
) where {T<:MPolyRingElem}
    is_constant(p) && return p
    v = poly_leader(p)
    idx = var_index(v)
    degree(p, idx) == 2 && return derivative(p, idx)
    return __algebraic_specialized_squarefree_part(p, E_lower)
end

function __algebraic_discriminant_issue(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    for i in eachindex(E)
        disc = __algebraic_squarefree_part(__algebraic_discriminant(E[i]))
        is_constant(disc) && !is_zero(disc) && continue
        E_partial, I_partial = __algebraic_initial_partial_system(E, I, disc)
        __compatible_zero_algebraic_system(disc, E_partial, I_partial) &&
            return (kind=:equation, index=i, discriminant=disc, E_partial=E_partial)
    end
    for i in eachindex(I)
        disc = __algebraic_squarefree_part(__algebraic_discriminant(I[i]))
        is_constant(disc) && !is_zero(disc) && continue
        E_partial, I_partial = __algebraic_initial_partial_system(E, I, disc)
        __compatible_zero_algebraic_system(disc, E_partial, I_partial) &&
            return (kind=:inequation, index=i, discriminant=disc, E_partial=E_partial)
    end
    return nothing
end

function __algebraic_discriminant_split(
    E::Vector{T},
    I::Vector{T},
    issue::NamedTuple,
) where {T<:MPolyRingElem}
    i = Int(issue.index)
    disc = issue.discriminant
    E_lower = unique!([issue.E_partial; disc])

    if issue.kind === :equation
        E_zero = copy(E)
        E_zero[i] = __algebraic_discriminant_zero_replacement(E[i], E_lower)
        push!(E_zero, disc)
        return __algebraic_normalized([(E, [I; disc]), (E_zero, I)])
    elseif issue.kind === :inequation
        I_zero = copy(I)
        I_zero[i] = __algebraic_discriminant_zero_replacement(I[i], E_lower)
        return __algebraic_normalized([(E, [I; disc]), ([E; disc], I_zero)])
    end

    error("Unknown discriminant issue kind: $(issue.kind)")
end