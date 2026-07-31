###############################################################################
#####  Quasi-simple and simple differential systems
###############################################################################

using Oscar
import Oscar: is_simple

if !isdefined(@__MODULE__, :__ACTION_POLY_CONVERSION_LOADED)
    include(joinpath(@__DIR__, "../../Algebraic/Conversion.jl"))
end

# Load this file after:
#
#   Code/NewDifferentialPolynomialRing.jl
#   Code/Algebraic.jl
#   Code/Quasi-Simple-Algebraic.jl
#   Conversion.jl

###############################################################################
#####  Differential jet and Janet data
###############################################################################

@doc raw"""
    __differential_jet_data(v::DifferentialPolyRingElem)

Return `(symbol, multiindex)` for the differential indeterminate `v`.

Here `symbol` is the position of the elementary differential indeterminate and
`multiindex` is the exponent vector of the commuting derivations.
"""
function __differential_jet_data(v::DifferentialPolyRingElem)
    jet_data = Oscar.__vtj(parent(v))
    haskey(jet_data, v) || error("$v is not a differential indeterminate")
    symbol, multiindex = jet_data[v]
    return (symbol=Int(symbol), multiindex=collect(Int, multiindex))
end

@doc raw"""
    differential_elementary_symbol(v::DifferentialPolyRingElem) -> Int

Return the position of the elementary differential indeterminate underlying
the jet variable `v`.

# Examples

```jldoctest
julia> R, (u, v) = differential_polynomial_ring(QQ, [:u, :v], 2);
julia> differential_elementary_symbol(diff_action(v, [2, 1]))
2
```
"""
function differential_elementary_symbol(v::DifferentialPolyRingElem)
    return __differential_jet_data(v).symbol
end

@doc raw"""
    differential_multiindex(v::DifferentialPolyRingElem) -> Vector{Int}

Return the differential multiindex of the jet variable `v`.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 2);
julia> differential_multiindex(diff_action(u, [2, 3]))
2-element Vector{Int64}:
 2
 3
```
"""
function differential_multiindex(v::DifferentialPolyRingElem)
    return __differential_jet_data(v).multiindex
end

"""
Internal cache for Janet computations on differential polynomials.
"""
struct __DifferentialJanetData{T}
    equations::Vector{DifferentialPolyRingElem{T}}
    leaders::Vector{DifferentialPolyRingElem{T}}
    symbols::Vector{Int}
    multiindices::Vector{Vector{Int}}
    by_symbol::Vector{Vector{Int}}
    multiplicative_variables::Vector{Vector{Int}}
end

function __differential_leader_data(
    equations::Vector{DifferentialPolyRingElem{T}},
) where {T}
    isempty(equations) && return (
        leaders=DifferentialPolyRingElem{T}[],
        symbols=Int[],
        multiindices=Vector{Int}[],
        by_symbol=Vector{Vector{Int}}(),
    )

    any(is_constant, equations) &&
        error("The differential equations must be nonconstant")

    R = parent(first(equations))
    any(f -> parent(f) !== R, equations) &&
        error("All differential equations must have the same parent")

    leaders = Vector{DifferentialPolyRingElem{T}}(undef, length(equations))
    symbols = Vector{Int}(undef, length(equations))
    multiindices = Vector{Vector{Int}}(undef, length(equations))
    by_symbol = [Int[] for _ in 1:n_elementary_symbols(R)]

    for i in eachindex(equations)
        ld = leader(equations[i])
        jet = __differential_jet_data(ld)
        leaders[i] = ld
        symbols[i] = jet.symbol
        multiindices[i] = jet.multiindex
        push!(by_symbol[jet.symbol], i)
    end

    return (
        leaders=leaders,
        symbols=symbols,
        multiindices=multiindices,
        by_symbol=by_symbol,
    )
end

function __differential_janet_data(
    equations::Vector{DifferentialPolyRingElem{T}},
) where {T}
    leader_data = __differential_leader_data(equations)
    multiplicative = Vector{Vector{Int}}(undef, length(equations))

    for indices in leader_data.by_symbol
        isempty(indices) && continue
        local_multiindices = leader_data.multiindices[indices]
        local_multiplicative = multiplicative_variables(local_multiindices)
        for (local_index, equation_index) in enumerate(indices)
            multiplicative[equation_index] =
                collect(Int, local_multiplicative[local_index])
        end
    end

    return __DifferentialJanetData{T}(
        equations,
        leader_data.leaders,
        leader_data.symbols,
        leader_data.multiindices,
        leader_data.by_symbol,
        multiplicative,
    )
end

function __differential_janet_data(
    equations::Vector{DifferentialPolyRingElem{T}},
    multiplicative::Vector{Vector{Int}},
) where {T}
    length(equations) == length(multiplicative) ||
        error("There must be one multiplicative-variable mask per equation")

    leader_data = __differential_leader_data(equations)
    if !isempty(equations)
        n = n_action_maps(parent(first(equations)))
        any(mask -> length(mask) != n, multiplicative) &&
            error("Each mask must have one entry per derivation")
    end

    return __DifferentialJanetData{T}(
        equations,
        leader_data.leaders,
        leader_data.symbols,
        leader_data.multiindices,
        leader_data.by_symbol,
        multiplicative,
    )
end

function __differential_janet_data(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    return __differential_janet_data(collect(equations))
end

function __differential_janet_data(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    multiplicative::AbstractVector{<:AbstractVector{<:Integer}},
) where {T}
    masks = [collect(Int, mask) for mask in multiplicative]
    return __differential_janet_data(collect(equations), masks)
end

function __differential_nonnegative(alpha::AbstractVector{<:Integer})
    return all(x -> x >= 0, alpha)
end

function __differential_admissible(
    alpha::AbstractVector{<:Integer},
    multiplicative::AbstractVector{<:Integer},
)
    length(alpha) == length(multiplicative) ||
        error("The multiindex and Janet mask have different lengths")

    for i in eachindex(alpha, multiplicative)
        !iszero(alpha[i]) && iszero(multiplicative[i]) && return false
    end
    return true
end

function __differential_prolongation(
    f::DifferentialPolyRingElem{T},
    alpha::AbstractVector{<:Integer},
) where {T<:RingElement}
    all(iszero, alpha) && return f + getfield(f, :parent)(0)
    multiplicities::Vector{Int} = collect(Int, alpha)
    result = f
    for derivation_index in eachindex(multiplicities)
        for _ in 1:multiplicities[derivation_index]
            result = diff_action(result, derivation_index)
        end
    end
    return result
end

###############################################################################
#####  Janet completeness
###############################################################################

@doc raw"""
    differential_multiplicative_variables(equations)
    multiplicative_variables(equations)

Return the Janet multiplicative-derivation masks of the differential equations.

The leaders are grouped by elementary differential indeterminate before Janet
division is applied to their multiindices.  The second signature is the common
public name used by the algebraic and difference-polynomial code.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 2);
julia> E = [diff_action(u, [1, 0]), diff_action(u, [0, 1])];
julia> differential_multiplicative_variables(E)
2-element Vector{Vector{Int64}}:
 [1, 1]
 [0, 1]
```
"""
function differential_multiplicative_variables(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    isempty(equations) && return Vector{Vector{Int}}()
    return __differential_janet_data(equations).multiplicative_variables
end

function multiplicative_variables(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    return differential_multiplicative_variables(equations)
end

function __is_differential_janet_complete(
    janet::__DifferentialJanetData;
    info::Bool=false,
    missing::Bool=false,
)
    if length(janet.leaders) != length(Set(janet.leaders))
        info && println("There are differential equations with the same leader")
        return false
    end

    for symbol in eachindex(janet.by_symbol)
        multiindices = janet.multiindices[janet.by_symbol[symbol]]
        missing_multiindex = __first_janet_missing(multiindices)
        missing_multiindex === nothing && continue

        if info
            println(
                "The leader multiindices for elementary symbol $symbol " *
                "are not Janet complete: $missing_multiindex is missing",
            )
        end
        return missing ?
            (false, symbol, collect(Int, missing_multiindex)) :
            false
    end
    return true
end

@doc raw"""
    is_janet_complete(
        equations::AbstractVector{<:DifferentialPolyRingElem};
        info::Bool=false,
        missing::Bool=false,
    )

Test Janet completeness of a finite set of differential equations.

With `missing=true`, return the first missing leader multiindex together with
the elementary-symbol position.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 2);
julia> E = [diff_action(u, [1, 0]), diff_action(u, [0, 1])];
julia> is_janet_complete(E)
true
julia> E = [u, diff_action(u, [2, 0])];
julia> is_janet_complete(E; missing=true)
(false, 1, [1, 0])
```
"""
function is_janet_complete(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
    missing::Bool=false,
) where {T}
    isempty(equations) && return true
    if any(is_constant, equations)
        info && println("There are constant differential equations")
        return false
    end
    return __is_differential_janet_complete(
        __differential_janet_data(equations);
        info,
        missing,
    )
end

###############################################################################
#####  Janet reducibility
###############################################################################

function __differential_reduction_candidate(
    p::DifferentialPolyRingElem{T},
    variable::DifferentialPolyRingElem{T},
    janet::__DifferentialJanetData{T},
) where {T}
    jet = __differential_jet_data(variable)

    for equation_index in janet.by_symbol[jet.symbol]
        alpha = jet.multiindex - janet.multiindices[equation_index]
        __differential_nonnegative(alpha) || continue
        __differential_admissible(
            alpha,
            janet.multiplicative_variables[equation_index],
        ) || continue

        prolongation =
            __differential_prolongation(janet.equations[equation_index], alpha)
        leader(prolongation) == variable || continue
        degree(p, variable) >= degree(prolongation, variable) || continue

        return (
            reducible=true,
            alpha=alpha,
            polynomial=janet.equations[equation_index],
            prolongation=prolongation,
            multiplicative_variables=
                janet.multiplicative_variables[equation_index],
        )
    end
    return false
end

function __is_differential_janet_reducible(
    p::DifferentialPolyRingElem,
    janet::__DifferentialJanetData,
)
    isempty(janet.equations) && return false
    parent(p) === parent(first(janet.equations)) ||
        error("The polynomial and equations must have the same parent")

    for variable in vars(p)
        reduction = __differential_reduction_candidate(p, variable, janet)
        reduction === false || return reduction
    end
    return false
end

function __is_differential_janet_reducible_univariate(
    p::DifferentialPolyRingElem{T},
    janet::__DifferentialJanetData{T},
) where {T}
    isempty(janet.equations) && return false
    is_constant(p) && return false
    parent(p) === parent(first(janet.equations)) ||
        error("The polynomial and equations must have the same parent")
    return __differential_reduction_candidate(p, leader(p), janet)
end

@doc raw"""
    is_janet_reducible(p, equations)
    is_janet_reducible(p, equations, multiplicative)

Return a Janet reduction certificate for the differential polynomial `p`, or
`false` if `p` is Janet reduced modulo `equations` and their admissible
derivatives.

The certificate contains the differential multiindex `alpha`, the original
equation, and its actual prolongation.  Computing the prolongation is
essential: unlike a shift, a nonzero derivative of a nonlinear equation is
linear in its new leader.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> E = [u^2 - 1];
julia> reduction = is_janet_reducible(diff_action(u, 1), E);
julia> reduction.alpha
1-element Vector{Int64}:
 1
julia> reduction.prolongation == 2*u*diff_action(u, 1)
true
```
"""
function is_janet_reducible(
    p::DifferentialPolyRingElem{T},
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    isempty(equations) && return false
    return __is_differential_janet_reducible(
        p,
        __differential_janet_data(equations),
    )
end

function is_janet_reducible(
    p::DifferentialPolyRingElem{T},
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    multiplicative::AbstractVector{<:AbstractVector{<:Integer}},
) where {T}
    isempty(equations) && return false
    return __is_differential_janet_reducible(
        p,
        __differential_janet_data(equations, multiplicative),
    )
end

function is_janet_reducible_univariate(
    p::DifferentialPolyRingElem{T},
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    isempty(equations) && return false
    return __is_differential_janet_reducible_univariate(
        p,
        __differential_janet_data(equations),
    )
end

###############################################################################
#####  Differential Janet normal form
###############################################################################

function __differential_normal_form_data(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    collected_equations = collect(equations)
    any(is_constant, collected_equations) &&
        error("The differential equations must be nonconstant")

    janet = __differential_janet_data(collected_equations)
    __is_differential_janet_complete(janet) ||
        error("The differential equations must be Janet complete")
    return janet
end

function __differential_normal_form(
    p::DifferentialPolyRingElem,
    janet::__DifferentialJanetData,
)
    R = parent(p)
    if !isempty(janet.equations)
        R === parent(first(janet.equations)) ||
            error("The polynomial and equations must have the same parent")
    end

    remainder = p + R(0)
    multiplier = R(1)

    while !is_constant(remainder)
        reduction =
            __is_differential_janet_reducible_univariate(remainder, janet)
        reduction === false && break

        v = leader(remainder)
        reducer = reduction.prolongation
        degree_difference = degree(remainder, v) - degree(reducer, v)
        reducer_initial = initial(reducer)

        multiplier *= reducer_initial
        remainder =
            reducer_initial * remainder -
            initial(remainder) * v^degree_difference * reducer
    end

    if !is_constant(remainder)
        v = leader(remainder)
        coefficients = univariate_coefficients(remainder, v)
        reduced_remainder = R(0)
        coefficient_multiplier = R(1)

        for (degree_offset, coefficient) in enumerate(coefficients)
            is_zero(coefficient) && continue
            reduced_coefficient =
                __differential_normal_form(coefficient, janet)
            exponent = degree_offset - 1

            reduced_remainder =
                reduced_coefficient.multiplier * reduced_remainder +
                coefficient_multiplier * reduced_coefficient.nf * v^exponent
            coefficient_multiplier *= reduced_coefficient.multiplier
        end

        remainder = reduced_remainder
        multiplier *= coefficient_multiplier

        # Products of coefficient multipliers can become reducible by an
        # equation with a strictly lower leader.  Normalize once more only in
        # that case; every such recurrence descends in the ranking.
        if __is_differential_janet_reducible(remainder, janet) !== false
            reduced_again = __differential_normal_form(remainder, janet)
            remainder = reduced_again.nf
            multiplier *= reduced_again.multiplier
        end
    end

    return (nf=remainder, multiplier=multiplier)
end

@doc raw"""
    normal_form(
        p::DifferentialPolyRingElem,
        equations::AbstractVector{<:DifferentialPolyRingElem},
    )

Compute the differential Janet normal form of `p` modulo a Janet-complete set
of equations.

The result is `(nf=r, multiplier=h)`, where `h*p-r` belongs to the
differential ideal generated by `equations`.  The multiplier is a product of
initials of admissible derivatives of the equations; these include separants
when nonzero derivatives of nonlinear equations are used.

This implementation is independent of the difference-polynomial normal-form
code and is typed throughout with `DifferentialPolyRingElem`.

# Examples

An ordinary linear equation:

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> du = diff_action(u, 1);
julia> E = [du - u];
julia> normal_form(diff_action(u, [2]) - u, E)
(nf = 0, multiplier = 1)
```

The derivative of a nonlinear equation is linear in its new leader:

```jldoctest
julia> K, t = rational_function_field(QQ, "t");
julia> R, u = differential_polynomial_ring(K, :u, [c -> derivative(c)]);
julia> f = u^2 - t;
julia> normal_form(diff_action(u, 1), [f])
(nf = 1, multiplier = 2*u[0])
```

Several coefficient reductions use a common product of initials:

```jldoctest
julia> R, (v, u) = differential_polynomial_ring(QQ, [:v, :u], 1);
julia> du = diff_action(u, 1); w = diff_action(v, [2]);
julia> E = [u*du - 1];
julia> p = du*w^2 + (du + u)*w;
julia> normal_form(p, E)
(nf = u[0]*v[2]^2 + (u[0]^3 + u[0])*v[2], multiplier = u[0]^2)
```

The common multiplier is normalized at lower leaders when necessary:

```jldoctest
julia> R, (v, u, a) = differential_polynomial_ring(QQ, [:v, :u, :a], 1);
julia> w = diff_action(v, [2]); du = diff_action(u, 1);
julia> E = [a^2 - 1, a*du - 1];
julia> normal_form(du*(w^2 + w + 1), E)
(nf = v[2]^2 + v[2] + 1, multiplier = a[0]^3)
```
"""
function normal_form(
    p::DifferentialPolyRingElem{T},
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    isempty(equations) && return (nf=p + parent(p)(0), multiplier=parent(p)(1))
    return __differential_normal_form(
        p,
        __differential_normal_form_data(equations),
    )
end

###############################################################################
#####  Passivity
###############################################################################

function __is_differential_passive(
    janet::__DifferentialJanetData{T};
    info::Bool=false,
    missing::Bool=false,
) where {T<:RingElement}
    isempty(janet.equations) && return true

    for equation_index in eachindex(janet.equations)
        mask = janet.multiplicative_variables[equation_index]
        for derivation_index in eachindex(mask)
            iszero(mask[derivation_index]) || continue

            prolongation =
                diff_action(janet.equations[equation_index], derivation_index)
            obstruction =
                __differential_normal_form(prolongation, janet).nf
            is_zero(obstruction) && continue

            if info
                println(
                    "$obstruction is the nonzero Janet normal form of " *
                    "derivation $derivation_index applied to equation " *
                    "$equation_index",
                )
            end
            return missing ?
                (false, obstruction, equation_index, derivation_index) :
                false
        end
    end
    return true
end

@doc raw"""
    is_passive(
        equations::AbstractVector{<:DifferentialPolyRingElem};
        info::Bool=false,
        missing::Bool=false,
    )
    is_passive(equations, multiplicative; info=false, missing=false)

Test passivity of a Janet-complete differential equation system.

For every equation and every non-admissible elementary derivation, the
function computes the differential Janet normal form of the corresponding
prolongation.  The system is passive exactly when all these normal forms are
zero.

# Examples

A single equation has only multiplicative derivations:

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 2);
julia> is_passive([diff_action(u, [1, 0])])
true
```

The equations `u_x=v`, `u_y=w` expose their compatibility condition:

```jldoctest
julia> R, (u, v, w) = differential_polynomial_ring(QQ, [:u, :v, :w], 2; index_ordering_name=:deglex);
julia> E = [diff_action(u, [1, 0]) - v,
            diff_action(u, [0, 1]) - w];
julia> is_janet_complete(E)
true
julia> is_passive(E)
false
```
"""
function is_passive(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
    missing::Bool=false,
) where {T}
    isempty(equations) && return true
    janet = __differential_janet_data(equations)
    __is_differential_janet_complete(janet) ||
        error("The differential equations must be Janet complete")
    return __is_differential_passive(janet; info, missing)
end

function is_passive(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    multiplicative::AbstractVector{<:AbstractVector{<:Integer}};
    info::Bool=false,
    missing::Bool=false,
) where {T}
    isempty(equations) && return true
    janet = __differential_janet_data(equations, multiplicative)
    __is_differential_janet_complete(janet) ||
        error("The differential equations must be Janet complete")
    return __is_differential_passive(janet; info, missing)
end

###############################################################################
#####  Algebraic quasi-simplicity and discriminants
###############################################################################

function __differential_algebraic_system_data(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    E = collect(equations)
    Q = collect(inequations)
    system = [E; Q]
    R = __differential_parent(system)

    any(p -> parent(p) !== R, system) &&
        error("All differential polynomials must have the same parent")
    any(is_constant, system) && return nothing

    conversion::__DifferentialAlgebraicData{T} =
        __differential_algebraic_conversion_data(system; variable_name=:x)
    converted = __conversion_convert_collection(
        system,
        conversion,
        :__convert_differential_polynomial,
    )

    number_of_equations = length(E)
    E_algebraic = converted[1:number_of_equations]
    Q_algebraic = converted[(number_of_equations + 1):end]
    return (
        equations=E,
        inequations=Q,
        system=system,
        conversion=conversion,
        algebraic_equations=E_algebraic,
        algebraic_inequations=Q_algebraic,
    )
end

function __is_differential_algebraically_quasi_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    isempty(equations) && isempty(inequations) && return true
    algebraic_data =
        __differential_algebraic_system_data(equations, inequations)
    if algebraic_data === nothing
        info && println("There are constant differential polynomials")
        return false
    end

    E_algebraic = algebraic_data.algebraic_equations
    Q_algebraic = algebraic_data.algebraic_inequations
    if isempty(E_algebraic)
        return is_quasi_simple_ineq(Q_algebraic; info)
    elseif isempty(Q_algebraic)
        return is_quasi_simple(E_algebraic; info)
    end
    return is_quasi_simple(E_algebraic, Q_algebraic; info)
end

__differential_is_algebraic_constant(p) = is_constant(p)

function __discriminant_has_compatible_zero(
    discriminant_algebraic::MPolyRingElem,
    equations_algebraic::AbstractVector,
    inequations_algebraic::AbstractVector,
)
    if __differential_is_algebraic_constant(discriminant_algebraic)
        return is_zero(discriminant_algebraic)
    end
    return solution_check(
        discriminant_algebraic,
        equations_algebraic,
        inequations_algebraic,
    )
end

function __is_differential_algebraically_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T<:RingElement}
    isempty(equations) && isempty(inequations) && return true
    algebraic_data =
        __differential_algebraic_system_data(equations, inequations)
    if algebraic_data === nothing
        info && println("There are constant differential polynomials")
        return false
    end

    E_algebraic = algebraic_data.algebraic_equations
    Q_algebraic = algebraic_data.algebraic_inequations
    algebraically_quasi_simple = if isempty(E_algebraic)
        is_quasi_simple_ineq(Q_algebraic; info)
    elseif isempty(Q_algebraic)
        is_quasi_simple(E_algebraic; info)
    else
        is_quasi_simple(E_algebraic, Q_algebraic; info)
    end
    algebraically_quasi_simple || return false

    for (index, p) in enumerate(algebraic_data.system)
        differential_discriminant = discriminant(p)
        algebraic_discriminant = __convert_differential_polynomial(
            differential_discriminant,
            algebraic_data.conversion,
        )

        __discriminant_has_compatible_zero(
            algebraic_discriminant,
            E_algebraic,
            Q_algebraic,
        ) || continue

        if info
            kind =
                index <= length(algebraic_data.equations) ?
                "equation" :
                "inequation"
            local_index =
                index <= length(algebraic_data.equations) ?
                index :
                index - length(algebraic_data.equations)
            println(
                "The discriminant of $kind $local_index vanishes on " *
                "the corresponding projected solution set",
            )
        end
        return false
    end
    return true
end

@doc raw"""
    is_simple_as_algebraic_differential_system(
        equations,
        inequations=empty(equations);
        info=false,
    )

Test whether the given system is simple as an algebraic system.
The existing algebraic `is_quasi_simple` implementation is used unchanged for
    nonconstant polynomials, distinct leaders, and regular initials.
This function adds the missing condition that the discriminant of every equation and every
    inequation has no zero on the corresponding projected solution set.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> is_simple_as_algebraic_differential_system([u^2])
false
julia> is_simple_as_algebraic_differential_system([u^2 - 1])
true
```
"""
function is_simple_as_algebraic_differential_system(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    return __is_differential_algebraically_simple(
        equations,
        inequations;
        info,
    )
end

function is_simple_as_algebraic_differential_system(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    isempty(equations) && return true
    Q = DifferentialPolyRingElem{T}[]
    return __is_differential_algebraically_simple(equations, Q; info)
end

###############################################################################
#####  Quasi-simple and simple differential systems
###############################################################################

function __differential_system_status(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    check_discriminants::Bool,
) where {T<:RingElement}
    E = collect(DifferentialPolyRingElem{T}, equations)
    Q = collect(DifferentialPolyRingElem{T}, inequations)
    isempty(E) && isempty(Q) && return (valid=true, message="")

    system = [E; Q]
    R = __differential_parent(system)
    any(p -> parent(p) !== R, system) &&
        error("All differential polynomials must have the same parent")

    algebraically_valid = if check_discriminants
        __is_differential_algebraically_simple(E, Q)
    else
        __is_differential_algebraically_quasi_simple(E, Q)
    end
    algebraically_valid || return (
        valid=false,
        message=check_discriminants ?
            "The system is not simple as an algebraic system" :
            "The system is not quasi-simple as an algebraic system",
    )

    isempty(E) && return (valid=true, message="")
    janet = __differential_janet_data(E)

    __is_differential_janet_complete(janet) || return (
        valid=false,
        message="The differential equations are not Janet complete",
    )
    __is_differential_passive(janet) || return (
        valid=false,
        message="The differential equations are not passive",
    )

    for q in Q
        __is_differential_janet_reducible(q, janet) === false || return (
            valid=false,
            message="The inequation $q is not Janet reduced",
        )
    end
    return (valid=true, message="")
end

@doc raw"""
    is_quasi_simple(
        equations::AbstractVector{<:DifferentialPolyRingElem},
        inequations::AbstractVector{<:DifferentialPolyRingElem};
        info=false,
    )
    is_quasi_simple(equations; info=false)

Test whether a differential system is quasi-simple:

1. the finite algebraic system in the occurring jets is quasi-simple;
2. the equations are Janet complete and passive;
3. every inequation is Janet reduced modulo the equations and their
   admissible derivatives.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> is_quasi_simple([diff_action(u, 1) - u], [u])
true
julia> is_quasi_simple([u^2])
true
julia> is_simple([u^2])
false
julia> E = [diff_action(u, 1)];
julia> is_quasi_simple(E, [diff_action(u, [2]) + 1]; info=true)
The inequation u[2] + 1 is not Janet reduced
false
```
"""
function is_quasi_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    status = __differential_system_status(
        equations,
        inequations;
        check_discriminants=false,
    )
    info && !status.valid && println(status.message)
    return status.valid
end

function is_quasi_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    Q = DifferentialPolyRingElem{T}[]
    return is_quasi_simple(equations, Q; info)
end

@doc raw"""
    is_simple(
        equations::AbstractVector{<:DifferentialPolyRingElem},
        inequations::AbstractVector{<:DifferentialPolyRingElem};
        info=false,
    )
    is_simple(equations; info=false)

Test whether a differential system is simple:

1. it is simple as a finite algebraic system, including the discriminant condition;
2. its differential equations are passive;
3. its inequations are Janet reduced modulo the passive equations.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> is_simple([diff_action(u, 1) - u], [u])
true
julia> is_simple([u^2]; info=true)
The system is not simple as an algebraic system
false
```
"""
function is_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    status = __differential_system_status(
        equations,
        inequations;
        check_discriminants=true,
    )
    info && !status.valid && println(status.message)
    return status.valid
end

function is_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    Q = DifferentialPolyRingElem{T}[]
    return is_simple(equations, Q; info)
end

@doc raw"""
    is_quasi_simple_ineq(inequations; info=false)
    is_simple_ineq(inequations; info=false)

Test a differential system containing only inequations.  Janet passivity is
vacuous; quasi-simplicity uses the existing algebraic test, while simplicity
also checks the discriminants.
"""
function is_quasi_simple_ineq(
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    isempty(inequations) && return true
    E = DifferentialPolyRingElem{T}[]
    return is_quasi_simple(E, inequations; info)
end

function is_simple_ineq(
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false,
) where {T}
    isempty(inequations) && return true
    E = DifferentialPolyRingElem{T}[]
    return is_simple(E, inequations; info)
end