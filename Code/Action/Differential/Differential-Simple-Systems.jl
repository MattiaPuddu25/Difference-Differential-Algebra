# Write an ActionSystem from vectors
function __action_system(
    E::AbstractVector{<:DifferencePolyRingElem{T}},
    I::AbstractVector{<:DifferencePolyRingElem{T}}
) where {T}
    return DifferenceSystem(E, I)
end

function __action_system(
    E::AbstractVector{<:DifferentialPolyRingElem{T}},
    I::AbstractVector{<:DifferentialPolyRingElem{T}}
) where {T}
    return DifferentialSystem(E, I)
end

###############################################################################
#####  Action (Quasi-)simple as an algebraic system
###############################################################################

@doc raw"""
    is_quasi_simple_as_algebraic_system(
        system::AbstractActionSystem;
        info::Bool=false
    )
    is_quasi_simple_as_algebraic_system(
        E::AbstractVector{<:P},
        I::AbstractVector{<:P};
        info::Bool=false
    ) where {P<:ActionPolyRingElem}
    is_quasi_simple_as_algebraic_system(
        E::AbstractVector{<:P};
        info::Bool=false
    ) where {P<:ActionPolyRingElem}

Test whether the given system is quasi-simple as an algebraic system,
using the algebraic `is_quasi_simple` implementation after conversion.\
If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.
"""

function is_quasi_simple_as_algebraic_system(
    system::AbstractActionSystem;
    info::Bool=false
)
    data = getfield(system, :_data)
    isempty(data.equations) && isempty(data.inequations) && return true
    algebraic_system = __conversion_algebraic_system(system)
    return is_quasi_simple(algebraic_system; info)
end

function is_quasi_simple_as_algebraic_system(
    E::AbstractVector{<:P},
    I::AbstractVector{<:P};
    info::Bool=false
) where {P<:ActionPolyRingElem}
    __require_same_parent(E, I)
    isempty(E) && isempty(I) && return true
    return is_quasi_simple_as_algebraic_system(__action_system(E, I); info)
end

function is_quasi_simple_as_algebraic_system(
    E::AbstractVector{<:P};
    info::Bool=false
) where {P<:ActionPolyRingElem}
    isempty(E) && return true
    return is_quasi_simple_as_algebraic_system(E, empty(E); info)
end

@doc raw"""
    is_simple_as_algebraic_system(
        system::AbstractActionSystem;
        info::Bool=false
    )
    is_simple_as_algebraic_system(
        E::AbstractVector{<:P},
        I::AbstractVector{<:P};
        info::Bool=false
    ) where {P<:ActionPolyRingElem}
    is_simple_as_algebraic_system(
        E::AbstractVector{<:P};
        info::Bool=false
    ) where {P<:ActionPolyRingElem}

Test whether the given system is simple as an algebraic system,
using the algebraic `is_simple` implementation after conversion.
If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.

    # Examples

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 1);
    julia> is_simple_as_algebraic_system([u^2]; info=true)
    The discriminant of Equation 1 has solutions in the corresponding subsystem
    false
    julia> is_simple_as_algebraic_system([u^2 - 1])
    true
    ```
"""
function is_simple_as_algebraic_system(
    system::AbstractActionSystem;
    info::Bool=false
)
    data = getfield(system, :_data)
    isempty(data.equations) && isempty(data.inequations) && return true
    algebraic_system = __conversion_algebraic_system(system)
    return is_simple(algebraic_system; info)
end

function is_simple_as_algebraic_system(
    E::AbstractVector{<:P},
    I::AbstractVector{<:P};
    info::Bool=false
) where {P<:ActionPolyRingElem}
    __require_same_parent(E, I)
    isempty(E) && isempty(I) && return true
    return is_simple_as_algebraic_system(__action_system(E, I); info)
end

function is_simple_as_algebraic_system(
    E::AbstractVector{<:P};
    info::Bool=false
) where {P<:ActionPolyRingElem}
    isempty(E) && return true
    return is_simple_as_algebraic_system(E, empty(E); info)
end

###############################################################################
#####  (Quasi-)simple differential systems
###############################################################################

function __differential_system_status(
    system::DifferentialSystem;
    check_discriminants::Bool,
    check_minimality::Bool=true
)
    stored = getfield(system, :_data)
    isempty(stored.equations) && isempty(stored.inequations) && return (valid=true, message="")
    R = system.ring
    (any(p -> parent(p) !== R, stored.equations) || any(p -> parent(p) !== R, stored.inequations)) &&
        error("All differential polynomials must have the same parent")
    algebraically_valid = if check_discriminants
        is_simple_as_algebraic_system(system)
    else
        is_quasi_simple_as_algebraic_system(system)
    end
    algebraically_valid || return (
        valid=false,
        message=check_discriminants ?
            "The system is not simple as an algebraic system" :
            "The system is not quasi-simple as an algebraic system"
    )
    isempty(stored.equations) && return (valid=true, message="")
    # These checks only read the equations and return a Boolean status and message.
    E, Q = stored.equations, stored.inequations
    janet = __action_janet_data(E)
    __is_janet_complete(janet) || return (
        valid=false,
        message="The differential equations are not Janet complete"
    )
    if check_minimality
    # Check minimality of the leader shifts, allowing prolongations required by Janet completion.
    # Do not infer dependencies between equations from their leaders alone;
    # see the examples in the docstring of __differential_minimality_status.
        minimality = __differential_minimality_status(janet)
        minimality.minimal || return (valid=false,
            message=
                "The differential equations do not form a minimal Janet leader set " *
                "for differential indeterminate $(minimality.symbol)"
        )
    end
    __is_passive(janet) || return (valid=false,
        message="The differential equations are not passive"
    )
    for q in Q
        __is_janet_reducible(q, janet) === false || return (valid=false,
            message="The inequation $q is not Janet reduced"
        )
    end
    return (valid=true, message="")
end

@doc raw"""
    is_quasi_simple(
        equations::AbstractVector{<:DifferentialPolyRingElem{T}},
        inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
        info::Bool=false
    ) where {T}
    is_quasi_simple(
        equations::AbstractVector{<:DifferentialPolyRingElem{T}};
        info::Bool=false
    ) where {T}
    is_quasi_simple(system::DifferentialSystem; info::Bool=false)

Test whether a differential system is quasi-simple:

1. It is simple as an algebraic system;
2. The set of equations are Janet complete;
3. The leaders of the equations form a (minimal) Janet basis;
4. The set of equations is passive;
5. Its inequations are Janet reduced modulo the passive equations.

The input can be either given as a `DifferentialSystem` or as vectors of `equations` and,
possibly, `inequations`.
If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.

    # Examples

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 1);
    julia> is_quasi_simple([u[1] - u], [u])
    true

    julia> P, (g4, g6) = polynomial_ring(QQ, [:G4, :G6]);
    julia> K = fraction_field(P);
    julia> G4 = K(g4);
    julia> G6 = K(g6);
    julia> R, u = differential_polynomial_ring(K, :u, [action_derivation(K)]);
    julia> p = u[1]^2 - 4*u[0]^3 + 60*G4*u[0] + 140*G6;
    julia> system = DifferentialSystem([p]);
    julia> is_quasi_simple(system)
    true

    julia> R, (y,x) = differential_polynomial_ring(QQ,[:y,:x],1);
    julia> D = DifferentialSystem([x*y^2+x*y, y[1]]);
    julia> is_quasi_simple(D; info=true)
    The system is not quasi-simple as an algebraic system
    false
    ```
"""
function is_quasi_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false
) where {T}
    isempty(equations) && isempty(inequations) && return true
    return is_quasi_simple(DifferentialSystem(equations, inequations); info)
end

function is_quasi_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false
) where {T}
    Q = DifferentialPolyRingElem{T}[]
    return is_quasi_simple(equations, Q; info)
end

is_quasi_simple(system::DifferentialSystem; info::Bool=false) =
    __certify_differential_system(system; check_discriminants=false, info)

@doc raw"""
    is_simple(
        equations::AbstractVector{<:DifferentialPolyRingElem},
        inequations::AbstractVector{<:DifferentialPolyRingElem};
        info=false
    )
    is_simple(
        equations::AbstractVector{<:DifferentialPolyRingElem{T}};
        info::Bool=false
    ) where {T}
    is_simple(system::DifferentialSystem; info::Bool=false)

Test whether a differential system is simple:

1. It is simple as an algebraic system;
2. The set of equations are Janet complete;
3. The leaders of the equations form a (minimal) Janet basis;
4. The set of equations is passive;
5. Its inequations are Janet reduced modulo the passive equations.

The input can be either given as a `DifferentialSystem` or as vectors of `equations` and,
possibly, `inequations`.
If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.

    # Examples

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 1);
    julia> is_simple([u[1] - u], [u])
    true
    julia> is_simple([u^2]; info=true)
    The system is not simple as an algebraic system
    false

    julia> P, (g4, g6) = polynomial_ring(QQ, [:G4, :G6]);
    julia> K = fraction_field(P);
    julia> G4 = K(g4);
    julia> G6 = K(g6);
    julia> R, u = differential_polynomial_ring(K, :u, [action_derivation(K)]);
    julia> p = u[1]^2 - 4*u[0]^3 + 60*G4*u[0] + 140*G6;
    julia> system = DifferentialSystem([p]);
    julia> is_simple(system; info=true)
    The system is not simple as an algebraic system
    false
    julia> D = thomas_decomposition(system)
    2-element Vector{DifferentialSystem}:
     DifferentialSystem(1 equations, 0 inequations; simple = true)
     DifferentialSystem(1 equations, 1 inequations; simple = true)
    julia> D[1].equations
    1-element Vector{DifferentialPolyRingElem{AbstractAlgebra.Generic.FracFieldElem{QQMPolyRingElem}}}:
     u[0]^3 - 15*G4*u[0] - 35*G6
    julia> is_simple(D[1])
    true
    julia> D[2].equations
    1-element Vector{DifferentialPolyRingElem{AbstractAlgebra.Generic.FracFieldElem{QQMPolyRingElem}}}:
     u[1]^2 - 4*u[0]^3 + 60*G4*u[0] + 140*G6
    julia> D[2].inequations
    1-element Vector{DifferentialPolyRingElem{AbstractAlgebra.Generic.FracFieldElem{QQMPolyRingElem}}}:
     u[0]^3 - 15*G4*u[0] - 35*G6
    julia> is_simple(D[2])
    true
```
"""
function is_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false
) where {T}
    isempty(equations) && isempty(inequations) && return true
    return is_simple(DifferentialSystem(equations, inequations); info)
end

function is_simple(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}};
    info::Bool=false
) where {T}
    Q = DifferentialPolyRingElem{T}[]
    return is_simple(equations, Q; info)
end

is_simple(system::DifferentialSystem; info::Bool=false) =
    __certify_differential_system(system; check_discriminants=true, info)

function __certify_differential_system(system::DifferentialSystem; check_discriminants::Bool, info::Bool=false)
    key = check_discriminants ? :simple : :quasi_simple
    certificates = getfield(system, :_data).certificates
    cached = get(certificates, key, nothing)
    if cached === nothing || (info && !cached)
        status = __differential_system_status(system; check_discriminants)
        certificates[key] = status.valid
        # A successful simple-system check also proves quasi-simplicity.
        check_discriminants && status.valid && (certificates[:quasi_simple] = true)
        info && !status.valid && println(status.message)
    end
    return certificates[key]
end

###############################################################################
#####  Janet minimality
###############################################################################

"""
Remove duplicate multiindices, then omit each multiindex `alpha` for which another multiindex `beta`
of the same length satisfies `beta[i] <= alpha[i]` for every `i`.
Construct a new list without modifying `shifts`.

Use the result only as an intermediate combinatorial object. 
This is not used to delete equations: divisibility of leader shifts does not imply that
one equation follows from another. For example,

`y = 0` and `y' + 1 = 0`

have shifts `[0]` and `[1]`, but deleting the second equation would turn an
inconsistent differential system into a consistent one.
Janet basis can still require proper multiples of these generators.
"""
function __minimal_shift_generators(shifts::AbstractVector{<:AbstractVector{<:Integer}})
    isempty(shifts) && return Vector{Int}[]
    # Deduplicate by hashing, retaining the first occurrence and its order.
    tuples = unique!(Tuple{Vararg{Int}}[Tuple(Int(component) for component in shift) for shift in shifts])
    generators = Vector{Int}[]
    sizehint!(generators, length(tuples))
    for i in eachindex(tuples)
        alpha = tuples[i]
        redundant = false
        for j in eachindex(tuples)
            i == j && continue
            beta = tuples[j]
            length(alpha) == length(beta) || continue
            divides = true
            @inbounds for k in eachindex(alpha, beta)
                if beta[k] > alpha[k]
                    divides = false
                    break
                end
            end
            if divides
                redundant = true
                break
            end
        end
        redundant || push!(generators, collect(alpha))
    end
    return generators
end

"""
Return the Janet completion of the divisibility-minimal generators of the
monomial ideal generated by `shifts`.

Use `__minimal_shift_generators` to remove monomial redundancies from an auxiliary list,
then add the multiples required for Janet completeness.
Preserve the input and do not remove or replace any equation.
Use the result as the expected set of shifts for a minimality check, not as
a list of equations to retain in the original system.

This preprocessing is kept separated from `janet_completion`: that function
completes the supplied set by adding missing monomials, without removing
its original elements.
"""
function __minimal_janet_completion(shifts::AbstractVector{<:AbstractVector{<:Integer}})
    isempty(shifts) && return Vector{Int}[]
    generators = __minimal_shift_generators(shifts)
    return janet_completion(generators)
end

"""
Test whether the leader shifts in an `__ActionJanetData` object form a
minimal Janet basis, separately for each differential indeterminate.\
Use this check when certifying a terminal differential Thomas branch.

For example, with two derivations and leader shifts `[2,0]` and `[0,1]` for the same unknown `y`,
Janet completion also requires `[1,1]`; the first direction is nonmultiplicative for `[0,1]`,
so its Janet cone does not cover `[1,1]`.
Thus accept `{[2,0], [0,1], [1,1]}` as minimal: the equation with leader `y[1,1]` must not be rejected
merely because this leader is a derivative of `y[0,1]`.\
Check passivity separately to verify the required differential consequences;
this helper checks only the "leader geometry".

For an example of a failed check (in characteristic zero) consider `y^2 + 1 = 0`
and `y' = 0`.
Report the nonminimal shifts `[0]` and `[1]`, independently of the degree in `y`.
Differentiate the first equation to obtain `2*y*y' = 0`.
Use `y^2 + 1 = 0` to certify `y != 0`, so the separant `2*y` is nonzero on the branch.
Then reduce `y'` to zero using this prolongation and its nonzero multiplier;
retain only `y^2 + 1 = 0`.
Justify the removal by this differential consequence, not by the shift comparison or by assuming that
`y = 0` is an equation of the system.

For a case requiring branching, take two differential unknowns `x < y` with one derivation,
and equations `x*y^2 + x*y = 0` and `y' = 0`.
Split on the initial `x` before discarding any equation.
On `x = 0`, the first equation vanishes identically; retain `y' = 0`, since it still constrains `y`.
Check minimality separately for `x` and `y`: their remaining leader shifts are `[0]` and `[1]`,
respectively, not two shifts belonging to the same unknown.
On `x != 0`, replace the first equation by `y*(y + 1) = 0`.
Differentiate to obtain `(2*y + 1)*y' = 0`; the separant is nonzero at both roots `y = 0` and `y = -1`,
so `y' = 0` is now a consequence.
With factorization, represent these cases by the disjoint branches

`{x = 0, y' = 0}`, `{x != 0, y = 0}` and `{x != 0, y + 1 = 0}`.

Do not interpret nonminimal leader shifts in the original input as grounds
to delete `y' = 0` before resolving the vanishing initial.

Report a failed comparison with `minimal=false`, `current` and `expected`;
do not delete equations or modify the Janet data.
Treat `expected` only as a combinatorial reference, not as permission to discard equations whose
shifts are absent from it.
Reprocess such equations using valid differential reductions on the branch: a remainder can vanish,
impose a new condition, or reveal a contradiction.
Respect Janet admissibility and the nonvanishing conditions on reduction multipliers
before discarding a zero remainder.
"""
function __differential_minimality_status(data::__ActionJanetData)
    return __differential_minimality_status(data.shifts, data.by_symbol)
end

function __differential_minimality_status(shifts, by_symbol)
    for symbol in eachindex(by_symbol)
        indices = by_symbol[symbol]
        length(indices) <= 1 && continue
        current_shifts = view(shifts, indices)
        expected_shifts = __minimal_janet_completion(current_shifts)
        current = Set(Tuple(shift) for shift in current_shifts)
        expected = Set(Tuple(shift) for shift in expected_shifts)
        if length(current) != length(current_shifts) || current != expected
            return (minimal=false, symbol=symbol, current=copy.(current_shifts), expected=expected_shifts)
        end
    end
    return (minimal=true, symbol=0, current=Vector{Int}[], expected=Vector{Int}[])
end
