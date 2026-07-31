###############################################################################
#####  Finite algebraic conversion of action polynomial rings
###############################################################################

using Oscar

# Quasi-Simple-Difference.jl and Quasi-Simple-Differential.jl uses this marker to make sure that the conversion layer is loaded first.
const __ACTION_POLY_CONVERSION_LOADED = true

###############################################################################
#####  Internal helpers
###############################################################################

function __conversion_variable_symbols(
    number_of_variables::Integer;
    variable_name::Union{Symbol,AbstractString}=:x,
)
    prefix = string(variable_name)
    return [Symbol(prefix, i) for i in 1:number_of_variables]
end

function __export_conversion_variables!(
    variables::AbstractVector;
    variable_name::Union{Symbol,AbstractString}=:x,
    target_module::Module=Main,
)
    prefix = string(variable_name)
    for (i, variable) in enumerate(variables)
        Core.eval(
            target_module,
            :($(Symbol(prefix, i)) = $(QuoteNode(variable))),
        )
    end
    return nothing
end

function __conversion_evaluate_action_polynomial(
    polynomial,
    source_ring,
    images::AbstractVector,
)
    getfield(polynomial, :parent) === source_ring ||
        error("The polynomial belongs to a different action polynomial ring")

    raw_polynomial = data(data(polynomial))
    number_of_raw_variables = ngens(parent(raw_polynomial))
    number_of_raw_variables <= length(images) ||
        error("The conversion data do not contain all variables of the polynomial")

    return evaluate(
        raw_polynomial,
        images[1:number_of_raw_variables],
    )
end

function __conversion_convert_collection(
    polynomials,
    conversion,
    converter_name::Symbol,
)
    converter = getfield(@__MODULE__, converter_name)
    return elem_type(conversion.ring)[
        converter(polynomial, conversion)
        for polynomial in polynomials
    ]
end

###############################################################################
#####  Difference variables in a finite ranking segment
###############################################################################

function __conversion_nonnegative_compositions(
    degree::Integer,
    length_of_composition::Integer,
)
    degree = Int(degree)
    length_of_composition = Int(length_of_composition)
    length_of_composition > 0 ||
        error("$length_of_composition is not a positive integer")
    degree < 0 && return Vector{Vector{Int}}()

    compositions = [
        reverse!(collect(Int, composition))
        for composition in weak_compositions(degree, length_of_composition)
    ]
    sort!(compositions)
    return compositions
end

function __conversion_partition_position(
    R::DifferencePolyRing,
    elementary_symbol::Integer,
)
    for (position, block) in enumerate(partition(ranking(R)))
        isone(block[elementary_symbol]) && return position
    end
    error("The elementary symbol is not contained in the ranking partition")
end

function __conversion_difference_jet_data(
    variable::DifferencePolyRingElem,
)
    R = getfield(variable, :parent)
    variable_to_jet = getfield(R, :var_to_jet)
    haskey(variable_to_jet, variable) ||
        error("$variable is not a difference variable")
    elementary_symbol, shift = variable_to_jet[variable]
    return (
        elementary_symbol=elementary_symbol,
        shift=collect(Int, shift),
    )
end

function __conversion_shift_or_copy(
    polynomial::DifferencePolyRingElem,
    shift::AbstractVector{<:Integer},
)
    multiplicities = collect(Int, shift)
    return all(iszero, multiplicities) ?
           polynomial + getfield(polynomial, :parent)(0) :
           diff_action(polynomial, multiplicities)
end

"""
Return all difference variables not greater than `maximum_variable`, in
increasing ranking order.

This is the conversion-private counterpart of `var_list_complete`.  Keeping
it here makes `Conversion.jl` usable before `Quasi-Simple-Difference.jl`.
"""
function __conversion_complete_difference_variables(
    maximum_variable::DifferencePolyRingElem{T},
) where {T}
    maximum_data = __conversion_difference_jet_data(maximum_variable)
    R = getfield(maximum_variable, :parent)
    number_of_actions = n_action_maps(R)
    zero_shift = zeros(Int, number_of_actions)
    elementary_variables = [
        gen(R, i, zero_shift)
        for i in 1:n_elementary_symbols(R)
    ]

    variables = DifferencePolyRingElem{T}[maximum_variable]
    maximum_degree = sum(maximum_data.shift)

    for degree in 0:(maximum_degree - 1)
        shifts = __conversion_nonnegative_compositions(
            degree,
            number_of_actions,
        )
        for elementary_variable in elementary_variables
            append!(
                variables,
                (
                    __conversion_shift_or_copy(elementary_variable, shift)
                    for shift in shifts
                ),
            )
        end
    end

    ordering_matrix = index_ordering_matrix(ranking(R))
    maximum_block = __conversion_partition_position(
        R,
        maximum_data.elementary_symbol,
    )
    ordered_maximum_shift = ordering_matrix * maximum_data.shift
    boundary_shifts = __conversion_nonnegative_compositions(
        maximum_degree,
        number_of_actions,
    )

    for elementary_variable in elementary_variables
        elementary_data =
            __conversion_difference_jet_data(elementary_variable)
        elementary_block = __conversion_partition_position(
            R,
            elementary_data.elementary_symbol,
        )

        if elementary_block < maximum_block
            append!(
                variables,
                (
                    __conversion_shift_or_copy(elementary_variable, shift)
                    for shift in boundary_shifts
                ),
            )
        elseif elementary_block == maximum_block
            for shift in boundary_shifts
                if shift == maximum_data.shift
                    if elementary_data.elementary_symbol >
                       maximum_data.elementary_symbol
                        push!(
                            variables,
                            __conversion_shift_or_copy(
                                elementary_variable,
                                shift,
                            ),
                        )
                    end
                elseif ordering_matrix * shift < ordered_maximum_shift
                    push!(
                        variables,
                        __conversion_shift_or_copy(
                            elementary_variable,
                            shift,
                        ),
                    )
                end
            end
        end
    end

    return sort!(variables)
end

###############################################################################
#####  Difference conversion data
###############################################################################

"""
Data for evaluating difference polynomials in a finite ordinary polynomial
ring.  The `positions` and `converted_positions` fields are retained for the
algebraic splitting code.
"""
struct __DifferenceAlgebraicData{
    T,
    SourceRing,
    AlgebraicRing,
    AlgebraicElement,
}
    source_ring::SourceRing
    ring::AlgebraicRing
    variables::Vector{AlgebraicElement}
    diff_vars::Vector{DifferencePolyRingElem{T}}
    images::Vector{AlgebraicElement}
    positions::Dict{DifferencePolyRingElem{T},Int}
    converted_positions::Dict{Int,Int}
end

function __conversion_parent_from_polys(
    polynomials::AbstractVector{<:DifferencePolyRingElem},
)
    isempty(polynomials) &&
        error("The list of difference polynomials must be nonempty")
    R = getfield(first(polynomials), :parent)
    any(p -> getfield(p, :parent) !== R, polynomials) &&
        error("All difference polynomials must belong to the same polynomial ring")
    return R
end

function __conversion_max_variable(
    R::DifferencePolyRing{T},
    polynomials::Union{
        Tuple{},
        AbstractVector{<:DifferencePolyRingElem{T}},
    }=(),
) where {T}
    leaders = DifferencePolyRingElem{T}[]
    for polynomial in polynomials
        is_constant(polynomial) || push!(leaders, leader(polynomial))
    end
    isempty(leaders) && append!(leaders, gens(R))
    isempty(leaders) &&
        error("The difference polynomial ring has no generators")
    return maximum(leaders)
end

function __conversion_data(
    R::DifferencePolyRing{T},
    polynomials::Union{
        Tuple{},
        AbstractVector{<:DifferencePolyRingElem{T}},
    }=();
    variable_name::Union{Symbol,AbstractString}=:x,
) where {T}
    maximum_variable = __conversion_max_variable(R, polynomials)
    difference_variables = reverse(
        __conversion_complete_difference_variables(maximum_variable),
    )
    variable_symbols = __conversion_variable_symbols(
        length(difference_variables);
        variable_name,
    )
    algebraic_ring, algebraic_variables =
        polynomial_ring(coefficient_ring(R), variable_symbols)

    images = [algebraic_ring(0) for _ in 1:ngens(base_ring(R))]
    positions = Dict{DifferencePolyRingElem{T},Int}()
    for variable in gens(R)
        raw_position = var_index(data(data(variable)))
        positions[variable] = raw_position
    end

    converted_positions = Dict{Int,Int}()
    for i in eachindex(difference_variables)
        raw_position = var_index(data(data(difference_variables[i])))
        images[raw_position] = algebraic_variables[i]
        converted_positions[raw_position] = i
    end

    return __DifferenceAlgebraicData{
        T,
        typeof(R),
        typeof(algebraic_ring),
        elem_type(algebraic_ring),
    }(
        R,
        algebraic_ring,
        algebraic_variables,
        difference_variables,
        images,
        positions,
        converted_positions,
    )
end

function __convert_difference_poly(
    polynomial::DifferencePolyRingElem{T},
    conversion::__DifferenceAlgebraicData{T},
) where {T}
    return __conversion_evaluate_action_polynomial(
        polynomial,
        conversion.source_ring,
        conversion.images,
    )
end

# Compatibility signature used by Splitting.jl.
function __convert_difference_poly(
    polynomial::DifferencePolyRingElem{T},
    algebraic_ring,
    difference_variables::AbstractVector{<:DifferencePolyRingElem{T}},
    converted_positions::Dict{Int,Int},
) where {T}
    R = getfield(polynomial, :parent)
    any(v -> getfield(v, :parent) !== R, difference_variables) &&
        error("All difference variables must belong to the parent of the polynomial")
    length(difference_variables) == length(converted_positions) ||
        error("The conversion positions do not match the difference variables")

    algebraic_variables = gens(algebraic_ring)
    images = [algebraic_ring(0) for _ in 1:ngens(base_ring(R))]
    for (raw_position, algebraic_position) in converted_positions
        images[raw_position] = algebraic_variables[algebraic_position]
    end
    return __conversion_evaluate_action_polynomial(
        polynomial,
        R,
        images,
    )
end

@doc raw"""
    conversion_difference_algebraic(R; variable_name=:x, ...)
    conversion_difference_algebraic(p; variable_name=:x, ...)
    conversion_difference_algebraic(F; variable_name=:x, ...)
    conversion_difference_algebraic(p, reference; variable_name=:x, ...)
    conversion_difference_algebraic(F, reference; variable_name=:x, ...)

Convert a finite part of a difference polynomial ring to an ordinary
multivariate polynomial ring over the same coefficient ring.

The ordinary variables follow the difference ranking.  The finite part is the
complete initial ranking segment ending at the greatest leader needed by the
input.  Conversion evaluates OSCAR's ordinary representation
`data(data(p))`; it does not reconstruct coefficients or exponent vectors.

The one-argument vector method is retained for compatibility and returns the
target ring and its variables.  To convert every polynomial in a vector, pass
the vector also as the reference argument.

By default, algebraic variables are exported to `target_module`, as in the original API. 
Set `export_variables=false` to avoid creating bindings.

# Examples

```jldoctest
julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name=:deglex);
julia> A, x = conversion_difference_algebraic(R; export_variables=false);
julia> f = diff_action(y2, [1, 1]);
julia> f_alg, B, z = conversion_difference_algebraic(f, [y1, y2]; variable_name=:z, export_variables=false);
julia> (f_alg, length(z), parent(f_alg) === B)
(z1, 10, true)
```
"""
function conversion_difference_algebraic(
    R::DifferencePolyRing{T};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=true,
    target_module::Module=Main,
) where {T}
    conversion = __conversion_data(R; variable_name)
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    return conversion.ring, conversion.variables
end

function conversion_difference_algebraic(
    polynomial::DifferencePolyRingElem{T};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=true,
    target_module::Module=Main,
) where {T}
    return conversion_difference_algebraic(
        polynomial,
        DifferencePolyRingElem{T}[];
        variable_name,
        export_variables,
        target_module,
    )
end

function conversion_difference_algebraic(
    polynomials::AbstractVector{<:DifferencePolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=true,
    target_module::Module=Main,
) where {T}
    collected_polynomials = collect(polynomials)
    R = __conversion_parent_from_polys(collected_polynomials)
    conversion = __conversion_data(
        R,
        collected_polynomials;
        variable_name,
    )
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    return conversion.ring, conversion.variables
end

function conversion_difference_algebraic(
    polynomial::DifferencePolyRingElem{T},
    reference::AbstractVector{<:DifferencePolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=true,
    target_module::Module=Main,
) where {T}
    collected_reference = collect(reference)
    R = getfield(polynomial, :parent)
    any(q -> getfield(q, :parent) !== R, collected_reference) &&
        error("All difference polynomials must have the same parent")

    conversion = __conversion_data(
        R,
        [polynomial; collected_reference];
        variable_name,
    )
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    return (
        __convert_difference_poly(polynomial, conversion),
        conversion.ring,
        conversion.variables,
    )
end

function conversion_difference_algebraic(
    polynomials::AbstractVector{<:DifferencePolyRingElem{T}},
    reference::AbstractVector{<:DifferencePolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=true,
    target_module::Module=Main,
) where {T}
    collected_polynomials = collect(polynomials)
    collected_reference = collect(reference)
    isempty(collected_polynomials) && isempty(collected_reference) &&
        error("At least one difference polynomial is required")

    R = if isempty(collected_polynomials)
        getfield(first(collected_reference), :parent)
    else
        getfield(first(collected_polynomials), :parent)
    end
    any(p -> getfield(p, :parent) !== R, collected_polynomials) &&
        error("All difference polynomials must have the same parent")
    any(p -> getfield(p, :parent) !== R, collected_reference) &&
        error("All difference polynomials must have the same parent")

    conversion::__DifferenceAlgebraicData{T} = __conversion_data(
        R,
        [collected_polynomials; collected_reference];
        variable_name,
    )
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    converted = __conversion_convert_collection(
        collected_polynomials,
        conversion,
        :__convert_difference_poly,
    )
    return converted, conversion.ring, conversion.variables
end

"""
Return `true` when the data-based difference conversion layer is loaded.
"""
difference_conversion_uses_internal_data() = true

###############################################################################
#####  Differential conversion data
###############################################################################

"""
Data for evaluating differential polynomials in a finite ordinary polynomial
ring ordered by the differential ranking.
"""
struct __DifferentialAlgebraicData{
    T,
    SourceRing,
    AlgebraicRing,
    AlgebraicElement,
}
    source_ring::SourceRing
    ring::AlgebraicRing
    variables::Vector{AlgebraicElement}
    differential_variables::Vector{DifferentialPolyRingElem{T}}
    images::Vector{AlgebraicElement}
    positions::Dict{DifferentialPolyRingElem{T},Int}
    converted_positions::Dict{Int,Int}
end

function __differential_parent(
    polynomials::AbstractVector{<:DifferentialPolyRingElem},
)
    isempty(polynomials) &&
        error("At least one differential polynomial is required")
    R = getfield(first(polynomials), :parent)
    any(p -> getfield(p, :parent) !== R, polynomials) &&
        error("All differential polynomials must have the same parent")
    return R
end

function __differential_conversion_variables(
    R::DifferentialPolyRing{T},
    polynomials::AbstractVector{<:DifferentialPolyRingElem{T}},
) where {T}
    differential_variables = DifferentialPolyRingElem{T}[]
    for polynomial in polynomials
        append!(differential_variables, vars(polynomial))
    end
    unique!(differential_variables)
    sort!(differential_variables; rev=true)

    # A constant polynomial still has a well-defined conversion.  In that
    # case use the currently constructed generators of its parent ring.
    isempty(differential_variables) &&
        append!(differential_variables, gens(R))
    isempty(differential_variables) &&
        error("The differential polynomial ring has no generators")
    return differential_variables
end

function __build_differential_algebraic_conversion_data(
    R::DifferentialPolyRing{T},
    differential_variables::Vector{DifferentialPolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
) where {T}
    variable_symbols = __conversion_variable_symbols(
        length(differential_variables);
        variable_name,
    )
    algebraic_ring, algebraic_variables =
        polynomial_ring(coefficient_ring(R), variable_symbols)

    images = [algebraic_ring(0) for _ in 1:ngens(base_ring(R))]
    positions = Dict{DifferentialPolyRingElem{T},Int}()
    converted_positions = Dict{Int,Int}()
    for i in eachindex(differential_variables)
        variable = differential_variables[i]
        raw_position = var_index(data(data(variable)))
        images[raw_position] = algebraic_variables[i]
        positions[variable] = raw_position
        converted_positions[raw_position] = i
    end

    return __DifferentialAlgebraicData{
        T,
        typeof(R),
        typeof(algebraic_ring),
        elem_type(algebraic_ring),
    }(
        R,
        algebraic_ring,
        algebraic_variables,
        differential_variables,
        images,
        positions,
        converted_positions,
    )
end

function __differential_algebraic_conversion_data(
    R::DifferentialPolyRing{T};
    variable_name::Union{Symbol,AbstractString}=:x,
) where {T}
    differential_variables =
        collect(DifferentialPolyRingElem{T}, gens(R))
    sort!(differential_variables; rev=true)
    isempty(differential_variables) &&
        error("The differential polynomial ring has no generators")
    return __build_differential_algebraic_conversion_data(
        R,
        differential_variables;
        variable_name,
    )
end

function __differential_algebraic_conversion_data(
    polynomials::AbstractVector{<:DifferentialPolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
) where {T}
    collected_polynomials = collect(polynomials)
    R = __differential_parent(collected_polynomials)
    differential_variables = __differential_conversion_variables(
        R,
        collected_polynomials,
    )
    return __build_differential_algebraic_conversion_data(
        R,
        differential_variables;
        variable_name,
    )
end

function __convert_differential_polynomial(
    polynomial::DifferentialPolyRingElem{T},
    conversion::__DifferentialAlgebraicData{T},
) where {T}
    return __conversion_evaluate_action_polynomial(
        polynomial,
        conversion.source_ring,
        conversion.images,
    )
end

@doc raw"""
    conversion_differential_algebraic(R; variable_name=:x, ...)
    conversion_differential_algebraic(p; variable_name=:x, ...)
    conversion_differential_algebraic(F; variable_name=:x, ...)
    conversion_differential_algebraic(p, reference; variable_name=:x, ...)
    conversion_differential_algebraic(F, reference; variable_name=:x, ...)

Convert a differential polynomial ring, a polynomial, or a finite family of
    polynomials to an ordinary multivariate polynomial ring over the same coefficient ring.

For a polynomial family, the target contains exactly the differential variables occurring in the family, in decreasing differential-ranking order.
For a ring, it contains all currently constructed differential variables. Conversion uses OSCAR's ordinary representation `data(data(p))`.

Differential conversions do not export algebraic variables by default.
Set `export_variables=true` when REPL bindings such as `x1`, ..., `xn` are useful.

# Examples

An ordinary differential polynomial ring over `QQ(t)`:

```jldoctest
julia> K, t = rational_function_field(QQ, "t");
julia> R, u = differential_polynomial_ring(K, :u, [c -> derivative(c)]);
julia> du = diff_action(u, 1);
julia> A, x = conversion_differential_algebraic(R);
julia> p = t*du + u^2;
julia> p_alg, B, z = conversion_differential_algebraic(p, [u]; variable_name=:z);
julia> (p_alg, parent(p_alg) === B)
(t*z1 + z2^2, true)
```

Two commuting derivations:

```jldoctest
julia> R, (u, v) = differential_polynomial_ring(QQ, [:u, :v], 2; index_ordering_name=:deglex);
julia> F = [diff_action(u, [1, 0])^2 + v, diff_action(v, [0, 1]) - u];
julia> F_alg, A, x = conversion_differential_algebraic(F);
julia> (F_alg, length(x), all(p -> parent(p) === A, F_alg))
(QQMPolyRingElem[x1^2 + x4, x2 - x3], 4, true)
```
"""
function conversion_differential_algebraic(
    R::DifferentialPolyRing{T};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=false,
    target_module::Module=Main,
) where {T}
    conversion =
        __differential_algebraic_conversion_data(R; variable_name)
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    return conversion.ring, conversion.variables
end

function conversion_differential_algebraic(
    polynomial::DifferentialPolyRingElem{T};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=false,
    target_module::Module=Main,
) where {T}
    return conversion_differential_algebraic(
        polynomial,
        DifferentialPolyRingElem{T}[];
        variable_name,
        export_variables,
        target_module,
    )
end

function conversion_differential_algebraic(
    polynomials::AbstractVector{<:DifferentialPolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=false,
    target_module::Module=Main,
) where {T}
    collected_polynomials = collect(polynomials)
    conversion::__DifferentialAlgebraicData{T} =
        __differential_algebraic_conversion_data(
            collected_polynomials;
            variable_name,
        )
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    converted = __conversion_convert_collection(
        collected_polynomials,
        conversion,
        :__convert_differential_polynomial,
    )
    return converted, conversion.ring, conversion.variables
end

function conversion_differential_algebraic(
    polynomial::DifferentialPolyRingElem{T},
    reference::AbstractVector{<:DifferentialPolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=false,
    target_module::Module=Main,
) where {T}
    collected_reference = collect(reference)
    R = getfield(polynomial, :parent)
    any(q -> getfield(q, :parent) !== R, collected_reference) &&
        error("All differential polynomials must have the same parent")
    conversion = __differential_algebraic_conversion_data(
        [polynomial; collected_reference];
        variable_name,
    )
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    return (
        __convert_differential_polynomial(polynomial, conversion),
        conversion.ring,
        conversion.variables,
    )
end

function conversion_differential_algebraic(
    polynomials::AbstractVector{<:DifferentialPolyRingElem{T}},
    reference::AbstractVector{<:DifferentialPolyRingElem{T}};
    variable_name::Union{Symbol,AbstractString}=:x,
    export_variables::Bool=false,
    target_module::Module=Main,
) where {T}
    collected_polynomials = collect(polynomials)
    collected_reference = collect(reference)
    isempty(collected_polynomials) && isempty(collected_reference) &&
        error("At least one differential polynomial is required")

    R = if isempty(collected_polynomials)
        getfield(first(collected_reference), :parent)
    else
        getfield(first(collected_polynomials), :parent)
    end
    any(p -> getfield(p, :parent) !== R, collected_polynomials) &&
        error("All differential polynomials must have the same parent")
    any(p -> getfield(p, :parent) !== R, collected_reference) &&
        error("All differential polynomials must have the same parent")

    conversion::__DifferentialAlgebraicData{T} =
        __differential_algebraic_conversion_data(
            [collected_polynomials; collected_reference];
            variable_name,
        )
    export_variables && __export_conversion_variables!(
        conversion.variables;
        variable_name,
        target_module,
    )
    converted = __conversion_convert_collection(
        collected_polynomials,
        conversion,
        :__convert_differential_polynomial,
    )
    return converted, conversion.ring, conversion.variables
end

"""
Return `true` when the data-based differential conversion layer is loaded.
"""
differential_conversion_uses_internal_data() = true