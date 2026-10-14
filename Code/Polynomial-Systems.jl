abstract type AbstractPolynomialSystem end

"""
Store the equality/hash key, separately from the system's data and certificates.
Key polynomials may be shared with the system's data, but both must remain unchanged.
Return working copies.
"""
struct __PolynomialSystemKey{R,P}
    ring::R
    equations::Tuple{Vararg{P}}
    inequations::Tuple{Vararg{P}}
end

"""
Compare polynomial system keys by ring identity and unordered equations and inequations
"""
function Base.isequal(a::__PolynomialSystemKey, b::__PolynomialSystemKey)
    a === b && return true
    a.ring === b.ring || return false
    for (left, right) in ((a.equations, b.equations), (a.inequations, b.inequations))
        length(left) == length(right) || return false
        all(p -> any(q -> isequal(p, q), right), left) || return false
    end
    return true
end

"""
Shortcut for isequal
    """
Base.:(==)(a::__PolynomialSystemKey, b::__PolynomialSystemKey) = isequal(a, b)

"""
Compute a hash for polynomial system keys, independent of the order within equations and inequations.
"""
function Base.hash(key::__PolynomialSystemKey, h::UInt)
    equations_hash = foldl(xor, (hash(p) for p in key.equations); init=zero(UInt))
    inequations_hash = foldl(xor, (hash(p) for p in key.inequations); init=zero(UInt))
    return hash((objectid(key.ring), length(key.equations), equations_hash,
                 length(key.inequations), inequations_hash), h)
end

"""
Build an order-independent key. By default, its polynomials are copied from the input.
Set copy_polynomials=false only when using system data that will remain unchanged.
The function `normalize` must not modify its arguments.
"""
function __polynomial_system_key(
    R::Union{Nothing,Ring},
    E::AbstractVector{P},
    I::AbstractVector{P};
    normalize=identity,
    copy_polynomials::Bool=true
    ) where P
    R === nothing || __require_parent(R, E, I)
    prepare = copy_polynomials ? (p -> deepcopy(normalize(p))) : normalize
    equations = unique!([prepare(p) for p in E])
    inequations = unique!([prepare(p) for p in I])
    return __PolynomialSystemKey{typeof(R),P}(R, Tuple(equations), Tuple(inequations))
end

###############################################################################
#####  Algebraic systems
###############################################################################

@doc raw"""
    AlgebraicSystem(R, equations, inequations)
    AlgebraicSystem(R, v=[]; equations=true)
    AlgebraicSystem(equations, inequations)
    AlgebraicSystem(v; equations=true)
    AlgebraicSystem(system::AlgebraicSystem)

Let R be a polynomial ring, and let equations, inequations be vectors of elements
of R. Represent the algebraic system

p = 0 for p in equations
q ≠ 0 for q in inequations

in R using a struct; if both vectors are empty, pass R explicitly.
If only one vector is provided, the optional argument equations=true treats it as a vector
of equations. Set equations=false to construct an algebraic system consisting only of inequations.
Every new system, including a copy, starts with the fields quasi_simple and simple equal to nothing.
The functions is_quasi_simple and is_simple each update only their own certificate.
The other certificate is left unchanged, even when its value follows from the result.
Previously computed certificates may be reused to avoid repeated checks.
Decomposition algorithms may certify both properties of their output systems.

Check the common parent for the elements of the input vectors.
Construct a new system to change its equations or inequations.

    # Examples

    ```jldoctest
    julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
    julia> system = AlgebraicSystem([x-y]);
    julia> system.quasi_simple === nothing
    true
    julia> system.simple === nothing
    true
    julia> is_quasi_simple(system)
    true
    julia> system.quasi_simple
    true
    julia> system.simple === nothing
    true
    julia> is_simple(system)
    true
    julia> system.simple
    true
    julia> is_simple(system)
    true

    julia> system = AlgebraicSystem([x^2-y]);
    julia> is_simple(system)
    false
    julia> system.simple === false
    true
    julia> system.quasi_simple === nothing
    true
    julia> is_quasi_simple(system)
    true
    julia> system.quasi_simple === true
    true
    ```
"""
struct AlgebraicSystem{R, P<:MPolyRingElem} <: AbstractPolynomialSystem
    ring::R
    equations::Vector{P}
    inequations::Vector{P}
    quasi_simple::Base.RefValue{Union{Nothing,Bool}}
    simple::Base.RefValue{Union{Nothing,Bool}}
    _key::__PolynomialSystemKey{R,P}
    """
    Store the input without copying; all polynomials must belong to R.
    The vectors and their polynomials must remain unchanged.
    """
    function AlgebraicSystem(R::Ring, E::AbstractVector, I::AbstractVector)
        P = elem_type(R)
        P <: MPolyRingElem || throw(ArgumentError("An algebraic polynomial ring is required"))
        all(p -> p isa MPolyRingElem, E) && all(p -> p isa MPolyRingElem, I) ||
            throw(ArgumentError("Equations and inequations must be algebraic polynomials"))
        __require_parent(R, E, I)
        equations = P[deepcopy(p) for p in E]
        inequations = P[deepcopy(p) for p in I]
        key = __polynomial_system_key(R, equations, inequations;
            normalize=__algebraic_remove_constant_factor, copy_polynomials=false)
        return new{typeof(R),P}(R, equations, inequations,
            Ref{Union{Nothing,Bool}}(nothing), Ref{Union{Nothing,Bool}}(nothing), key)
    end
    """
    Internal constructor to acquire exclusively owned vectors and polynomials.
    """
    function AlgebraicSystem(::Val{:owned}, R::Ring, equations::Vector{P}, inequations::Vector{P}) where {P<:MPolyRingElem}
        key = __polynomial_system_key(
        R, equations, inequations;
        normalize=__algebraic_remove_constant_factor,
        copy_polynomials=false
        )
    return new{typeof(R),P}(R, equations, inequations,
        Ref{Union{Nothing,Bool}}(nothing), Ref{Union{Nothing,Bool}}(nothing), key)
    end
    """
    Internal constructor for the output of __algebraic_cleanup and __algebraic_normalized.
    The polynomials involved already have normalized coefficients, the lists contain no duplicates.
    Copy the input polynomials once and share them with the key.
    """
    function AlgebraicSystem(
        ::Val{:normalized},
        R::Ring,
        E::Vector{P},
        I::Vector{P}
    ) where {P<:MPolyRingElem}
        __require_parent(R, E, I)
        equations, inequations = P[deepcopy(p) for p in E], P[deepcopy(p) for p in I]
        key = __PolynomialSystemKey{typeof(R),P}(R, Tuple(equations), Tuple(inequations))
        return new{typeof(R),P}(R, equations, inequations,
            Ref{Union{Nothing,Bool}}(nothing), Ref{Union{Nothing,Bool}}(nothing), key)
    end
    """
    Copy the normalized key instead of normalizing the polynomials again.
    Both the system's data and the key polynomials are copied independently of
    the original system.
    Polynomials shared within the original system are still shared in the copy.
    """
    function AlgebraicSystem(system::AlgebraicSystem{R,P}) where {R,P}
        key = __algebraic_key(system)
        copies = IdDict{P,P}()
        copy_polynomial(p) = get!(() -> deepcopy(p), copies, p)
        equations = P[copy_polynomial(p) for p in getfield(system, :equations)]
        inequations = P[copy_polynomial(p) for p in getfield(system, :inequations)]
        copied_key = __PolynomialSystemKey{R,P}(system.ring,
            Tuple(copy_polynomial(p) for p in key.equations),
            Tuple(copy_polynomial(p) for p in key.inequations))
        return new{R,P}(system.ring, equations, inequations,
            Ref{Union{Nothing,Bool}}(nothing), Ref{Union{Nothing,Bool}}(nothing), copied_key)
    end
end

function AlgebraicSystem(R::Ring, v::AbstractVector=elem_type(R)[];
        equations::Bool=true)
    return equations ? AlgebraicSystem(R, v, elem_type(R)[]) :
                       AlgebraicSystem(R, elem_type(R)[], v)
end

function AlgebraicSystem(v::AbstractVector;
        equations::Bool=true)
    isempty(v) &&
        throw(ArgumentError("Pass the polynomial ring explicitly for an empty system"))
    return AlgebraicSystem(parent(first(v)), v; equations)
end

function AlgebraicSystem(E::AbstractVector, I::AbstractVector)
    isempty(E) && isempty(I) &&
        throw(ArgumentError("Pass the polynomial ring explicitly for an empty system"))
    p = isempty(E) ? first(I) : first(E)
    return AlgebraicSystem(parent(p), E, I)
end

"""
Default deepcopy preserves the certificates; every new system starts uncertified.
"""
Base.copy(system::AlgebraicSystem) = AlgebraicSystem(system)
function Base.deepcopy_internal(system::AlgebraicSystem, dict::IdDict)
    return get!(dict, system) do
        AlgebraicSystem(system)
    end
end

"""
Return independent copies of equations and inequations.
Extract the values of quasi_simple and simple from their references.
"""
function Base.getproperty(system::AlgebraicSystem, name::Symbol)
    name === :equations && return deepcopy(getfield(system, :equations))
    name === :inequations && return deepcopy(getfield(system, :inequations))
    name === :quasi_simple && return getfield(system, :quasi_simple)[]
    name === :simple && return getfield(system, :simple)[]
    return getfield(system, name)
end

"""
Include the key only when private properties are requested.
"""
Base.propertynames(::AlgebraicSystem, private::Bool=false) = private ?
    (:ring, :equations, :inequations, :quasi_simple, :simple, :_key) :
    (:ring, :equations, :inequations, :quasi_simple, :simple)

"""
Compare algebraic systems using their keys.
"""
Base.isequal(a::AlgebraicSystem, b::AlgebraicSystem) = isequal(__algebraic_key(a), __algebraic_key(b))
Base.:(==)(a::AlgebraicSystem, b::AlgebraicSystem) = isequal(a, b)

"""
Use AlgebraicSystem with Set e Dict
"""
Base.hash(system::AlgebraicSystem, h::UInt) = hash(__algebraic_key(system), h)

###############################################################################
#####  Action Systems
###############################################################################

abstract type AbstractActionSystem <: AbstractPolynomialSystem end

"""
Store polynomial data and certificates for both action-system types.
Certificates are stored in a dictionary to take into account properties specific to
each system type. The key uses the same __PolynomialSystemKey representation as
AlgebraicSystem. Only certificates may change when their values are computed and cached.
"""
struct __ActionSystemData{R,P}
    ring::R
    equations::Vector{P}
    inequations::Vector{P}
    certificates::Dict{Symbol,Bool}
    key::__PolynomialSystemKey{R,P}
end

function __action_system_data(
    R::ActionPolyRing,
    E::AbstractVector,
    I::AbstractVector)
    P = elem_type(R)
    all(p -> p isa P, E) && all(p -> p isa P, I) ||
        throw(ArgumentError("Equations and inequations must be polynomials of the supplied action ring"))
    __require_parent(R, E, I)
    equations = P[deepcopy(p) for p in E]
    inequations = P[deepcopy(p) for p in I]
    key = __polynomial_system_key(R, equations, inequations; copy_polynomials=false)
    return __ActionSystemData(R, equations, inequations, Dict{Symbol,Bool}(), key)
end

"""
Copy the action system's data and key without recomputing the key.
Polynomials shared between the system's data and the key are still shared in the copy.
"""
function __copy_action_system_data(data::__ActionSystemData{R,P}) where {R,P}
    copies = IdDict{P,P}()
    copy_polynomial(p) = get!(() -> deepcopy(p), copies, p)
    equations = P[copy_polynomial(p) for p in data.equations]
    inequations = P[copy_polynomial(p) for p in data.inequations]
    key = __PolynomialSystemKey{R,P}(data.ring,
        Tuple(copy_polynomial(p) for p in data.key.equations),
        Tuple(copy_polynomial(p) for p in data.key.inequations))
    return __ActionSystemData(data.ring, equations, inequations, Dict{Symbol,Bool}(), key)
end

"""
Access the key stored in an action system.
"""
__action_system_key(system::AbstractActionSystem) = getfield(system, :_data).key

Base.propertynames(system::AbstractActionSystem, private::Bool=false) =
    (:ring, :equations, :inequations, __system_certificate_names(system)..., (private ? (:_data,) : ())...)

@doc raw"""
    DifferentialSystem(R, equations, inequations)
    DifferentialSystem(R, v=[]; equations=true)
    DifferentialSystem(equations, inequations)
    DifferentialSystem(v; equations=true)
    DifferentialSystem(system::DifferentialSystem)

Let R be a differential polynomial ring, and let equations, inequations be vectors of elements
of R. Represent the differential system

p = 0 for p in equations
q ≠ 0 for q in inequations

in R using a struct; if both vectors are empty, pass R explicitly.
If only one vector is provided, the optional argument equations=true treats it as a vector
of equations. Set equations=false to construct a differential system consisting only of inequations.
Every new system, including a copy, starts with the fields quasi_simple and simple equal to nothing.
The functions is_quasi_simple and is_simple each update only their own certificate.
The other certificate is left unchanged, even when its value follows from the result.
Previously computed certificates may be reused to avoid repeated checks.
Decomposition algorithms may certify both properties of their output systems.

Check the common parent for the elements of the input vectors.
Construct a new system to change its equations and inequations.

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 1);
    julia> system = DifferentialSystem([u^2]);
    julia> system.quasi_simple === nothing
    true
    julia> system.simple === nothing
    true
    julia> is_quasi_simple(system)
    true
    julia> system.simple === nothing
    true
    julia> is_simple(system)
    false
    julia> system.simple
    false
    ```
"""
struct DifferentialSystem{R, P} <: AbstractActionSystem
    _data::__ActionSystemData{R,P}
    function DifferentialSystem(R::DifferentialPolyRing, E::AbstractVector, I::AbstractVector)
        data = __action_system_data(R, E, I)
        return new{typeof(R),elem_type(R)}(data)
    end
    """
    Copy the system's data and key without recomputing the key.
    Polynomials shared between the system's data and the key are still shared in the copy.
    """
    function DifferentialSystem(system::DifferentialSystem{R,P}) where {R,P}
        data = __copy_action_system_data(getfield(system, :_data))
        return new{R,P}(data)
    end
end

function DifferentialSystem(R::DifferentialPolyRing, v::AbstractVector=elem_type(R)[];
        equations::Bool=true)
    return equations ? DifferentialSystem(R, v, elem_type(R)[]) :
                       DifferentialSystem(R, elem_type(R)[], v)
end

function DifferentialSystem(v::AbstractVector;
        equations::Bool=true)
    isempty(v) &&
        throw(ArgumentError("Pass the differential polynomial ring explicitly for an empty system"))
    return DifferentialSystem(parent(first(v)), v; equations)
end

function DifferentialSystem(E::AbstractVector, I::AbstractVector)
    isempty(E) && isempty(I) &&
        throw(ArgumentError("Pass the differential polynomial ring explicitly for an empty system"))
    p = isempty(E) ? first(I) : first(E)
    return DifferentialSystem(parent(p), E, I)
end

@doc raw"""
    DifferenceSystem(R, equations, inequations)
    DifferenceSystem(R, v=[]; equations=true)
    DifferenceSystem(equations, inequations)
    DifferenceSystem(v; equations=true)
    DifferenceSystem(system::DifferenceSystem)

Let R be a difference polynomial ring, and let equations, inequations be vectors of elements
of R. Represent the difference system

p = 0 for p in equations
q ≠ 0 for q in inequations

in R using a struct; if both vectors are empty, pass R explicitly.
If only one vector is provided, the optional argument equations=true treats it as a vector
of equations. Set equations=false to construct a difference system consisting only of inequations.

Check the common parent for the elements of the input vectors.
Construct a new system to change its equations and inequations.

    ```jldoctest
    julia> R, u = difference_polynomial_ring(QQ, :u, 1);
    julia> system = DifferenceSystem([u[2]-u[1]-u])
    DifferenceSystem(1 equations, 0 inequations)
    julia> system = DifferenceSystem([u[2]-u[1]-u]; equations=false)
    DifferenceSystem(0 equations, 1 inequations)
    ```
"""
struct DifferenceSystem{R,P} <: AbstractActionSystem
    _data::__ActionSystemData{R,P}
    function DifferenceSystem(R::DifferencePolyRing, E::AbstractVector, I::AbstractVector)
        data = __action_system_data(R, E, I)
        return new{typeof(R),elem_type(R)}(data)
    end
    """
    Copy the system's data and key without recomputing the key.
    Polynomials shared between the system's data and the key are still shared in the copy.
    """
    function DifferenceSystem(system::DifferenceSystem{R,P}) where {R,P}
        data = __copy_action_system_data(getfield(system, :_data))
        return new{R,P}(data)
    end
end

function DifferenceSystem(R::DifferencePolyRing, v::AbstractVector=elem_type(R)[];
        equations::Bool=true)
    return equations ? DifferenceSystem(R, v, elem_type(R)[]) :
                       DifferenceSystem(R, elem_type(R)[], v)
end

function DifferenceSystem(v::AbstractVector;
        equations::Bool=true)
    isempty(v) &&
        throw(ArgumentError("Pass the difference polynomial ring explicitly for an empty system"))
    return DifferenceSystem(parent(first(v)), v; equations)
end

function DifferenceSystem(E::AbstractVector, I::AbstractVector)
    isempty(E) && isempty(I) &&
        throw(ArgumentError("Pass the polynomial ring explicitly for an empty system"))
    p = isempty(E) ? first(I) : first(E)
    return DifferenceSystem(parent(p), E, I)
end

__system_certificate_names(::DifferentialSystem) = (:quasi_simple, :simple)
__system_certificate_names(::DifferenceSystem) = ()

"""
Default deepcopy preserves the certificates; every new system starts uncertified.
"""
Base.copy(system::DifferentialSystem) = DifferentialSystem(system)
Base.copy(system::DifferenceSystem) = DifferenceSystem(system)
function Base.deepcopy_internal(system::AbstractActionSystem, dict::IdDict)
    return get!(dict, system) do
        copy(system)
    end
end

"""
Return independent copies of equations and inequations, so that modifying them
does not change the system or invalidate its certificates.
Return the stored values of the available certificates.
"""
function Base.getproperty(system::AbstractActionSystem, name::Symbol)
    data = getfield(system, :_data)
    name === :ring && return data.ring
    name === :equations && return deepcopy(data.equations)
    name === :inequations && return deepcopy(data.inequations)
    name in __system_certificate_names(system) && return get(data.certificates, name, nothing)
    return getfield(system, name)
end

"""
Return the system's polynomial vectors for read-only internal computations.
Do not mutate the vectors or their polynomials, or expose them through a
public return value. Use `__system_vectors` when independent working copies
are needed.
"""
function __action_read_system(system::AbstractActionSystem)
    data = getfield(system, :_data)
    return data.equations, data.inequations
end

Base.isequal(a::AbstractActionSystem, b::AbstractActionSystem) = isequal(__action_system_key(a), __action_system_key(b))
Base.:(==)(a::AbstractActionSystem, b::AbstractActionSystem) = isequal(a, b)
Base.hash(system::AbstractActionSystem, h::UInt) = hash(__action_system_key(system), h)

function __action_branch_vectors(input::AbstractActionSystem, branches::AbstractVector{<:AbstractActionSystem})
    P = elem_type(input.ring)
    __require_parent(input.ring, branches)
    return Tuple{Vector{P},Vector{P}}[system_to_vectors(branch) for branch in branches]
end

@doc raw"""
Convert the equations and inequations in one AbstractActionSystem to elements of a common
finite polynomial ring. Include only the occurring jet variables, in decreasing ranking
order; keep coefficients in the same ring. Return a new, uncertified AlgebraicSystem.
"""
function __conversion_algebraic_system(system::AbstractActionSystem)
    stored = getfield(system, :_data)
    E, I = stored.equations, stored.inequations
    conversion = Oscar.__algebraic_conversion_data_from_polys(system.ring, [E; I])
    P = elem_type(conversion.mpr)
    return AlgebraicSystem(Val(:owned), conversion.mpr, P[conversion(p) for p in E], P[conversion(q) for q in I])
end

@doc raw"""
    system_to_vectors(system::AbstractPolynomialSystem)
    system_to_vectors(systems::AbstractVector{<:AbstractPolynomialSystem})

Convert a system into a pair of vectors, representing the equations and the inequations
of the system.

Return independent copies, so that modifying the vectors does not change the system.
The second method applies the first to each system in a vector.
"""
function system_to_vectors(system::AbstractPolynomialSystem)
    return system.equations, system.inequations
end

function system_to_vectors(systems::AbstractVector{<:AbstractPolynomialSystem})
    return [system_to_vectors(system) for system in systems]
end

__system_vectors(system::AbstractPolynomialSystem) = system_to_vectors(system)
