abstract type AbstractPolynomialSystem end

"""
Store the shared equality/hash key, separately from working data and certificates.
Key polynomials may be shared with the system's owned data; never mutate either
through internal references. Public polynomial properties return working copies.
"""
struct __PolynomialSystemKey{R,P}
    ring::R
    equations::Tuple{Vararg{P}}
    inequations::Tuple{Vararg{P}}
end

function Base.isequal(a::__PolynomialSystemKey, b::__PolynomialSystemKey)
    a.ring === b.ring || return false
    for (left, right) in ((a.equations, b.equations), (a.inequations, b.inequations))
        length(left) == length(right) || return false
        all(p -> any(q -> isequal(p, q), right), left) || return false
    end
    return true
end

Base.:(==)(a::__PolynomialSystemKey, b::__PolynomialSystemKey) = isequal(a, b)

"""
Compute a hash independent of the order within equations and inequations.
"""
function Base.hash(key::__PolynomialSystemKey, h::UInt)
    equations_hash = foldl(xor, (hash(p) for p in key.equations); init=zero(UInt))
    inequations_hash = foldl(xor, (hash(p) for p in key.inequations); init=zero(UInt))
    return hash((objectid(key.ring), length(key.equations), equations_hash,
                 length(key.inequations), inequations_hash), h)
end

"""
Build an order-independent key. By default, detach its polynomials from the input.
Set `copy_polynomials=false` only for already owned system data that will remain unchanged;
`normalize` must also leave its argument unchanged.
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

__system_vectors(system::AbstractPolynomialSystem) = system_to_vectors(system)

###############################################################################
#####  Algebraic systems
###############################################################################

@doc raw"""
    AlgebraicSystem(R, equations=[], inequations=[])
    AlgebraicSystem(equations, inequations=[])
    AlgebraicSystem(system::AlgebraicSystem)

Represent an algebraic system

`equations = 0`\
`inequations ≠ 0`

in the polynomial ring `R`. Pass `R` explicitly if both lists are empty.

Every new system, including a copy, starts with `quasi_simple === nothing` and `simple === nothing`.
Cache the two properties separately through `is_quasi_simple(system)` and `is_simple(system)`.\
A simple system is also quasi-simple; a failed discriminant check does not invalidate quasi-simplicity.

The constructor checks the common parent and owns independent copies of the polynomials.
The properties `equations` and `inequations` return independent working copies; changing those
copies does not change the system or invalidate its certificate.
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

    julia> system = AlgebraicSystem([x^2-y]);
    julia> is_quasi_simple(system), is_simple(system)
    (true, false)
    julia> system.quasi_simple, system.simple
    (true, false)
    ```
"""
struct AlgebraicSystem{R, P<:MPolyRingElem} <: AbstractPolynomialSystem
    ring::R
    equations::Vector{P}
    inequations::Vector{P}
    quasi_simple::Base.RefValue{Union{Nothing,Bool}}
    simple::Base.RefValue{Union{Nothing,Bool}}
    _key::__PolynomialSystemKey{R,P}
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
    Internal entry for the output of __algebraic_cleanup and __algebraic_normalized.
    Those lists already have normalized coefficients and no duplicates.
    Copy the input once, then share the owned polynomials with the key without normalizing again.
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
    Copy the normalized key snapshot instead of normalizing the polynomials again.
    Keep both the working data and the key polynomials independent of the source.
    Preserve sharing within the copy, even when polynomial deepcopy methods do not
    record their result in Julia's deepcopy dictionary.
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

AlgebraicSystem(R::Ring, E::AbstractVector=elem_type(R)[]) = AlgebraicSystem(R, E, elem_type(R)[])

function AlgebraicSystem(E::AbstractVector, I::AbstractVector=eltype(E)[])
    isempty(E) && isempty(I) &&
        throw(ArgumentError("Pass the polynomial ring explicitly for an empty system"))
    p = isempty(E) ? first(I) : first(E)
    return AlgebraicSystem(parent(p), E, I)
end

""" Default deepcopy would preserve the certificate; every new system starts uncertified. """
Base.deepcopy_internal(system::AlgebraicSystem, ::IdDict) = AlgebraicSystem(system)

"""
Return independent copies of `equations` and `inequations`, and return the
values stored in the `quasi_simple` and `simple` references
"""
function Base.getproperty(system::AlgebraicSystem, name::Symbol)
    name === :equations && return deepcopy(getfield(system, :equations))
    name === :inequations && return deepcopy(getfield(system, :inequations))
    name in (:quasi_simple, :simple) && return getfield(system, name)[]
    return getfield(system, name)
end

Base.propertynames(::AlgebraicSystem, private::Bool=false) = private ?
    (:ring, :equations, :inequations, :quasi_simple, :simple, :_key) :
    (:ring, :equations, :inequations, :quasi_simple, :simple)

Base.isequal(a::AlgebraicSystem, b::AlgebraicSystem) = isequal(__algebraic_key(a), __algebraic_key(b))

Base.:(==)(a::AlgebraicSystem, b::AlgebraicSystem) = isequal(a, b)

"""
To use AlgebraicSystem with Set e Dict
"""
Base.hash(system::AlgebraicSystem, h::UInt) = hash(__algebraic_key(system), h)

###############################################################################
#####  Action Systems
###############################################################################

abstract type AbstractActionSystem <: AbstractPolynomialSystem end

"""
Share owned polynomial data and certificates between the two action-system types.
The key uses the same __PolynomialSystemKey representation as AlgebraicSystem.
Only certification may change in place.
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
    # Exact polynomial identity is independent of display settings and ranking.
    key = __polynomial_system_key(R, equations, inequations; copy_polynomials=false)
    return __ActionSystemData(R, equations, inequations, Dict{Symbol,Bool}(), key)
end

struct DifferenceSystem{R,P} <: AbstractActionSystem
    _data::__ActionSystemData{R,P}
    function DifferenceSystem(
        R::DifferencePolyRing,
        E::AbstractVector,
        I::AbstractVector
    )
        data = __action_system_data(R, E, I)
        return new{typeof(R),elem_type(R)}(data)
    end
end

@doc raw"""
    DifferentialSystem(R, equations=[], inequations=[])
    DifferentialSystem(equations, inequations=[])
    DifferentialSystem(system::DifferentialSystem)

An immutable snapshot of a differential system. All polynomials must have the
same parent; give the ring explicitly for a completely empty system.
Every constructor and copy starts with `quasi_simple === nothing`.
The function `is_quasi_simple` returns and caches a Boolean for the full differential
definition.
New systems and copies always have `quasi_simple === nothing` and `simple === nothing`.

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 1);
    julia> system = DifferentialSystem([u^2]);
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
struct DifferentialSystem{R,P} <: AbstractActionSystem
    _data::__ActionSystemData{R,P}
    function DifferentialSystem(R::DifferentialPolyRing, E::AbstractVector, I::AbstractVector)
        data = __action_system_data(R, E, I)
        return new{typeof(R),elem_type(R)}(data)
    end
end

DifferenceSystem(R::DifferencePolyRing, E::AbstractVector=elem_type(R)[]) =
    DifferenceSystem(R, E, elem_type(R)[])
DifferentialSystem(R::DifferentialPolyRing, E::AbstractVector=elem_type(R)[]) =
    DifferentialSystem(R, E, elem_type(R)[])

function DifferenceSystem(E::AbstractVector, I::AbstractVector=eltype(E)[])
    R = __require_same_parent(E, I)
    R isa DifferencePolyRing || throw(ArgumentError("Supply a difference ring explicitly for an empty system"))
    return DifferenceSystem(R, E, I)
end

function DifferentialSystem(E::AbstractVector, I::AbstractVector=eltype(E)[])
    R = __require_same_parent(E, I)
    R isa DifferentialPolyRing || throw(ArgumentError("Supply a differential ring explicitly for an empty system"))
    return DifferentialSystem(R, E, I)
end

function DifferenceSystem(system::DifferenceSystem)
    data = getfield(system, :_data)
    return DifferenceSystem(data.ring, data.equations, data.inequations)
end

function DifferentialSystem(system::DifferentialSystem)
    data = getfield(system, :_data)
    return DifferentialSystem(data.ring, data.equations, data.inequations)
end

Base.copy(system::DifferenceSystem) = DifferenceSystem(system)
Base.copy(system::DifferentialSystem) = DifferentialSystem(system)
Base.deepcopy_internal(system::AbstractActionSystem, ::IdDict) = copy(system)

function Base.getproperty(system::AbstractActionSystem, name::Symbol)
    data = getfield(system, :_data)
    name === :ring && return data.ring
    name === :equations && return deepcopy(data.equations)
    name === :inequations && return deepcopy(data.inequations)
    name in __system_certificate_names(system) && return get(data.certificates, name, nothing)
    return getfield(system, name)
end

__system_certificate_names(::DifferentialSystem) = (:quasi_simple, :simple)
Base.propertynames(system::AbstractActionSystem, private::Bool=false) =
    (:ring, :equations, :inequations, __system_certificate_names(system)..., (private ? (:_data,) : ())...)

__action_system_key(system::AbstractActionSystem) = getfield(system, :_data).key

"""
Borrow the owned vectors for read-only internal computations. Do not mutate the
vectors or their polynomials, or expose them through a public return value.
Use `__system_vectors` when independent working copies are needed.
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
Convert the equations and inequations to a common finite polynomial ring.
Include only the occurring jet variables, in decreasing ranking order; keep coefficients in the same ring.
Return a new, uncertified `AlgebraicSystem`, including for empty systems.
"""
function __conversion_algebraic_system(system::AbstractActionSystem)
    stored = getfield(system, :_data)
    E, I = stored.equations, stored.inequations
    conversion = Oscar.__algebraic_conversion_data_from_polys(system.ring, [E; I])
    P = elem_type(conversion.mpr)
    return AlgebraicSystem(conversion.mpr, P[conversion(p) for p in E], P[conversion(q) for q in I])
end

@doc raw"""
    system_to_vectors(system::Union{AbstractActionSystem,AlgebraicSystem})
    system_to_vectors(systems::AbstractVector{<:Union{AbstractActionSystem,AlgebraicSystem}})

Convert a system (`AlgebraicSystem`, `DifferentialSystem`, or `DifferenceSystem`) into a pair
of vectors, representing in order the equations and the inequations of the system.

Return independent polynomial copies, so changing the vectors does not change the systems.
The second method applies the same conversion to each system in a vector.
"""
function system_to_vectors(system::Union{AbstractActionSystem,AlgebraicSystem})
    return system.equations, system.inequations
end

function system_to_vectors(systems::AbstractVector{<:Union{AbstractActionSystem,AlgebraicSystem}})
    return [system_to_vectors(system) for system in systems]
end