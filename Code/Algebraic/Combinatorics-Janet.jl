# Keep algebraic leader conventions internal; use OSCAR leader/initial for action polynomials.

function __algebraic_leader(p::MPolyRingElem{T}) where T
    is_constant(p) && return parent(p)(1)
    return maximum(vars(p))
end

function __algebraic_initial(p::MPolyRingElem{T}) where T
    is_constant(p) && error("The initial of a constant polynomial is not defined")
    ld = __algebraic_leader(p)
    return leading_coefficient(p, var_index(ld))
end

###############################################################################
#####  Multiplicative variables
###############################################################################

@doc raw"""
    multiplicative_variables(
        M::AbstractVector{<:AbstractVector{<:Integer}}
    )
    multiplicative_variables(
        M::AbstractVector{<:MPolyRingElem{T}}
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.\
For a vector `M` = [m\_1, ..., m\_n] of monomials, compute the multiplicative variables of each m\_i in `M`.

If `M` is given by exponent vectors (identify m = x\_1^(a\_1)\*...\*x\_l^(a\_l) with [a\_1, ..., a\_l]),
return a `Vector{BitVector}` `MV`: `MV`[i][j]=1 if x\_j is multiplicative for m\_i.
If the vectors in `M` do not have the same length, the shorter ones are padded with zeros to reach the maximum length.\
If `M` is given by monomials, return the corresponding variables themselves.

    # Examples

    ```jldoctest
    julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]];
    julia> multiplicative_variables(M)
    4-element Vector{BitVector}:
     [1, 1, 1, 1]
     [0, 0, 1, 1]
     [0, 1, 1, 1]
     [0, 0, 0, 1]
    julia> M1 = [[4], [0, 1, 2], [0, 2, 0, 4], [0, 1, 1, 2]];
    julia> multiplicative_variables(M) == multiplicative_variables(M1)
    true

    julia> R, (x1, x2, x3, x4) = polynomial_ring(QQ, [:x1, :x2, :x3, :x4]);
    julia> M = [x1^4, x2*x3^2, x2^2*x4^4, x2*x3*x4^2];
    julia> multiplicative_variables(M)
    4-element Vector{Vector{QQMPolyRingElem}}:
     [x1, x2, x3, x4]
     [x3, x4]
     [x2, x3, x4]
     [x4]
    ```
"""
function multiplicative_variables(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    return __multiplicative_variable_masks(M)
end

function multiplicative_variables(
    M::AbstractVector{<:MPolyRingElem{T}}
) where T
    R = __require_same_parent(M)
    isempty(M) && return Vector{Vector{eltype(M)}}()
    if any(p -> length(p) != 1, M)
        error("The elements of the input vector must be monomials")
    end
    exps = [exponent_vector(p, 1) for p in M]
    MV = __multiplicative_variable_masks_unchecked(exps)
    R_gens = gens(R)
    W = [R_gens[m] for m in MV]
return W
end

function __multiplicative_variable_masks(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    return __multiplicative_variable_masks_unchecked(__padded_integer_vectors(M))
end

# Read nonnegative exponent vectors of equal length without copying or validating them.
function __multiplicative_variable_masks_unchecked(M)
    isempty(M) && return BitVector[]
    n = length(M)
    ind = length(first(M))
    # Lexicographic ordering
    perm = sortperm(M)
    sorted_M = M[perm]
    # Initialization
    sorted_MV = [falses(ind) for _ in 1:n]
    # At step h, these delimit blocks having the same prefix of length h - 1
    bounds = Int[1, n + 1]
    sizehint!(bounds, n + 1)
    next_bounds = Int[]
    sizehint!(next_bounds, n + 1)
    for h in 1:ind
        empty!(next_bounds)
        @inbounds for b in 1:(length(bounds) - 1)
            lower = bounds[b]
            higher = bounds[b + 1] - 1
            h < ind && push!(next_bounds, lower)
            # Inside a block, the h-th exponents are sorted, thus the last run corresponds to the maximum
            first_max = lower
            for x in (lower + 1):higher
                if sorted_M[x][h] != sorted_M[x - 1][h]
                    first_max = x
                    h < ind && push!(next_bounds, x)
                end
            end
            # Exactly these monomials attain the maximum h-th exponent in the current block
            for x in first_max:higher
                sorted_MV[x][h] = true
            end
        end
        if h < ind
            push!(next_bounds, n + 1)
            bounds, next_bounds = next_bounds, bounds
        end
    end
    # Restore the original order
    MV = Vector{BitVector}(undef, n)
    @inbounds for new_idx in eachindex(perm)
        MV[perm[new_idx]] = sorted_MV[new_idx]
    end
    return MV
end

function __padded_integer_vectors(
    M::AbstractVector{<:AbstractVector{<:Integer}},
    n::Integer=isempty(M) ? 0 : maximum(length, M)
)
    V = [collect(Int, __integer_check(m)) for m in M]
    for v in V
        l = length(v)
        if l < n
            resize!(v,n)
            @inbounds for i in (l+1):n
                v[i] = 0
            end
        end
    end
    return V
end

###############################################################################
#####  Janet cones
###############################################################################

@doc raw"""
    cones_check(
        w::AbstractVector{<:Integer},
        M::AbstractVector{<:AbstractVector{<:Integer}}
    )
    cones_check(
        w::AbstractVector{<:Integer},
        M::AbstractVector{<:AbstractVector{<:Integer}},
        MV::AbstractVector{<:BitVector}
    )
    cones_check(
        w::MPolyRingElem{T},
        M::AbstractVector{<:MPolyRingElem{T}}
    ) where T
    cones_check(
        w::MPolyRingElem{T},
        M::AbstractVector{<:MPolyRingElem{T}},
        MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}}
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.\

Given a monomial `w` and a vector `M` of monomials (possibly represented by the corresponding exponent vectors)
return `true` if `w` is in one of the cones with vertex in one element of `M`,
spreading in the directions of the corresponding multiplicative variables, and `false` otherwise.

If `MV` is supplied, it is used as the vector of multiplicative variables; otherwise it is computed from `M`.

    # Examples

    ```jldoctest
    julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]];
    julia> w = [1, 1, 2, 0];
    julia> cones_check(w, M)
    false

    julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]];
    julia> w = [0, 1, 2, 5];
    julia> cones_check(w, M)
    true

    julia> R, (x1, x2, x3, x4) = polynomial_ring(QQ, [:x1, :x2, :x3, :x4]);
    julia> M = [x1^4, x2*x3^2, x2^2*x4^4, x2*x3*x4^2];
    julia> w = x1*x2*x3^2;
    julia> cones_check(w, M)
    false

    julia> R, (x1, x2, x3, x4) = polynomial_ring(QQ, [:x1, :x2, :x3, :x4]);
    julia> M = [x1^4, x2*x3^2, x2^2*x4^4, x2*x3*x4^2];
    julia> w = x2*x3^2*x4^5;
    julia> cones_check(w, M)
    true
    ```
"""
function cones_check(
    w::AbstractVector{<:Integer},
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    w = __integer_check(w)
    all(m -> length(m) == length(w), M) ||
        error("All monomial exponent vectors must have the same length as w")
    M = __padded_integer_vectors(M, length(w))
    MV = __multiplicative_variable_masks_unchecked(M)
    return __cones_check_unchecked(w, M, MV)
end

function cones_check(
    w::AbstractVector{<:Integer},
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:BitVector}
)
    __check_janet_data(w, M, MV)
    return __cones_check_unchecked(w, M, MV)
end

@inline function __cones_check_unchecked(w, M, MV)
    for (m, l) in zip(M, MV)
        # Detect the positions in which there is a shift (positive value) and
        # whether there is a position in which a shift is not possible
        ok = true
        @inbounds for i in eachindex(w)
            c = w[i] - m[i]
            if c < 0 || (c > 0 && !l[i])
                ok = false
                break
            end
        end
        # If w is not an admissible shift of m, it may be an admissible shift of another monomial
        ok ? (return true) : continue
    end
    return false
end

function cones_check(
    w::MPolyRingElem{T},
    M::AbstractVector{<:MPolyRingElem{T}}
) where T
    __require_same_parent(w, M)
    is_term(w) && all(is_term, M) ||
        throw(ArgumentError("The inputs must be nonzero monomials"))
    w = exponent_vector(w, 1)
    M = [exponent_vector(m, 1) for m in M]
    MV = __multiplicative_variable_masks_unchecked(M)
    return __cones_check_unchecked(w, M, MV)
end

function cones_check(
    w::MPolyRingElem{T},
    M::AbstractVector{<:MPolyRingElem{T}},
    MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}}
) where T
    R = __require_same_parent(w, M, MV)
    is_term(w) && all(is_term, M) ||
        throw(ArgumentError("The inputs must be nonzero monomials"))
    length(M) == length(MV) ||
        error("The monomial and multiplicative-variable lists must have the same length")
    w = exponent_vector(w, 1)
    M = [exponent_vector(m, 1) for m in M]
    MV_conv = __variable_masks_from_vars(R, MV)
    return __cones_check_unchecked(w, M, MV_conv)
end

function __check_janet_data(
    w::AbstractVector{<:Integer},
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:BitVector}
)
    w = __integer_check(w)
    foreach(__integer_check, M)
    if length(M) != length(MV)
        error("The monomial and multiplicative-variable lists must have the same length")
    end
    n = length(w)
    if !all(m -> length(m) == n, M)
        error("All monomial exponent vectors must have the same length as $w")
    end
    if !all(l -> length(l) == n, MV)
        error("All multiplicative-variable vectors must have the same length as $w")
    end
    return nothing
end

###############################################################################
#####  Janet completeness
###############################################################################

@doc raw"""
    is_janet_complete(
        M::AbstractVector{<:AbstractVector{<:Integer}},
        MV::AbstractVector{<:BitVector};
        info::Bool=false,
        missing::Bool=false
    )
    is_janet_complete(
        M::AbstractVector{<:AbstractVector{<:Integer}};
        info::Bool=false,
        missing::Bool=false
    )
    is_janet_complete(
        M::AbstractVector{<:MPolyRingElem{T}},
        MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}};
        info::Bool=false,
        missing::Bool=false
    ) where T
    is_janet_complete(
        M::AbstractVector{<:MPolyRingElem{T}};
        info::Bool=false,
        missing::Bool=false
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.\
For a vector `M` = [m\_1, ...m\_n] of monomials (possibly represented by the corresponding exponent vectors),
return `true` if `M` is Janet-complete, and `false` otherwise.

If `MV` is supplied, it is used as the vector of multiplicative variables; otherwise it is computed from `M`.\
If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.\
If the optional argument `missing` is set to `true`, the first missing element encountered is also returned.

    # Examples

    ```jldoctest
    julia> M = [[0, 0], [1, 0], [0, 2], [0, 1]];
    julia> is_janet_complete(M)
    true

    julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]];
    julia> MV = multiplicative_variables(M)
    4-element Vector{BitVector}:
     [1, 1, 1, 1]
     [0, 0, 1, 1]
     [0, 1, 1, 1]
     [0, 0, 0, 1]
    julia> is_janet_complete(M, MV)
    false
    julia> is_janet_complete(M, MV; info=true)
    [1, 1, 2, 0] should be added
    false
    julia> is_janet_complete(M, MV; missing=true)
    (false, [1, 1, 2, 0])

    julia> R, (x1, x2, x3, x4) = polynomial_ring(QQ, [:x1, :x2, :x3, :x4]);
    julia> M = [x1^4, x2*x3^2, x2^2*x4^4, x2*x3*x4^2];
    julia> is_janet_complete(M; missing=true)
    (false, x1*x2*x3^2)

    julia> R, (x1, x2) = polynomial_ring(QQ, [:x1, :x2]);
    julia> M = [R(1), x1, x2^2];
    julia> is_janet_complete(M; missing=true)
    (false, x2)
    julia> push!(M, x2);
    julia> is_janet_complete(M)
    true
    ```
"""
function is_janet_complete(
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::Union{Nothing,AbstractVector{<:BitVector}}=nothing;
    info::Bool=false,
    missing::Bool=false
)
    missing_monomial = (MV === nothing) ? __first_janet_missing(M) : __first_janet_missing(M, MV)
    missing_monomial === nothing && return true
    (info || missing) || return false
    info && println("$missing_monomial should be added")
    missing && return false, missing_monomial
end

# Callers supply nonnegative exponent vectors and matching masks of the ring's width.
function __is_janet_complete_poly(
    R::MPolyRing,
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:BitVector};
    info::Bool=false,
    missing::Bool=false
)
    exponents = __first_janet_missing_unchecked(M, MV)
    exponents === nothing && return true
    (info || missing) || return false
    p = R([one(coefficient_ring(R))], [exponents])
    info && println("$p should be added")
    missing && return false, p
end

function is_janet_complete(
    M::AbstractVector{<:MPolyRingElem{T}},
    MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}};
    info::Bool=false,
    missing::Bool=false
) where T
    R = __require_same_parent(M, MV)
    length(M) == length(MV) ||
        throw(ArgumentError("The monomial and multiplicative-variable lists must have the same length"))
    isempty(M) && return true
    all(is_term, M) ||
        throw(ArgumentError("The inputs must be nonzero monomials"))
    exps = [exponent_vector(m, 1) for m in M]
    masks = __variable_masks_from_vars(R, MV)
    return __is_janet_complete_poly(
        R, exps, masks;
        info=info, missing=missing
    )
end

function is_janet_complete(
    M::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false,
    missing::Bool=false
) where T
    R = __require_same_parent(M)
    isempty(M) && return true
    all(is_term, M) ||
        throw(ArgumentError("The inputs must be nonzero monomials"))
    exps = [exponent_vector(m, 1) for m in M]
    MV = __multiplicative_variable_masks_unchecked(exps)
    return __is_janet_complete_poly(
        R, exps, MV;
        info=info, missing=missing
    )
end

###############################################################################
#####  Janet completion
###############################################################################

@doc raw"""
    janet_completion(
        M::AbstractVector{<:AbstractVector{<:Integer}}
    )
    janet_completion(
        M::AbstractVector{<:MPolyRingElem{T}}
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.
For a vector `M` = [m\_1, ...m\_n] of monomials ((possibly represented by the corresponding exponent vectors),
compute the Janet-completion of `M`.

The output uses the same representation as the input.

    # Examples

    ```jldoctest
    julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]];
    julia> janet_completion(M)
    29-element Vector{Vector{Int64}}:
     [4, 0, 0, 0]
     [0, 1, 2, 0]
     [0, 2, 0, 4]
     [0, 1, 1, 2]
     [1, 1, 2, 0]
     [0, 2, 2, 0]
     [1, 2, 0, 4]
     [0, 2, 1, 4]
     [1, 1, 1, 2]
     [0, 2, 1, 2]
     ⋮
     [2, 2, 1, 4]
     [3, 1, 1, 2]
     [2, 2, 1, 2]
     [1, 2, 1, 3]
     [3, 2, 2, 0]
     [3, 2, 1, 4]
     [3, 2, 1, 2]
     [2, 2, 1, 3]
     [3, 2, 1, 3]

    julia> M = [[0, 0], [1, 0], [0, 2]];
    julia> janet_completion(M)
    4-element Vector{Vector{Int64}}:
     [0, 0]
     [1, 0]
     [0, 2]
     [0, 1]

    julia> R, (x1, x2, x3, x4) = polynomial_ring(QQ, [:x1, :x2, :x3, :x4]);
    julia> M = [x1^4, x2*x3^2, x2^2*x4^4, x2*x3*x4^2];
    julia> janet_completion(M)
    29-element Vector{QQMPolyRingElem}:
     x1^4
     x2*x3^2
     x2^2*x4^4
     x2*x3*x4^2
     x1*x2*x3^2
     x2^2*x3^2
     x1*x2^2*x4^4
     x2^2*x3*x4^4
     x1*x2*x3*x4^2
     x2^2*x3*x4^2
     ⋮
     x1^2*x2^2*x3*x4^4
     x1^3*x2*x3*x4^2
     x1^2*x2^2*x3*x4^2
     x1*x2^2*x3*x4^3
     x1^3*x2^2*x3^2
     x1^3*x2^2*x3*x4^4
     x1^3*x2^2*x3*x4^2
     x1^2*x2^2*x3*x4^3
     x1^3*x2^2*x3*x4^3

    julia> R, (x1, x2) = polynomial_ring(QQ, [:x1, :x2]);
    julia> M = [R(1), x1, x2^2];
    julia> janet_completion(M)
    4-element Vector{QQMPolyRingElem}:
     1
     x1
     x2^2
     x2
    ```
"""
function janet_completion(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    J = __padded_integer_vectors(M)
    isempty(J) && return J
    MV = BitVector[]
    sizehint!(MV, length(J))
    groups = [Dict{Tuple,Tuple{Int,Vector{Int}}}() for _ in eachindex(first(J))]
    @inbounds for i in eachindex(J)
        __janet_insert_mask!(MV, groups, J[i], i)
    end
    while true
        missing_monomial = __first_janet_missing_unchecked(J, MV)
        missing_monomial === nothing && break
        # A missing prolongation cannot already be a vertex of its own cone.
        push!(J, missing_monomial)
        __janet_insert_mask!(MV, groups, missing_monomial, length(J))
    end
    return J
end

# Only the groups sharing a prefix with the new vertex can change their maxima.
# Store only the members attaining each current maximum.
function __janet_insert_mask!(MV, groups, m, index)
    mask = trues(length(m))
    push!(MV, mask)
    @inbounds for j in eachindex(m)
        prefix = Tuple(@view m[1:j-1])
        group = get(groups[j], prefix, nothing)
        if group === nothing
            groups[j][prefix] = (m[j], [index])
            continue
        end
        maximum, members = group
        if m[j] > maximum
            for i in members
                MV[i][j] = false
            end
            empty!(members)
            push!(members, index)
            groups[j][prefix] = (m[j], members)
        elseif m[j] == maximum
            push!(members, index)
        else
            mask[j] = false
        end
    end
    return nothing
end

function janet_completion(
    M::AbstractVector{<:MPolyRingElem{T}}
) where T
    R = __require_same_parent(M)
    isempty(M) && return collect(M)
    all(is_term, M) || throw(ArgumentError("The inputs must be nonzero monomials"))
    M = [exponent_vector(m, 1) for m in M]
    r = janet_completion(M)
    unit = one(coefficient_ring(R))
    return [R([unit], [c]) for c in r]
end

###############################################################################
#####  Internal helpers
###############################################################################

function __integer_check(V::AbstractVector{<:Integer})
    any(x -> x < 0, V) && throw(ArgumentError("Monomial exponents must be non-negative"))
    return V
end

"""
Return the first exponent vector that is missing from the Janet cone decomposition of `M`,
or `nothing` if `M` is Janet complete.

The first method uses the already computed multiplicative-variable masks `MV`;
the second method computes them from `M`.
"""
function __first_janet_missing(
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:BitVector}
)
    if length(M) != length(MV)
        error("The monomial and multiplicative-variable lists must have the same length")
    end
    isempty(M) && return nothing
    width = max(maximum(length, M), maximum(length, MV))
    M = __padded_integer_vectors(M, width)
    MV = __pad_variable_masks(MV, width)
    return __first_janet_missing_unchecked(M, MV)
end

function __first_janet_missing_unchecked(M, MV)
    w = nothing
    @inbounds for k in eachindex(M)
        m = M[k]
        l = MV[k]
        all(l) && continue
        if w === nothing
            w = copy(m)
        else
            copyto!(w, m)
        end
        for n in eachindex(l)
            l[n] && continue
            w[n] += 1
            !__cones_check_unchecked(w, M, MV) && return w
            w[n] = m[n]
        end
    end
    return nothing
end

function __first_janet_missing(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    M = __padded_integer_vectors(M)
    MV = __multiplicative_variable_masks_unchecked(M)
    return __first_janet_missing_unchecked(M, MV)
end

# Return read-only masks, sharing those that need no padding.
function __pad_variable_masks(
    L::AbstractVector{<:BitVector},
    n::Integer,
)
    all(l -> length(l) >= n, L) && return L
    out = Vector{BitVector}(undef, length(L))
    @inbounds for i in eachindex(L)
        l = L[i]
        if length(l) >= n
            out[i] = l
        else
            b = falses(n)
            copyto!(b, 1, l, 1, length(l))
            out[i] = b
        end
    end
    return out
end

# Callers validate the common parent before converting the variables.
function __variable_masks_from_vars(
    R,
    MV::AbstractVector{<:AbstractVector}
)
    n = nvars(R)
    MV_conv = Vector{BitVector}(undef, length(MV))
    idx_cache = Dict{elem_type(R),Int}()
    sizehint!(idx_cache, min(n, sum(length, MV; init=0)))
    for (i, vars) in enumerate(MV)
        r = falses(n)
        for v in vars
            idx = get!(idx_cache, v) do
                var_index(v)
            end
            r[idx] = true
        end
        MV_conv[i] = r
    end
    return MV_conv
end
