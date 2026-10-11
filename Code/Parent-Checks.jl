"""
Return the common parent of the supplied polynomials/polynomial systems.
Inspect vectors and tuples recursively, including nested ones, and compare parents by ===;
throw an `ArgumentError` if they differ.
Return `nothing` with no arguments or only empty, possibly nested, vectors or tuples.
"""
function __require_same_parent(polynomials...)
    R = nothing
    for p in polynomials
        R = __common_polynomial_parent(R, p)
    end
    return R
end

"""
Check that every supplied polynomial or polynomial system has parent `R`.
Inspect vectors and tuples recursively, and throw an `ArgumentError` if
a parent is not `=== R`.
Return `nothing` on success, including when no arguments or only empty collections are supplied.
Supply an actual parent as `R`, not `nothing`.
Use `__require_same_parent` when the common parent must be inferred from the arguments.
"""
function __require_parent(R, polynomials...)
    for p in polynomials
        __common_polynomial_parent(R, p)
    end
    return nothing
end

"""
Return the parent of `p` requiring identity with `R` unless `R === nothing`.
For vectors and tuples, inspect the elements recursively and
carry the parent found so far through the collection.
Return `R` unchanged for an empty collection and throw an `ArgumentError` on a parent mismatch.
"""
function __common_polynomial_parent(R, p)
    S = parent(p)
    R === nothing || R === S ||
        throw(ArgumentError("All polynomials must have the same parent"))
    return S
end

function __common_polynomial_parent(R, polynomials::Union{AbstractVector,Tuple})
    for p in polynomials
        R = __common_polynomial_parent(R, p)
    end
    return R
end

function __common_polynomial_parent(R, system::AbstractPolynomialSystem)
    S = system.ring
    R === nothing || R === S || throw(ArgumentError("All polynomials must have the same parent"))
    return S
end