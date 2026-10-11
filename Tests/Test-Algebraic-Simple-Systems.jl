using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "A shared element type does not imply a shared parent" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached=false)
    Q, (u, v) = polynomial_ring(QQ, [:x, :y]; cached=false)
    @test typeof(x) === typeof(u) && parent(x) !== parent(u)
    @test solution_check(x, [x])
    @test !solution_check(x, [x], [x])
    @test solution_check(x, MPolyRingElem{QQFieldElem}[x])
    @test_throws ArgumentError solution_check(x, [u])
    @test_throws ArgumentError solution_check(x, [x], [v])
end

@testset "Discriminants and generic fallback" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    for p in [R(0), R(3), a*x+b]
        @test __algebraic_discriminant(p) == R(1)
    end
    for p in [a*x^2+b*x+1, x^3+a*x+b, (x-a)^2, a*(x^4+b*x+1)]
        idx = var_index(__algebraic_leader(p))
        d = degree(p, idx)
        expected = divexact((-1)^div(d*(d-1), 2) * resultant(p, derivative(p, idx), idx), __algebraic_initial(p))
        @test __algebraic_discriminant(p) == expected
    end
    K, t = rational_function_field(QQ, :t)
    A, (u, v) = polynomial_ring(K, [:u, :v])
    @test __algebraic_discriminant(u^2+t*u+v) == t^2-4*v
end

@testset "Known factors preserve discriminants, collisions and coefficient factors" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, a, b) = polynomial_ring(field, [:x, :a, :b])
        scalars = field === K ? (K(1), K(2), t+1) : (field(1), field(2), field(-3))
        for factors in ([x-a, x-b], [2*x-a, 3*x-b], [x^2-a, x-b], [x-a, x-a]), c in scalars
            p = c*prod(factors)
            snapshot = deepcopy((p, factors))
            @test __algebraic_discriminant(p, 1, factors) == __algebraic_discriminant(p, 1)
            @test (p, factors) == snapshot
        end
        # Both linear factors have discriminant one, but their roots meet at a=b.
        @test __algebraic_discriminant((x-a)*(x-b), 1, [x-a, x-b]) == (a-b)^2
        @test __algebraic_discriminant((2*x-a)*(3*x-b), 1, [2*x-a, 3*x-b]) == (3*a-2*b)^2
        # A list without multiplicities must not certify a repeated-root polynomial.
        @test iszero(__algebraic_discriminant((x-a)^2*(x-b), 1, [x-a, x-b]))
        # Factors independent of the chosen leader use the direct method.
        @test __algebraic_discriminant(a*(x^2-b), 1, [a, x^2-b]) == 4*a^2*b
    end
end

@testset "Leader discriminants over different coefficient fields" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, y, z) = polynomial_ring(field, [:x, :y, :z])
        @test __algebraic_discriminant(y*x^2+z*x+1) == z^2-4*y
        @test iszero(__algebraic_discriminant((x-y)^2))
    end
end

#using Random
# Deliberately use full ideal operations as independent mathematical oracles.
function reference_compatible(r, E, I)
    R = parent(r)
    J = ideal(R, [E; r])
    h = prod(I; init=one(R))
    return !is_one(saturation(J, ideal(R, [h])))
end

function reference_regular(p, E, I)
    R = parent(p)
    K = saturation(ideal(R, E), ideal(R, [prod(I; init=one(R))]))
    return saturation(K, ideal(R, [p])) == K
end

@testset "Exact decisions without unnecessary Groebner bases" begin
    for field in (QQ, GF(7))
        R, (x, y, z) = polynomial_ring(field, [:x, :y, :z])
        P = elem_type(R)
        equations = [P[], [R(0)], [R(1)], [x^4*y^2],
                     [x^2*(x-1), x*(x-1)^2], [x*y, x*y*(x+1)],
                     [x, y], [x*y-1, x], [x^2, x*y], [x^2-y, y^2-z]]
        guards = [P[], [R(0)], [R(2)], [x], [y], [x, y], [x+y+1]]
        for E in equations, I in guards, p in (R(0), R(1), x, y, x+y)
            @test __compatible_zero_algebraic_system(p, E, I) == reference_compatible(p, E, I)
            @test __algebraic_vanishes_on_system(p, E, I) == !reference_compatible(R(0), E, [I; p])
        end
        # gcd(x,y)=1 does not imply ideal(x,y)=ideal(1).
        @test __algebraic_principal_generator([x,y]) === nothing
        @test __compatible_zero_algebraic_system(R(0), [x,y], P[])
        @test !__compatible_zero_algebraic_system(R(0), [x*y-1,x], P[])
    end
end

@testset "Existence checks project certified highest-leader constraints" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, y, a) = polynomial_ring(field, [:x, :y, :a])
        P = elem_type(R)
        cases = [
            ([x^2-y, y^2-a], [a]),
            ([a*x^2+y, y^2-a], [a]),
            ([y^2-a], [x^3+y, x+1, a]),
            ([a], [a*x]),
            ([a], [a*x+1]),
            ([a*x-1, a], P[]),
            ([x^2-a], [x]),
            ([x^2-a, x^2-a+1], P[]),
            ([y*a], [a*x+1, a]),
            ([y*a], [a*x, y]),
            ([a], [x^7-x]),
        ]
        for (E, I) in cases
            snapshot = deepcopy((E, I))
            projected_E, projected_I = __algebraic_compatibility_projection(E, I)
            expected = reference_compatible(zero(R), E, I)
            @test reference_compatible(zero(R), projected_E, projected_I) == expected
            @test __compatible_zero_algebraic_system(zero(R), E, I) == expected
            @test (E, I) == snapshot
        end
        # Eliminate equations only when every lower solution can be extended.
        @test __algebraic_compatibility_projection([x^2-y, y^2-a], [a]) == (P[], [a])
        @test __algebraic_compatibility_projection([a*x-1, a], P[]) == ([a*x-1, a], P[])
        # An equation and a guard at the same leader must still be checked together.
        @test __algebraic_compatibility_projection([x^2-a], [x]) == ([x^2-a], [x])
        # The guard a*x must not certify its own initial a.
        @test __algebraic_compatibility_projection([a], [a*x]) == ([a], [a*x])
        __with_algebraic_cache() do
            @test __compatible_zero_algebraic_system(zero(R), [x^2-y, y^2-a], [a])
            cache = task_local_storage()[:__algebraic_splitting]
            # Existence checks must not use ideal computations.
            @test all(key -> first(key) !== :ideal, keys(cache))
            E = [y*a]
            I = [x^3+y, y, a]
            _, projected_I = __algebraic_compatibility_projection(E, I)
            @test !__compatible_zero_algebraic_system(zero(R), E, I)
            @test x^3+y ∉ projected_I
            # Cache layout is optional; when present, it must reflect the projection.
            for key in keys(cache)
                first(key) === :compatible || continue
                @test x^3+y ∉ key[4]
            end
        end
        # Accept abstractly typed equation vectors as the public checks do.
        abstract_E = MPolyRingElem{elem_type(field)}[x^2-y, y^2-a]
        @test __compatible_zero_algebraic_system(zero(R), abstract_E, [a])
    end
end

@testset "Cache lifetime, parents, guards and polynomial ownership" begin
    R,(x,y) = polynomial_ring(QQ, [:x,:y];cached=false)
    S,(u,v) = polynomial_ring(QQ, [:x,:y];cached=false)
    cache_key = :__algebraic_splitting
    @test !haskey(task_local_storage(),cache_key)
    __with_algebraic_cache() do
        E = [x^2-y]
        first_ideal = __algebraic_equation_ideal(R,E)
        @test __algebraic_equation_ideal(R,E) === first_ideal
        @test __algebraic_equation_ideal(S,[u^2-v]) !== first_ideal
        E[1] = y
        @test __algebraic_equation_ideal(R,E) !== first_ideal
        @test gens(first_ideal) == [x^2-y]
        @test __compatible_zero_algebraic_system(R(0),[x*y],[x])
        @test !__compatible_zero_algebraic_system(R(0),[x*y],[x,y])
        factors = __algebraic_nonconstant_factors(x*(x+1))
        factors[1] = R(0)
        @test all(!iszero,__algebraic_nonconstant_factors(x*(x+1)))
        for i in 1:260
            __algebraic_equation_ideal(R,[x+i*y])
        end
        @test length(task_local_storage()[cache_key]) <= 256
        @test __algebraic_equation_ideal(R,[x^2-y]) == first_ideal
    end
    @test !haskey(task_local_storage(),cache_key)
    @test_throws ErrorException __with_algebraic_cache(() -> error("test cleanup"))
    @test !haskey(task_local_storage(),cache_key)
end

@testset "Cycles and exact quotient direction" begin
    R,(x,y) = polynomial_ring(QQ,[:x,:y])
    keys = [__algebraic_key(AlgebraicSystem([x-i*y])) for i in 1:4]
    edges = Dict{__PolynomialSystemKey,Vector{__PolynomialSystemKey}}()
    for (a,b) in ((x,x*y), (R(2),R(8)),(x,x+1),(R(0),x),(x,R(0)))
        d = __checked_divides(a,b)
        @test d.divides == first(Oscar.divides(b,a))
        @test d.divides ? a*d.quotient == b : d.quotient === nothing
    end
    @test __checked_divides(R(2),R(8)).quotient == 4
    @test __algebraic_discriminant(y*x^2+2*x) == 4
    for constructor in (difference_polynomial_ring,differential_polynomial_ring)
        D,t = constructor(QQ,:t,1)
        @test __checked_divides(t,t*(t+1)).quotient == t+1
        @test !__checked_divides(t,t+1).divides
    end
end

const snapshot_count = Ref(0)
mutable struct CollisionInspection
    value::Int
end

Base.isequal(a::CollisionInspection,b::CollisionInspection) = a.value == b.value
Base.hash(::CollisionInspection,h::UInt) = hash(:intentional_collision,h)
function Base.deepcopy_internal(p::CollisionInspection,::IdDict)
    snapshot_count[] += 1
    return CollisionInspection(p.value)
end

@testset "Cache inspections, snapshots, false and nothing" begin
    R,(x,y) = polynomial_ring(QQ,[:x,:y])
    __with_algebraic_cache() do
        E = [CollisionInspection(1)]
        I = [CollisionInspection(2)]
        @test __algebraic_cached(() -> false,:inspection,R,E,I) === false
        count = snapshot_count[]
        @test __algebraic_cached(() -> error("hit recomputed"),:inspection,R,[CollisionInspection(1)],[CollisionInspection(2)]) === false
        @test snapshot_count[] == count
        E[1].value = 3
        @test __algebraic_cached(() -> nothing,:inspection,R,E,I) === nothing
        count = snapshot_count[]
        @test __algebraic_cached(() -> error("nothing recomputed"),:inspection,R,E,I) === nothing
        @test snapshot_count[] == count
        @test __algebraic_cached(() -> error("snapshot changed"),:inspection,R,[CollisionInspection(1)],I) === false
        I[1].value = 4
        @test __algebraic_cached(() -> :new_guard,:inspection,R,E,I) === :new_guard
        @test __algebraic_cached(() -> :new_operation,:other,R,E,I) === :new_operation
        S,_ = polynomial_ring(QQ,[:x,:y];cached=false)
        @test __algebraic_cached(() -> :new_parent,:inspection,S,E,I) === :new_parent
        @test __algebraic_cached(() -> 1,:order,R,[x,y]) == 1
        @test __algebraic_cached(() -> 2,:order,R,[y,x]) == 2
        # Snapshot before running the callback, even if that callback changes input.
        E2 = [CollisionInspection(10)]
        @test __algebraic_cached(:mutating_callback,R,E2) do
            E2[1].value = 11
            :original
        end === :original
        @test __algebraic_cached(() -> error("late snapshot"),:mutating_callback,R,[CollisionInspection(10)]) === :original
        @test __algebraic_cached(() -> :changed,:mutating_callback,R,E2) === :changed
    end
end

@testset "Ideal and key snapshots survive polynomial mutation" begin
    K,t = rational_function_field(QQ,:t)
    for field in (QQ,GF(7),K)
        R,(x,y) = polynomial_ring(field,[:x,:y];cached=false)
        __with_algebraic_cache() do
            p=x^2-y; E=[p]; J=__algebraic_equation_ideal(R,E)
            zero!(p)
            @test gens(J)==[x^2-y]
            @test __algebraic_equation_ideal(R,[x^2-y]) === J
            @test __algebraic_equation_ideal(R,E) !== J
            # Stored ideal polynomials must also be independent of cache-key snapshots.
            zero!(first(gens(J)))
            @test __algebraic_equation_ideal(R,[x^2-y]) === J
            @test isempty(gens(__algebraic_equation_ideal(R,elem_type(R)[])))
        end
    end
end

# Use full ideal saturation as an independent check of the discriminant condition.
function reference_algebraic_simple(E, I)
    is_quasi_simple(E, I) || return false
    for p in Iterators.flatten((E, I))
        R = parent(p)
        v = __algebraic_leader(p)
        equations = [f for f in E if __algebraic_leader(f) <= v]
        guards = [q for q in I if __algebraic_leader(q) <= v]
        J = ideal(R, [equations; __algebraic_discriminant(p)])
        h = prod(guards; init=one(R))
        is_one(saturation(J, ideal(R, [h]))) || return false
    end
    return true
end

@testset "Separate simplicity certificates and the full discriminant subsystem" begin
    R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z])
    P = elem_type(R)
    cases = [(P[], P[]), ([x^2-y], P[]), ([x^2-y], [y]),
             (P[], [x^2-y]), (P[], [(x-y)^2]), ([x-y, y^2-1], P[]),
             ([x*y-1], [y]), ([x, x+y], P[]), ([R(1)], P[]),
             ([x^3+y*x+z], P[])]
    for (E, I) in cases
        expected = reference_algebraic_simple(E, I)
        quasi = is_quasi_simple(E, I)
        @test is_simple(E, I) == expected
        isempty(I) && @test is_simple(E) == expected
        for quasi_first in (false, true)
            system = AlgebraicSystem(R, E, I)
            quasi_first && @test is_quasi_simple(system) == quasi
            @test is_simple(system) == expected
            @test is_quasi_simple(system) == quasi
            @test system.simple === expected
            @test system.quasi_simple === quasi
            for copied in (AlgebraicSystem(system), deepcopy(system))
                @test copied.simple === nothing && copied.quasi_simple === nothing
                @test copied == system && hash(copied) == hash(system)
            end
        end
    end
    # Include constraints up to the original leader, even for a zero discriminant.
    @test __algebraic_discriminant_issue([x^2, y], [y]) === nothing
    issue = __algebraic_discriminant_issue([x^2-y, z], P[])
    @test issue.E_partial == [x^2-y, z]
end
