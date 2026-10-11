using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Reusing a target issue preserves splitting and certification" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    for (system, kind) in [(AlgebraicSystem([x^2-1]), :factorization),
                           (AlgebraicSystem([x^2-y]), :discriminant)]
        issue = __algebraic_target_issue(system; check_discriminants=true)
        @test issue.kind === kind
        reused = __algebraic_step(system; check_discriminants=true, target_issue=issue)
        fresh = __algebraic_step(system; check_discriminants=true)
        @test reused == fresh
        @test all(s -> s.quasi_simple === nothing, reused)
        @test splitting_covers_system(system, reused)
        @test all(!algebraic_branch_intersection(reused[i], reused[j]) for i in eachindex(reused) for j in (i+1):length(reused))
    end
    system = AlgebraicSystem([x*y-1], [x])
    @test __algebraic_target_issue(system; check_discriminants=false).kind === :quasi_simple
end

@testset "Pseudo-division retains its variable and multiplier contract" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    for (p, g) in [(x^4+x+1, a*x^2+b), (x^3+1, a*x+b)]
        q, r = __algebraic_pseudo_divrem(p, g)
        h = __algebraic_initial(g)
        @test any(h^n*p == q*g+r for n in 0:degree(p, 1))
        @test degree(r, 1) < degree(g, 1)
    end
end

@testset "Pseudo-remainders do not require constructing quotients" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    for (f, g) in ((x^5+a*x+1, x^2+b), (x^5+a*x+1, a*x^2+b),
                   (x^3+b, 2*x+a), (x+b, x^2+a), (R(7), a*x+b))
        snapshot = deepcopy((f, g))
        q, r = __algebraic_pseudo_divrem(f, g)
        skipped_q, same_r = __algebraic_pseudo_divrem(f, g; compute_quotient=false)
        @test skipped_q === nothing
        @test same_r == r
        @test any(__algebraic_initial(g)^n*f == q*g+r for n in 0:max(0, degree(f, 1)))
        @test (f, g) == snapshot
    end
end

@testset "Leader tail versus monomial tail" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, y, z) = polynomial_ring(field, [:x, :y, :z])
        p = (y+1)*x^2+x+z
        original = deepcopy(p)
        @test __algebraic_tail(p) == x+z
        @test Oscar.tail(p; ordering=lex(R)) == x^2+x+z
        point = field.([1, -1, -1])
        @test iszero(evaluate(p, point))
        @test iszero(evaluate(__algebraic_tail(p), point))
        @test !iszero(evaluate(Oscar.tail(p; ordering=lex(R)), point))
        @test p == original
    end
end

@testset "Leader pseudo-division versus pseudodivrem" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, y, z) = polynomial_ring(field, [:x, :y, :z])
        f, g = x^4+1, y*x^2
        q, r = __algebraic_pseudo_divrem(f, g)
        @test (q, r) == (x^2, y)
        @test y*f == q*g+r
        U, u = polynomial_ring(R, :u)
        Q, S = pseudodivrem(u^4+1, U(y)*u^2)
        @test (evaluate(Q, x), evaluate(S, x)) == (y^2*q, y^2*r)
    end
end

@testset "Lex remainder: product criterion and general fallback" begin
    R, (x,y,z) = polynomial_ring(QQ, [:x,:y,:z]; internal_ordering=:degrevlex)
    P = elem_type(R)
    batches = [P[], [R(0)], [R(1)], [x^2-y^3], [x^2-y,y^2-z],
               [x*y-1,x^2-z], [x*y-1,x], [x^2-y,y^2-z,x^2-y]]
    polynomials = [R(0),R(1), x^3+x+1, x*y^4+z^3, x^2*y^2-y*z]
    for E in batches
        expected = Oscar.normal_form(polynomials, ideal(R,E); ordering=lex(R))
        @test __algebraic_normal_form_mod_equations(polynomials,E) == expected
        @test [__algebraic_normal_form_mod_equations(p,E) for p in polynomials] == expected
    end
    @test __algebraic_coprime_leading_monomials(R,[x^2-y,y^2-z])
    @test !__algebraic_coprime_leading_monomials(R,[x*y-1,x^2-z])
    # Direct reduction alone is not a normal form for arbitrary generators.
    E = [x*y-1,x]
    @test Oscar.reduce(R(1),E; ordering=lex(R),complete_reduction=true) == 1
    @test __algebraic_normal_form_mod_equations(R(1),E) == 0
end

@testset "Discriminant specialization retains all components" begin
    R,(x,a,b) = polynomial_ring(QQ,[:x,:a,:b])
    for input in (AlgebraicSystem([(x-a)^3]), AlgebraicSystem([],[(x-a)^3]),
                  AlgebraicSystem([x^3+a*x+b]))
        E,I = only(branch_conversion([input]))
        issue = __algebraic_discriminant_issue(E,I)
        @test issue !== nothing
        branches = __algebraic_discriminant_split(E,I,issue)
        @test splitting_covers_system(input,branches)
        @test all(!algebraic_branch_intersection(branches[i],branches[j])
                  for i in eachindex(branches) for j in 1:i-1)
    end
end

@testset "Discriminant specialization avoids radical computation when the degree drops" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    P = elem_type(R)
    p = x^4-a^2
    for input in (AlgebraicSystem([p]), AlgebraicSystem(R, P[], [p])), factorize in (false, true)
        E, I = only(branch_conversion([input]))
        snapshot = deepcopy((E, I))
        issue = __algebraic_discriminant_issue(E, I)
        @test issue !== nothing
        __with_algebraic_cache() do
            branches = __algebraic_discriminant_split(E, I, issue; factorize)
            cache = task_local_storage()[:__algebraic_splitting]
            @test all(key -> first(key) !== :radical_lex, keys(cache))
            @test length(branches) == 2
            @test splitting_covers_system(input, branches)
            @test !algebraic_branch_intersection(branches[1], branches[2])
        end
        @test (E, I) == snapshot
    end

    # Coefficients must be reduced even when the original initial is already one.
    @test __algebraic_specialized_squarefree_part(x^4-a^2, [a], P[], 1) == x
    @test iszero(__algebraic_specialized_squarefree_part(a*x^3, [a], P[], 1))
    @test isone(__algebraic_specialized_squarefree_part(a*x^3+1, [a], P[], 1))

    # Remove the factor b^2 of a lower equation only under the guard b != 0.
    @test __algebraic_specialized_squarefree_part(x^3+a, [b^2*a], P[], 1) === nothing
    @test __algebraic_specialized_squarefree_part(x^3+a, [b^2*a], [b], 1) == x
    # Here b^2 is not a common factor of the equation: a genuine pseudo-step is needed.
    # b^2*(a*x^3+b^2*a-1) - (x^3+b^2)*(b^2*a-1) = x^3.
    scaled = a*x^3+b^2*a-1
    @test __algebraic_specialized_squarefree_part(scaled, [b^2*a-1], P[], 1) === nothing
    @test __algebraic_specialized_squarefree_part(scaled, [b^2*a-1], [b], 1) == x
    # Do not use the target inequation itself to cancel its factor b.
    guard = b*(x^3+a)
    replacement = __algebraic_specialized_squarefree_part(guard, [b*a], [guard], 1)
    @test replacement == b*x
    @test iszero(evaluate(replacement, QQ.([1, 2, 0])))

    # Retain the general fallback when polynomial reduction exposes no degree decrease.
    cubic = x^3+a*x+b
    @test __algebraic_specialized_squarefree_part(cubic, [-4*a^3-27*b^2], P[], 1) === nothing
end

@testset "Separate discriminant factors expose root collisions without radicals" begin
    R, (x, a, b, c) = polynomial_ring(QQ, [:x, :a, :b, :c])
    P = elem_type(R)
    p = (x-a)*(x-b)*(x-c)
    for input in (AlgebraicSystem([p]), AlgebraicSystem(R, P[], [p]))
        E, I = only(branch_conversion([input]))
        snapshot = deepcopy((E, I))
        issue = __algebraic_discriminant_issue(E, I)
        @test issue !== nothing
        disc = __algebraic_squarefree_part(issue.discriminant)
        # The product of all collision conditions does not yet identify a repeated factor.
        @test __algebraic_specialized_squarefree_part(p, [issue.E_partial; disc], I, 1) === nothing
        __with_algebraic_cache() do
            branches = __algebraic_discriminant_split(E, I, issue; factorize=true)
            cache = task_local_storage()[:__algebraic_splitting]
            @test all(key -> first(key) !== :radical_lex, keys(cache))
            @test length(branches) == 2
            @test splitting_covers_system(input, branches)
            @test !algebraic_branch_intersection(branches[1], branches[2])
            E_special, I_special = branches[2]
            selected = issue.kind === :equation ? E_special : I_special
            @test maximum(f -> degree(f, 1), selected; init=0) < degree(p, 1)
        end
        @test (E, I) == snapshot
    end
    # Repeated refinements must exhaust the collisions without losing the generic branch.
    input = AlgebraicSystem(R, P[], [p])
    branches = thomas_decomposition(input)
    @test splitting_covers_system(input, branches)
    @test all(is_simple, branches)
    @test all(!algebraic_branch_intersection(branches[i], branches[j])
              for i in eachindex(branches) for j in 1:i-1)
end

@testset "Products of quadratic guards on the double-sombrero discriminant locus" begin
    K, k = rational_function_field(QQ, :k)
    R, (v, w, q) = polynomial_ring(K, [:v, :w, :q])
    # v=q1_t, w=q2_t, q=q1: the three factors in the inequation of Example 6.4.
    s = q^2*w^2
    B, C = s-k, 3*s-k
    p = (v^2+B)*(3*v^2+B)*(3*v^2+C)
    guards = [q, w]
    monic(f) = divexact(f, R(leading_coefficient(f)))
    for (condition, expected) in (
        (B*C, v*(9*v^4+(21*s-15*k)*v^2-2*k*B)),
        (B, v*(3*v^2+2*k)),
        (C, v*(v^2-2*k/3)*(3*v^2-2*k/3)),
    )
        snapshot = deepcopy((p, condition, guards))
        replacement = __algebraic_specialized_squarefree_part(p, [condition], guards, 1)
        @test replacement !== nothing
        if replacement !== nothing
            @test degree(replacement, 1) < degree(p, 1)
            @test monic(replacement) == monic(expected)
        end
        @test (p, condition, guards) == snapshot
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

@testset "Content, signs and independent polynomial snapshots" begin
    for K in (ZZ, QQ, GF(7))
        R, (x, y) = polynomial_ring(K, [:x, :y])
        for p in (R(0), R(1), R(-2), x+2*y, 2*x+4*y, -2*x-4*y)
            snapshot = deepcopy(p)
            normalized = __algebraic_remove_constant_factor(p)
            @test p == snapshot
            @test __algebraic_remove_constant_factor(normalized) == normalized
            @test iszero(p) || isone(content(normalized))
        end
        @test is_constant(zero(R))
        if K === ZZ || K === QQ
            @test content(2*x+4*y) == 2
            @test __algebraic_remove_constant_factor(-2*x-4*y) == x+2*y
            @test AlgebraicSystem([x+2*y]) == AlgebraicSystem([-6*x-12*y])
        end
        if K === QQ
            p = divexact(x+2*y, 3)
            @test content(p) == QQ(1, 3)
            @test __algebraic_remove_constant_factor(p) == x+2*y
        end
        E, I = [x+2*y], [x-y]
        e, i = __algebraic_copy_system(E, I)
        zero!(e[1]); zero!(i[1])
        @test E == [x+2*y] && I == [x-y]
        source = AlgebraicSystem(E, I)
        copied = AlgebraicSystem(source)
        @test copied == source && hash(copied) == hash(source)
        @test __algebraic_key(copied) == __algebraic_key(E, I)
        # The copy is independent of the source; its key may share its own stored data.
        @test first(__algebraic_key(copied).equations) !== first(__algebraic_key(source).equations)
        working = copied.equations
        zero!(first(working))
        @test source.equations == E
        @test copied.equations == E
        @test __algebraic_key(copied) == __algebraic_key(E, I)
    end
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    E = [x+y]
    missing = splitting_covers_system(E, []; return_remainder=true, vector_output=true)
    @test !missing.covers && length(missing.missing) == 1
    zero!(missing.missing[1][1][1])
    @test E == [x+y]
    branches = [([x+y], elem_type(R)[])]
    extra = splitting_covers_system([R(1)], branches; return_remainder=true, vector_output=true)
    @test !extra.covers && length(extra.extra) == 1
    zero!(extra.extra[1][1][1])
    @test branches[1][1] == [x+y]
end

@testset "Autoreduction reaches a stable result after successive cancellations" begin
    for K in (QQ, GF(7)), ordering in (:lex, :degrevlex)
        R, (x, y, z) = polynomial_ring(K, [:x, :y, :z]; internal_ordering=ordering)
        P = elem_type(R)
        for (E, I) in (([x-y^2, y-z^2, z-1], P[]),
                       ([x^2*y^3], [y]), ([x-y, y-1], [x+y+1]))
            original = deepcopy((E, I))
            clean = __algebraic_cleanup(E, I)
            @test clean !== nothing
            @test __algebraic_cleanup(clean...) == clean
            @test splitting_covers_system(E, I, [clean])
            @test (E, I) == original
        end
        @test __algebraic_cleanup([x*y-1, y], P[]) === nothing
        @test __algebraic_cleanup([x-y], [x-y]) === nothing
        @test __algebraic_cleanup([x], [x+1]) == ([x], P[])
    end
end

@testset "Unified splitting preserves solutions and disjointness" begin
    R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z])
    P = elem_type(R)
    cases = [([x^2-y], P[]), (P[], [x^2-y]),
             ([y*x^2+z*x+1], P[]), (P[], [y*x^2+z*x+1]),
             ([(x-y)^3], P[]), (P[], [(x-y)^3]),
             ([x^3+y*x+z], P[]), ([x^2-y, x-y], P[])]
    for (E, I) in cases, factorization in (false, true)
        input = AlgebraicSystem(R, E, I)
        for split in (quasi_simple_splitting, thomas_decomposition)
            branches = split(input; factorization)
            @test all(is_quasi_simple, branches)
            if split === thomas_decomposition
                @test all(s -> s.simple === true, branches)
                @test all(s -> reference_algebraic_simple(s.equations, s.inequations), branches)
            end
            @test splitting_covers_system(input, branches)
            @test all(!algebraic_branch_intersection(branches[i], branches[j])
                      for i in eachindex(branches) for j in 1:i-1)
            @test input.equations == E && input.inequations == I
            @test input.simple === nothing && input.quasi_simple === nothing
        end
    end
    input = AlgebraicSystem([x^2-y])
    @test quasi_simple_splitting(input) == quasi_simple_splitting(input)
    @test __quasi_simple_splitting(input; simple=true) == thomas_decomposition(input)
    @test thomas_decomposition(input; vector_output=true) == branch_conversion(thomas_decomposition(input))
    @test all(s -> s.simple === true, thomas_decomposition(input; simple=false))
    @test all(s -> s.simple === nothing, quasi_simple_splitting(input; simple=true))
    @test only(thomas_decomposition(AlgebraicSystem(R))).simple === true
    S, (u, v) = polynomial_ring(GF(7), [:u, :v])
    @test all(is_quasi_simple, quasi_simple_splitting([u^2-v]))
    @test_throws ArgumentError thomas_decomposition([u^2-v])
    @test_throws ArgumentError is_simple(AlgebraicSystem([u^2-v]))
end

@testset "Algebraic quasi-simple splitting does not impose differential separant conditions" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    source = AlgebraicSystem([x^2 - y])
    @test __algebraic_target_issue(source; check_discriminants=false, factorize=false) === nothing
    branch = only(quasi_simple_splitting(source; factorization=false))
    @test branch.equations == [x^2 - y]
    @test isempty(branch.inequations)
    @test is_quasi_simple(branch)
    @test !is_simple(branch)

    S, (u, v) = polynomial_ring(GF(7), [:u, :v])
    inseparable = u^7 - v
    @test iszero(derivative(inseparable, 1))
    source_positive = AlgebraicSystem([inseparable])
    @test __algebraic_target_issue(source_positive; check_discriminants=false, factorize=false) === nothing
    branch_positive = only(quasi_simple_splitting(source_positive; factorization=false))
    @test branch_positive.equations == [inseparable]
    @test isempty(branch_positive.inequations)
    @test is_quasi_simple(branch_positive)
end

@testset "Cleanup, nonemptiness and detached factor lists" begin
    for field in (QQ, GF(7))
        R, (x, y, z) = polynomial_ring(field, [:x, :y, :z])
        P = elem_type(R)
        cases = [(P[], P[]), ([R(0)], [R(1)]), ([R(1)], P[]),
                 (P[], [R(0)]), ([x-y, y-z, z-1], [x+1]),
                 ([x, y], [x+y]), ([x-y], [x-y+1]),
                 ([x^2-y, y-z], [z]), ([x*y], [y]),
                 ([x*y-1, x], P[]), ([x^2-y], [y, y]),
                 ([x^2+y^3, y^2-z^3], [x+y^3])]
        values = field === QQ ? QQ.(-1:1) : field.(0:6)
        satisfies(E, I, point) = all(p -> iszero(evaluate(p, point)), E) &&
            all(p -> !iszero(evaluate(p, point)), I)
        for (E, I) in cases
            original = deepcopy((E, I))
            J = ideal(R, E)
            expected = !is_one(saturation(J, ideal(R, [prod(I; init=one(R))])))
            @test __algebraic_branch_has_solution(E, I) == expected
            branches = __algebraic_normalized_nonempty([(E, I), (E, I)])
            @test isempty(branches) == !expected
            @test length(branches) <= 1
            @test all(Iterators.product(values, values, values)) do coordinates
                point = collect(coordinates)
                satisfies(E, I, point) == any(b -> satisfies(b..., point), branches)
            end
            @test (E, I) == original
        end
        branches = [([x], [y]), ([2*x], [3*y]), ([y], P[]), ([x, x], [y])]
        original = deepcopy(branches)
        retained = field === QQ ? [1, 3] : [1, 2, 3]
        @test __algebraic_deduplicate_branch_list(branches) == branches[retained]
        @test branches == original
        @test __algebraic_deduplicate_branch_list(empty(branches)) isa typeof(branches)
        __with_algebraic_cache() do
            p = x^2*y^3
            snapshot = deepcopy(p)
            factors = __algebraic_nonconstant_factors(p)
            @test __algebraic_reduce_polynomial(p, P[], P[]) == p
            @test __algebraic_reduce_polynomial(p, P[], [x]) == y
            @test __algebraic_reduce_polynomial(p, P[], [x, y]) == one(R)
            @test __algebraic_nonconstant_factors(p) == factors
            @test p == snapshot
        end
    end
end

@testset "Object-based steps preserve values, ownership and structural keys" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, y, z) = polynomial_ring(field, [:x, :y, :z]; cached=false)
        P = elem_type(R)
        inputs = [(P[], P[]), ([x^2-1], P[]), ([x^2-y, x-y], P[]),
                  ([x*y-1], [x]), (P[], [x, x+y]), ([y], [x*y+1]),
                  ([x^2+y^3, y^2-z^3], [x+y^3]), ([x^2-y], P[])]
        for (E, I) in inputs, factorize in (false, true)
            original = deepcopy((E, I))
            source = AlgebraicSystem(R, E, I)
            source_hash = hash(source)
            source_key = __algebraic_key(AlgebraicSystem(source))
            issue = __algebraic_target_issue(source; check_discriminants=false, factorize)
            certificate = source.quasi_simple
            children = __algebraic_step(source; factorize, target_issue=issue)
            vectors = __algebraic_step(E, I; factorize, target_issue=issue)
            @test (E, I) == original
            @test (source.equations, source.inequations) == original
            @test source.quasi_simple === certificate
            @test hash(source) == source_hash
            @test __algebraic_key(source) == source_key
            @test (children === nothing) == (vectors === nothing)
            children === nothing && continue
            @test branch_conversion(children) == vectors
            @test splitting_covers_system(source, children)
            @test all(child -> child.quasi_simple === nothing, children)
            for child in children
                # The optimized key must agree with the general public constructor.
                reference = AlgebraicSystem(child.ring, child.equations, child.inequations)
                @test child == reference
                @test hash(child) == hash(reference)
                detached = child.equations
                isempty(detached) || zero!(first(detached))
                empty!(detached)
                @test child.equations == reference.equations
            end
            for (e, i) in vectors
                isempty(e) || zero!(first(e))
                isempty(i) || zero!(first(i))
                empty!(e); empty!(i)
            end
            @test (source.equations, source.inequations) == original
        end
        # Cleanup provides canonical lists even with scalar factors and duplicates.
        for (E, I) in (([2*x, -3*x, R(0)], [2*y, y, R(1)]),
                       ([R(0)], [R(1)]), (P[], P[]))
            source = AlgebraicSystem(R, E, I)
            cleaned = __algebraic_cleanup(source)
            e, i = __algebraic_cleanup(E, I)
            reference = AlgebraicSystem(R, e, i)
            @test cleaned == reference
            @test hash(cleaned) == hash(reference)
            @test cleaned.quasi_simple === nothing
            @test cleaned.ring === R
        end
        Q, (u, v, w) = polynomial_ring(field, [:x, :y, :z]; cached=false)
        @test __algebraic_step(P[], P[]) === nothing
        @test_throws ArgumentError __algebraic_step([x], [u])
    end
end

@testset "Discriminant children own their data independently of their source and siblings" begin
    R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z])
    for E in ([x^2-y], [x^3+y*x+z])
        source = AlgebraicSystem(E)
        initial = deepcopy(source)
        issue = __algebraic_target_issue(source; check_discriminants=true, factorize=false)
        children = __algebraic_step(source; check_discriminants=true, factorize=false, target_issue=issue)
        @test children !== nothing
        @test source == initial
        @test source.equations == initial.equations
        @test splitting_covers_system(source, children)
        @test all(child -> child.quasi_simple === nothing, children)
        @test all(!algebraic_branch_intersection(children[i], children[j])
                  for i in eachindex(children) for j in 1:i-1)
        for child in children
            @test child == AlgebraicSystem(child)
            @test hash(child) == hash(AlgebraicSystem(child))
        end
    end
    source = AlgebraicSystem([x^2-1], [y])
    children = __algebraic_step(source)
    @test length(children) == 2
    other = deepcopy(children[2])
    key = __algebraic_key(AlgebraicSystem(children[1]))
    # Public working copies must leave both the data and the key unchanged.
    working = children[1].inequations
    zero!(first(working))
    @test children[1].inequations == [y]
    @test __algebraic_key(children[1]) == key
    # Deliberately violate the read-only internal contract only to check that
    # different systems do not share polynomials. This invalidates children[1].
    zero!(first(getfield(children[1], :inequations)))
    @test children[2].inequations == other.inequations
    @test __algebraic_key(children[2]) == __algebraic_key(other)
    @test source.inequations == [y]
    @test key.inequations == (y,)
end

@testset "Algebraic reduction without a context preserves multiplicities" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    E = elem_type(R)[]
    p = 6*(x + y)^4*(x - 1)^2
    @test __algebraic_reduce_polynomial(p, E, E) == __algebraic_remove_constant_factor(p)
end

@testset "Reduce algebraic inequation tails by unit-linear equations" begin
    R, (w, z, x) = polynomial_ring(QQ, [:w, :z, :x])
    P = elem_type(R)
    source = AlgebraicSystem(R, P[], [z + x, x*w + 1])
    original = (source.equations, source.inequations)
    branches = thomas_decomposition(source)
    @test length(branches) == 2
    @test all(is_simple, branches)
    @test splitting_covers_system(source, branches)
    @test all(!algebraic_branch_intersection(branches[i], branches[j])
              for i in eachindex(branches) for j in 1:i-1)
    zero_branches = filter(b -> b.equations == [x], branches)
    @test length(zero_branches) == 1
    if length(zero_branches) == 1
        @test only(zero_branches).inequations == [z]
    end
    @test (source.equations, source.inequations) == original

    # Repeat cleanup when a substitution introduces a previously visited
    # variable. No Groebner basis or nonconstant multiplier is needed.
    for equations in ([x, z - x], [z - x, x])
        cleaned = __algebraic_cleanup(equations, [w + z])
        @test cleaned !== nothing
        if cleaned !== nothing
            @test cleaned[2] == [w]
        end
    end
    @test __algebraic_cleanup([x], [x]) === nothing
    @test __algebraic_cleanup([x], [x + 1]) == ([x], P[])
    @test __algebraic_substitute_linear_equations(w + z, [2*z - x]) == w + QQ(1, 2)*x
    # A nonconstant initial cannot be inverted by the substitution shortcut.
    @test __algebraic_substitute_linear_equations(w + z, [x*z - 1]) == w + z
end

function __pairwise_disjoint_algebraic(branches)
    for i in eachindex(branches), j in (i + 1):length(branches)
        algebraic_branch_intersection(branches[i], branches[j]) && return false
    end
    return true
end

@testset "Literature: algebraic Thomas decomposition (BGLR 2012, Example 2.1)" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    p = x^3 + (3y + 1)*x^2 + (3y^2 + 2y)*x + y^3
    d = 27y^3 - 4y
    input = AlgebraicSystem(R, [p], elem_type(R)[])
    # The input is quasi-simple, but not simple because the discriminant
    # vanishes on a lower-dimensional locus.
    @test is_quasi_simple(input)
    @test !is_simple(input)

    paper_branches = [
        AlgebraicSystem(R, [p], [d]),
        AlgebraicSystem(
            R,
            [6x^2 + (-27y^2 + 12y + 6)*x - 3y^2 + 2y, d],
            elem_type(R)[],
        ),
    ]

    @test all(is_simple, paper_branches)
    @test splitting_covers_system(input, paper_branches)
    @test __pairwise_disjoint_algebraic(paper_branches)

    T = thomas_decomposition(input; factorization=false)
    @test all(is_simple, T)
    @test splitting_covers_system(input, T)
    @test __pairwise_disjoint_algebraic(T)

    # Strong regression against the decomposition shape displayed in the paper.
    # If only this assertion fails while the semantic assertions above pass,
    # the implementation may simply be choosing another valid Thomas decomposition.
    @test length(T) == 2
end

@testset "Literature: generic quadratic (BGLR 2012, Examples 2.5 and 4.1)" begin
    # Paper ordering: a < b < c < x
    R, (x, c, b, a) = polynomial_ring(QQ, [:x, :c, :b, :a])
    p = a*x^2 + b*x + c
    disc_guard = 4a*c - b^2

    input = AlgebraicSystem(R, [p], elem_type(R)[])
    @test !is_quasi_simple(input)
    @test !is_simple(input)

    paper_branches = [
        AlgebraicSystem(R, [p], [disc_guard, a]),
        AlgebraicSystem(R, [2a*x + b, disc_guard], [a]),
        AlgebraicSystem(R, [b*x + c, a], [b]),
        AlgebraicSystem(R, [c, b, a], elem_type(R)[]),
    ]

    @test all(is_simple, paper_branches)
    @test splitting_covers_system(input, paper_branches)
    @test __pairwise_disjoint_algebraic(paper_branches)

    T = thomas_decomposition(input; factorization=false)
    @test all(is_simple, T)
    @test splitting_covers_system(input, T)
    @test __pairwise_disjoint_algebraic(T)
    @test length(T) == 4

    # Example 4.1 repeats the computation after adding a != 0.
    restricted = AlgebraicSystem(R, [p], [a])
    restricted_paper_branches = paper_branches[1:2]

    @test all(is_simple, restricted_paper_branches)
    @test splitting_covers_system(restricted, restricted_paper_branches)
    @test __pairwise_disjoint_algebraic(restricted_paper_branches)

    T_restricted = thomas_decomposition(restricted; factorization=false)
    @test all(is_simple, T_restricted)
    @test splitting_covers_system(restricted, T_restricted)
    @test __pairwise_disjoint_algebraic(T_restricted)
    @test length(T_restricted) == 2
end

@testset "Literature: equation + inequation (BGLR 2012, Example 2.26)" begin
    # Paper ordering: a < x.
    R, (x, a) = polynomial_ring(QQ, [:x, :a])

    input = AlgebraicSystem(R, [x^2 + x + 1], [x + a])
    @test !is_quasi_simple(input)
    @test !is_simple(input)

    paper_branches = [
        AlgebraicSystem(R, [x^2 + x + 1], [a^2 - a + 1]),
        AlgebraicSystem(R, [x - a + 1, a^2 - a + 1], elem_type(R)[]),
    ]

    @test all(is_simple, paper_branches)
    @test splitting_covers_system(input, paper_branches)
    @test __pairwise_disjoint_algebraic(paper_branches)

    T = thomas_decomposition(input; factorization=false)
    @test all(is_simple, T)
    @test splitting_covers_system(input, T)
    @test __pairwise_disjoint_algebraic(T)
    @test length(T) == 2
end

@testset "Literature: circle (Robertz 2018, Example 3.11)" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    p = x^2 + y^2 - 1
    input = AlgebraicSystem(R, [p], elem_type(R)[])

    @test is_quasi_simple(input)
    @test !is_simple(input)

    paper_branches = [
        AlgebraicSystem(R, [p], [y^2 - 1]),
        AlgebraicSystem(R, [x, y^2 - 1], elem_type(R)[]),
    ]

    @test all(is_simple, paper_branches)
    @test splitting_covers_system(input, paper_branches)
    @test __pairwise_disjoint_algebraic(paper_branches)

    T = thomas_decomposition(input; factorization=false)
    @test all(is_simple, T)
    @test splitting_covers_system(input, T)
    @test __pairwise_disjoint_algebraic(T)
    @test length(T) == 2
end

@testset "Literature: stereographic projection (Robertz 2018, Example 3.11)" begin
    # Ordering in the paper: x > y > t.
    R, (x, y, t) = polynomial_ring(QQ, [:x, :y, :t])

    input = AlgebraicSystem(R, [x^2 + y^2 - 1, (1 - y)*t - x], elem_type(R)[])

    paper_branches = [
        AlgebraicSystem(
            R,
            [(1 + t^2)*x - 2t, (1 + t^2)*y - t^2 + 1],
            [t^2 + 1],
        ),
        AlgebraicSystem(R, [x, y - 1], elem_type(R)[]),
    ]

    @test all(is_simple, paper_branches)
    @test splitting_covers_system(input, paper_branches)
    @test __pairwise_disjoint_algebraic(paper_branches)

    T = thomas_decomposition(input; factorization=true)
    @test all(is_simple, T)
    @test splitting_covers_system(input, T)
    @test __pairwise_disjoint_algebraic(T)
    @test length(T) == 2
end

@testset "Prepared head reduction preserves the reduction policy" begin
    R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z])
    P = elem_type(R)
    @test __algebraic_reduce(x+y, [y], P[]) == x+y
    @test __algebraic_reduce(y*x^2+x+1, [y], P[]) == x+1
    # The initial y is not certified nonzero, so no pseudo-step is allowed.
    @test __algebraic_reduce(x^2, [y*x-z], P[]) == x^2
    @test __algebraic_reduce(x^2, [y*x-z], [y]) == z^2
    @test __algebraic_reduce(x^2-y, [x^2-y, x-z, y-1, z-2], P[]) == R(1)
    # Repeated leaders retain the original equation order on equal-degree ties.
    @test __algebraic_reduction_equation(x^2, [x-y, x-z], P[]) == x-y
    @test __algebraic_reduction_equation(x^2, [x-z, x-y], P[]) == x-z
end

@testset "Reuse cached general factors with independent nonzero certificates" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    P = elem_type(R)
    f, h = a+b, x^2+b
    p, guard = f^2*h^3, f*(b+1)
    snapshot = deepcopy((p, guard))
    task_local_storage(:__algebraic_splitting, nothing) do
        @test __algebraic_strip_cached_guard_factors(p, [f]) === p
    end
    # Use a fresh dictionary for the cache-enabled case, independently of any outer cache.
    task_local_storage(:__algebraic_splitting, Dict{Any,Any}()) do
        cache = task_local_storage()[:__algebraic_splitting]
        @test __algebraic_strip_cached_guard_factors(p, [f]) === p
        @test isempty(cache)
        __algebraic_nonconstant_factors(p)
        # An unfactored product guard does not certify f by equality alone.
        @test __algebraic_strip_cached_guard_factors(p, [guard]) === p
        @test length(cache) == 1
        @test __algebraic_strip_cached_guard_factors(p, [f]) == h

        __algebraic_nonconstant_factors(guard)
        saved_cache = deepcopy(cache)
        @test __algebraic_strip_cached_guard_factors(p, [guard]) == h
        @test __algebraic_strip_cached_guard_factors(p, [f, h]) == one(R)
        @test __algebraic_strip_cached_guard_factors(p, [a]) === p
        # Without an independent guard, an inequation cannot cancel itself.
        @test __algebraic_strip_cached_guard_factors(guard, P[]) === guard
        @test cache == saved_cache

        g = guard*(x^2+1)
        __algebraic_nonconstant_factors(g)
        E, I = [p], [g, guard]
        original = deepcopy((E, I))
        branches = __algebraic_subresultant_pair_step(E, I, 1, 1, 1, :mixed)
        @test splitting_covers_system(AlgebraicSystem(E, I), branches)
        @test (E, I) == original
    end
    @test (p, guard) == snapshot
end

@testset "Cancel monomial factors skipped by cleanup before the subresultant step" begin
    R, (x, a) = polynomial_ring(QQ, [:x, :a])
    h = sum(a^k for k in 0:8)
    E = [a^5*(x^2-1), a^7*(x^2+x-2)]
    I = [a*h]
    original = deepcopy((E, I))
    reduced_E = [x^2-1, x^2+x-2]

    # The nine-term guard is skipped by the earlier divisibility certificate.
    @test length(only(I)) == 9
    @test !__algebraic_guard_certifies_nonzero(a, I)
    task_local_storage(:__algebraic_splitting, Dict{Any,Any}()) do
        clean = __algebraic_cleanup(AlgebraicSystem(E, I))
        @test clean.equations == E
        @test clean.inequations == I
        issue = __algebraic_target_issue(clean; check_discriminants=true, factorize=false)
        @test issue.kind === :quasi_simple && issue.reason === :leaders

        # Remove a^5 and a^7, then compute the GCD in the same driver step.
        child = only(__algebraic_step(clean;
            check_discriminants=true, factorize=false, target_issue=issue))
        @test child.equations == [x-1]
        @test child.inequations == I
        @test clean.equations == E
    end

    # Writing the same nonzero condition as two guards lets cleanup do this earlier.
    split_I = [a, h]
    clean = __algebraic_cleanup(AlgebraicSystem(E, split_I))
    @test Set(clean.equations) == Set(reduced_E)

    expected = AlgebraicSystem([x-1], I)
    for guards in (I, split_I)
        result = thomas_decomposition(E, guards; factorization=false)
        @test length(result) == 1
        @test all(is_simple, result)
        @test splitting_covers_system(AlgebraicSystem(E, guards), result)
        @test splitting_covers_system(expected, result)
        @test all(branch -> splitting_covers_system(branch, [expected]), result)
    end
    @test (E, I) == original
    @test split_I == [a, h]
end

@testset "Cancel cached general factors skipped by cleanup before the subresultant step" begin
    R, (x, a) = polynomial_ring(QQ, [:x, :a])
    # Nine terms, no monomial factor, and irreducible by Eisenstein at 2.
    f = a^8 + 2*sum(a^k for k in 0:7)
    E = [f^2*(x^2-1), f^3*(x^2+x-2)]
    I = [f]
    original = deepcopy((E, I))

    @test length(f) == 9
    @test !__algebraic_guard_certifies_nonzero(f, I)
    @test isempty(__algebraic_nonzero_monomial_variables(I))
    task_local_storage(:__algebraic_splitting, Dict{Any,Any}()) do
        # Populate the cache through ordinary cleanup, without seeding factors by hand.
        clean = __algebraic_cleanup(AlgebraicSystem(E, I))
        @test clean.equations == E
        @test clean.inequations == I
        issue = __algebraic_target_issue(clean; check_discriminants=true, factorize=false)
        @test issue.kind === :quasi_simple && issue.reason === :leaders

        # Cancel f using cached factors, then compute the GCD in the same step.
        # Monomial cancellation leaves these inputs unchanged.
        child = only(__algebraic_step(clean;
            check_discriminants=true, factorize=false, target_issue=issue))
        @test child.equations == [x-1]
        @test child.inequations == I
        @test clean.equations == E
    end

    expected = AlgebraicSystem([x-1], I)
    result = thomas_decomposition(E, I; factorization=false)
    @test length(result) == 1
    @test all(is_simple, result)
    @test splitting_covers_system(AlgebraicSystem(E, I), result)
    @test splitting_covers_system(expected, result)
    @test all(branch -> splitting_covers_system(branch, [expected]), result)
    @test (E, I) == original
end

@testset "Continue cancelled pairs only while their leaders remain unchanged" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])

    # Equal equations after cancellation need only one copy; no GCD is needed.
    E, I = [a^2*(x^2-1), a^3*(x^2-1)], [a]
    original = deepcopy((E, I))
    branches = __algebraic_subresultant_pair_step(E, I, 1, 2, 1, :equations)
    @test branches == [([x^2-1], I)]
    @test (E, I) == original

    # Equal equation/inequation polynomials certify an inconsistent branch.
    E, I = [a^2*(x^2-1)], [a^3*(x^2-1), a]
    original = deepcopy((E, I))
    @test isempty(__algebraic_subresultant_pair_step(E, I, 1, 1, 1, :mixed))
    @test (E, I) == original

    # Cancellation exposes a linear equation. Reuse its Euclidean split,
    # including the exceptional case a = 0, instead of recursing into the GCD step.
    E, I = [x^2*(a*x+b), x^2+1], [x]
    original = deepcopy((E, I))
    branches = __algebraic_subresultant_pair_step(E, I, 1, 2, 1, :equations)
    expected = __algebraic_same_leader_equations([a*x+b, x^2+1], I, 1, 2)
    @test branches == expected
    @test splitting_covers_system(AlgebraicSystem(E, I), branches)
    @test all(!algebraic_branch_intersection(AlgebraicSystem(left...), AlgebraicSystem(right...))
              for (i, left) in pairs(branches) for right in branches[i+1:end])
    @test (E, I) == original

    # A mixed pair keeps the updated inequation when computing the cofactor.
    E, I = [a^2*(x^2-1)], [a^3*(x+1), a]
    original = deepcopy((E, I))
    branches = __algebraic_subresultant_pair_step(E, I, 1, 1, 1, :mixed)
    @test branches == [([x-1], [a])]
    @test splitting_covers_system(AlgebraicSystem(E, I), branches)
    @test (E, I) == original

    # The equation a*x^2 = 0 becomes a = 0. Return that lower equation
    # without taking a GCD in x that would split again on a.
    E, I = [a*x^2, x^2+x+1], [x]
    original = deepcopy((E, I))
    branches = __algebraic_subresultant_pair_step(E, I, 1, 2, 1, :equations)
    @test Set(first(only(branches))) == Set([a, x^2+x+1])
    @test last(only(branches)) == I
    @test splitting_covers_system(AlgebraicSystem(E, I), branches)
    @test (E, I) == original

    # A nonzero constant equation is rejected by normalization before any GCD.
    E, I = [x^2, x^2+1], [x]
    original = deepcopy((E, I))
    @test isempty(__algebraic_subresultant_pair_step(E, I, 1, 2, 1, :equations))
    @test (E, I) == original
end
