using Test
const QUIET_TEST_OUTPUT = true

function run_test_file(f, filename)
    result = if !QUIET_TEST_OUTPUT
        @testset "$filename" begin f() end
    else
        mktemp() do _, io
            failed = true
            try
                file_tests = redirect_stdout(io) do
                    @testset "$filename" begin f() end
                end
                counts = Test.get_test_counts(file_tests)
                failed = counts.fails + counts.errors + counts.cumulative_fails + counts.cumulative_errors > 0
                return file_tests
            finally
                flush(io)
                if failed
                    seekstart(io)
                    print(stdout, read(io, String))
                    flush(stdout)
                end
            end
        end
    end
    # Print this file's counts and elapsed time before starting the next file.
    Test.print_test_results(result)
    flush(stdout)
    return result
end

@testset "Difference-Differential-Algebra on OSCAR dev" begin
    # Each Code file has a matching test file and an independent module.
    for filename in (
        "Test-Polynomial-Systems.jl",
        "Test-Parent-Checks.jl",
        "Test-Combinatorics-Janet.jl",
        "Test-Subresultants.jl",
        "Test-Algebraic-Simple-Systems.jl",
        "Test-Algebraic-Thomas-Decomposition.jl",
        "Test-Covering-Checks.jl",
        "Test-Action-Janet.jl",
        "Test-Normal-Form.jl",
        "Test-Differential-Simple-Systems.jl",
        "Test-Differential-Thomas-Decomposition.jl",
        "Test-Optimizations.jl",
        "Test-Doctests.jl",
        )
        run_test_file(filename) do
            scope = Module(gensym(:ProjectTests))
            Core.eval(scope, :(include(path) = Base.include(@__MODULE__, path)))
            Base.include(scope, joinpath(@__DIR__, filename))
        end
    end
end