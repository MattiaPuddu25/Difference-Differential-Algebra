# Execute and verify the REPL examples using only Julia and OSCAR dependencies.
using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

function read_doctest(body::AbstractString)
    lines = split(String(body), '\n'; keepempty=true)
    nonempty = filter(line -> !isempty(strip(line)), lines)
    isempty(nonempty) && return String(body)
    indent = minimum(length(line) - length(lstrip(line)) for line in nonempty)
    return join([isempty(strip(line)) ? "" : chop(line; head=indent, tail=0) for line in lines], '\n')
end

function check_doc_examples()
    root = normpath(joinpath(@__DIR__, ".."))
    files = String[]
root = normpath(joinpath(@__DIR__, ".."))
code_root = joinpath(root, "Code")
for (dir, _, filenames) in walkdir(code_root)
        for filename in filenames
            endswith(filename, ".jl") || continue
            push!(files, joinpath(dir, filename))
        end
    end
    sort!(files)
    count = 0
    @testset "OSCAR dev documentation examples" begin
        for path in files
            source = read(path, String)
            for block in eachmatch(r"```jldoctest[^\n]*\n(.*?)```"s, source)
                body = read_doctest(block.captures[1])
                prompts = collect(eachmatch(r"^julia> (.*)(?:\n|$)"m, body))
                for (i, prompt) in enumerate(prompts)
                    stop = i < length(prompts) ? prompts[i+1].offset : ncodeunits(body)+1
                    start = prompt.offset + ncodeunits(prompt.match)
                    lines = split(SubString(body, start, stop-1), '\n'; keepempty=true)
                    code = String(prompt.captures[1])
                    while !isempty(lines) && startswith(first(lines), "       ")
                        code *= "\n" * chop(popfirst!(lines); head=7, tail=0)
                    end
                    expected = strip(join(lines, "\n"))
                    exception = nothing
                    actual = mktemp() do temporary, io
                        redirect_stdout(io) do
                            try
                                value = include_string(@__MODULE__, code, path)
                                if value !== nothing && !endswith(strip(code), ";")
                                    show(IOContext(io, :limit => true, :displaysize => (24, 80), :module => @__MODULE__), MIME"text/plain"(), value)
                                    println(io)
                                end
                            catch err
                                while err isa LoadError
                                    err = err.error
                                end
                                exception = "ERROR: " * sprint(showerror, err)
                            end
                        end
                        flush(io)
                        strip(read(temporary, String))
                    end
                    count += 1
                    if exception !== nothing
                        occursin(exception, expected) || println(stderr,
                            "Unexpected exception in ", relpath(path, root), ": ", code, "\n", exception)
                        @test occursin(exception, expected)
                    else
                        if actual != expected
                            println(stderr, "Example mismatch in ", relpath(path, root), ": ", code)
                        end
                        @test actual == expected
                    end
                end
            end
        end
    end
    return count
end

check_doc_examples()