using Butterfly
using CSV
using DataFrames
using Statistics
using BenchmarkTools
using NPZ
using ProgressMeter

simdata = CSV.read(joinpath(@__DIR__, "benchmark_data.csv"))

datadir = joinpath(@__DIR__, "data")
mkpath(datadir)

times = DataFrame(time = Float64[], nallocs = Int[], datafile = String[])
@showprogress for (i, row) in enumerate(eachrow(simdata))
    b = @benchmarkable simulate($row)
    tune!(b)
    t, df = BenchmarkTools.run_result(b)
    dfile = joinpath(datadir, "julia_$i.npy")
    npzwrite(dfile, df)
    push!(times, [mean(t).time * 1e-9, t.allocs, dfile])
end

CSV.write(joinpath(@__DIR__, "julia_times.csv"), times)
