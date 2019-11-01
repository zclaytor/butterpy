using NPZ
using CSV
using Plots

simdata = CSV.read(joinpath(@__DIR__, "benchmark_data.csv"))
julia_times = CSV.read(joinpath(@__DIR__, "julia_times.csv"))
python_times = CSV.read(joinpath(@__DIR__, "python_times.csv"))

# Global times
jtime = julia_times.time
ptime = python_times.time
ratios = ptime ./ jtime

# Read in julia DFs
jtime_filtered = []
ptime_filtered = []

for f in readdir(joinpath(@__DIR__, "data"))
    m = match(r"(julia|python)_(\d+)\.npy", f)
    isnothing(m) && continue
    it = parse(Int, m.captures[2])
    data = npzread(joinpath(@__DIR__, "data", f))
    if m.captures[1] == "julia" && !all(iszero.(data))
        push!(jtime_filtered, jtime[it])
    elseif m.captures[1] == "python" && !all(iszero.(data))
        push!(ptime_filtered, ptime[it+1])
    end
end

ratios_filtered = ptime_filtered ./ jtime_filtered

# @layout [ a b ;
#            c   ]

histogram(jtime, alpha=0.3)
histogram!(jtime_filtered)

histogram(ptime, alpha=0.3)
histogram!(ptime_filtered)

using Statistics
scatter(ratios_filtered, yscale=:log10, markerstrokealpha=0, label="")
m = mean(ratios_filtered)
hline!([m], c=1, label="μ=$(round(m, digits=1))")
hline!([1], label="", c=:black, ls=:dash)
ylabel!("Time Ratio (Python/Julia)")

# histogram(ratios, alpha=0.3, xscale=:log10, xlabel="Time ratio in seconds (Python/Julia)")
# histogram!(ratios_filtered)
