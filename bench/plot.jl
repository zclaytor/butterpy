using NPZ
using CSV
using Plots
pyplot()

simdata = CSV.read(joinpath(@__DIR__, "benchmark_data.csv"))
julia_times = CSV.read(joinpath(@__DIR__, "julia_times.csv"))
python_times = CSV.read(joinpath(@__DIR__, "python_times.csv"))

# Global times
jtime = julia_times.time
ptime = python_times.time
ratios = ptime ./ jtime

# @layout [ a b ;
#            c   ]

histogram(jtime, alpha=0.3)

histogram(ptime, alpha=0.3)

using Statistics

scatter(ratios, yscale=:log10, markerstrokealpha=0, label="")
m = mean(ratios)
hline!([m], c=1, label="μ=$(round(m, digits=1))")
hline!([1], label="", c=:black, ls=:dash)
ylabel!("Time Ratio (Python/Julia)")
xlabel!("Run Number")

# histogram(ratios, alpha=0.3, xscale=:log10, xlabel="Time ratio in seconds (Python/Julia)")
