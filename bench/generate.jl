using Butterfly
using CSV
using Random

Random.seed!(777)

simdata = generate_simdata(100)
CSV.write(joinpath(@__DIR__, "benchmark_data.csv"), simdata)
