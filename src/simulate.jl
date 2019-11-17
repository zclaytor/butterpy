using Random
using DataFrames
using Distributions

export generate_simdata, simulate

"""
    generate_simdata(n::Integer)

Generate `n` simulation datasets returned in a `DataFrame`.
"""
function generate_simdata(Nlc::Integer)
    inclination = asin.(sqrt.(rand(Nlc)))
    ar = 10 .^ rand(Uniform(-2, 1), Nlc)
    clen = 10 .^ rand(Uniform(0, 1.6), Nlc)
    cover = 10 .^ rand(Uniform(-1, 0.5), Nlc)
    θ_low = rand(Uniform(0, 40), Nlc)
    θ_high = rand.(Uniform.(θ_low .+ 5, 80))
    period = 10 .^ rand(Uniform(-1, 2), Nlc)
    τ_evol = 10 .^ rand(Nlc)
    butterfly = Bool.(rand(Bernoulli(0.8), Nlc))
    Δω = 10 .^ rand(Uniform(-1, 0), Nlc)

    ω = 24.5 ./ period
    Δω .*= ω

    return DataFrame(
        ar = ar,
        clen = clen,
        cover = cover,
        inclination = inclination,
        θ_low = θ_low,
        θ_high = θ_high,
        period = period,
        ω = ω,
        Δω = Δω,
        τ_decay = τ_evol,
        butterfly = butterfly
    )
end

"""
    simulate(::DataFrameRow; duration=3650, cadence=30)

Given a row from a dataframe with simulation data, will simulate the lightcurve modulation over `duration` days every `cadence` minutes.
"""
function simulate(simrow::DataFrameRow; duration = 3650, cadence=30)
    spots = evolve(
            butterfly = simrow.butterfly,
            activity_rate = simrow.ar,
            cycle_length = simrow.clen,
            cycle_overlap = simrow.cover,
            max_ave_lat = simrow.θ_high,
            min_ave_lat = simrow.θ_low,
            tsim = duration)

    length(spots) == 0 && return zeros(length(0:cadence/1440:duration))

    S = SpotDynamics(spots;
        inclination = simrow.inclination,
        ω = simrow.ω,
        Δω = simrow.Δω,
        alpha_med = 3e-4sqrt(simrow.ar),
        τ_decay = simrow.τ_decay)

    return simulate(S, duration=duration, cadence=cadence)
end

"""
    simulate(::DataFrame; duration=3650, cadence=30)

Given a full dataframe with simulation data, will return a `Vector` of lightcurve modulations over `duration` days every `cadence` minutes.
"""
function simulate(simdata::DataFrame; duration=3650, cadence=30)
    dFs = []
    for row in eachrow(simdata)
        df = simulate(row, duration=duration, cadence=cadence)
        push!(dFs, df)
    end
    return dFs
end

"""
    simulate(::SpotDynamics; duration=3650, cadence=30)

Simulate the lightcurve modulation over `duration` days every `cadence` minutes.
"""
function simulate(r::SpotDynamics; duration=3650, cadence=30)
    t = 0:cadence/1440:duration
    dF = Vector{Float64}(undef, length(t))
    Threads.@threads for i in eachindex(t)
        @inbounds dF[i] = modulate(r, t[i])
    end
    return dF
end
