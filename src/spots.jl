using Random
using Statistics
using LinearAlgebra

"""
Siderial rotation period at the equator
"""
const PROT_SUN = 24.5
const OMEGA_SUN = 2 * π / PROT_SUN / 86400

struct Spot{T<:Number}
    nday::Integer # day index
    lat::T
    lon::T
    Φmax::T
end

struct Spots
    spots::Vector{Spot}
    duration
    inclination
    ω
    Δω
    equatorial_period
    diffrot_func
    τ_emergence
    τ_decay
    area_max
end

function Spots(spots::AbstractVector{Spot};
    duration = maximum([s.nday for s in spots]),
    alpha_med = 0.0001,
    inclination = asin(rand()),
    ω = 2.0,
    Δω = 0.3
    diffrot_func = diffrot,
    τ_decay = 5.0,
    threshold = 0.1)
    
    ω *= OMEGA_SUN
    Δω *= OMEGA_SUN
    eq_per = 2π / ω / 86400
    τ_emergence = max(2.0, eq_per * τ_decay / 5)
    τ_decay *= eq_per


    filt = s -> (s.nday < duration) && (s.ϕmax > threshold)
    spots = filter(filt, spots)
    Φmaxes = [s.Φmax for s in spots]
    area_max = alpha_med .* Φmaxes ./ median(Φmaxes)

    return Spots(spots, duration, inclination, ω, Δω, eq_per, diffrot, τ_emergence, τ_decay, area_max)
end

Base.length(spots::Spots) = length(spots.spots)
Base.size(spots::Spots) = size(spots.spots)
Base.size(spots::Spots, i) = size(spots.spots, i)

"""
    diffrot(ω₀, Δω, lat)

Default differental rotation function

Returns angular velocity as a function of latitude [0°, 90°]
"""
diffrot(ω₀, Δω, lat) = ω₀ - Δω * sin(lat)^2

"""
    modulate(::Spots, time)

Modulate the flux for all spots
"""
function modulate(spots::Spots, time)
    M = length(spots)
    N = length(time)
    lats = [s.lat for s in spots.spots]
    lons = [s.lon for s in spots.spots]

    # Get spot area
    tt = repeat(time, outer=M) .- repeat([s.nday for s in spots.spots], outer=N)
    area = repeat(spots.area_max, outer=N)
    emerge = repeat(spots.τ_emergence, N)
    decay = repeat(spots.τ_decay, N)
    timescale = ifelse.(tt .< 0, emerge, decay)
    area .*= exp(-(tt ./ timescale) .^ 2 ./ 2)

    # rotation rate
    omega = spots.diffrot_func(spots.ω, spots.Δω, lats)
    phase = omega * (time .* 86400)' .+ repeat(lons, outer=N)

    # foreshortening
    cos_beta = cos(spots.inclination) .* repeat(sin.(lats), outer=N) .+ sin(spots.inclination) .* repeat(cos.(lats), outer=N) .* cos.(phase)

    dFlux = -area .* maximum(cos_beta)

    return sum(dFlux)
end



