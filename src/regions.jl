export evolve

const n_bins = 5
const ΔlnA = 0.5
const max_area = 100

const τ₁ = 5
const τ₂ = 15

const prob = 0.001
const nlon = 36
const nlat = 16

const dcon = 2sinh(ΔlnA / 2)
const fact = exp.(ΔlnA .* (0:n_bins - 1))
const ftot = sum(fact)
const bipole_widths = sqrt.(max_area ./ fact)



""" 
    evolve(;
        butterfly = true,
        activity_rate = 1,
        cycle_length = 11,
        cycle_overlap = 2,
        max_ave_lat = 35,
        min_ave_lat = 7,
        tsim = 3650,
        tstart = 0)

Simulates the emergence and evolution of starspots. 

Output is a list of active regions.

# Parameters
* butterfly = bool - have spots decrease from maxlat to minlat or be randomly located in latitude
* activityrate = Number of magnetic bipoles, normalized such that for the Sun, activityrate = 1.
* cycle_length - length of cycle in years (Sun is 11)
* cycle_overlap - overlap of cycles in years
* max_ave_lat = maximum average latitude of spot emergence (deg)
* min_ave_lat = minimum average latitutde of emergence (deg)
* tsim = how many days to emerge spots for
* tstart = First day to simulate bipoles

Based on Section 4 of van Ballegooijen 1998, ApJ 501: 866
and Schrijver and Harvey 1994, SoPh 150: 1S
Written by Joe Llama (joe.llama@lowell.edu) V 11/1/16
*Converted to Python 3 9/5/2017*

According to Schrijver and Harvey (1994), the number of active regions
emerging with areas in the range [A, A+dA] in time interval dt is given by

    n(A, t) dA dt = a(t) A^(-2) dA dt,

where A is the "initial" bipole area in square degrees, and t is the time
in days; a(t) varies from 1.23 at cycle minimum to 10 at cycle maximum.

The bipole area is the area with the 25-Gauss contour in the "initial"
state, i.e., at the time of maximum development of the active region.
The assumed peak flux density in the initial state is 100 G, and
width = 0.4bsiz.
"""
function evolve(;
    butterfly = true,
    activity_rate = 1,
    cycle_length = 11,
    cycle_overlap = 2,
    max_ave_lat = 35,
    min_ave_lat = 7,
    tsim = 3650,
    tstart = 0)

    amplitude = 10 * activity_rate
    cycle_length_days = cycle_length * 365
    nclen = (cycle_length + cycle_overlap) * 365

    τ = fill(Float64(τ₂), nlon, nlat, 2)

    lat_width = 7 # degrees
    lat_max = max_ave_lat + lat_width
    lat_min = max(min_ave_lat - lat_width, 0)
    dlat = (lat_max - lat_min) / nlat
    dlon = 360 / nlon

    n_count = 0
    n_current_cycle = 0

    spots = Spot[]

    Nday = repeat(0:tsim - 1, inner = 2)
    Icycle = repeat([0, 1], tsim)

    n_current_cycle = Nday .÷ cycle_length_days
    Nc = n_current_cycle .- Icycle
    Nstart = trunc.(Int, cycle_length_days .* Nc)
    phase = @. (Nday - Nstart) / nclen % 1
    ru0_tot = @. amplitude * sin(π * phase)^2 * dcon / max_area

    lats = active_latitudes.(min_ave_lat, max_ave_lat, phase, butterfly = butterfly)
    latavg = reshape([l[1] for l in lats], (1, length(lats)))
    latrms = reshape([l[2] for l in lats], (1, length(lats)))
    lat_bins = 0:nlat - 1
    lat_bins_matrix = repeat(lat_bins, 1, length(Nday))

    p = @. exp(-((lat_min + (lat_bins_matrix + 0.5) * dlat - latavg) / latrms)^2)
    for (i_count, nday) in enumerate(Nday)
        τ .+= 1

        rc0 = zeros(nlon, nlat, 2)
        index = τ₁ .< τ .< τ₂
        rc0[index] .= prob / (τ₂ - τ₁)

        psum = sum(p[:, i_count])
        if psum == 0
            ru0 = p[:, i_count]
        else
            ru0 = ru0_tot[i_count] .* p[:, i_count] ./ (2nlon * psum)
        end

        for k in [0, 1], j in lat_bins
            r0 = ru0[j + 1] .+ rc0[:, j + 1, k + 1]

            rtot = sum(r0)
            sumv = rtot * ftot
            x = rand()

            # choose to emerge spot
            sumv > x || continue
            cum_sum = rtot .* cumsum(fact)
            nb = findfirst(cum_sum .≥ x)
            sumb = cum_sum[max(nb - 1, 1)]
            
            cum_sum = sumb .+ fact[nb] .* cumsum(r0)
            i = findfirst(cum_sum .≥ x)
            sumb = cum_sum[i]

            lon = dlon * (rand() + i - 1)
            lat = lat_min + dlat * (rand() + j)
            
            nday > tstart || continue

            flux_dist_width = 0.4 * bipole_widths[nb]
            width_thresh = 4.0

            Bmax = 250
            Φmax = Bmax * (flux_dist_width / width_thresh)^2

            lat_rad = deg2rad(lat)
            lon_rad = deg2rad(lon)

            push!(spots, Spot(nday, (1 - 2 * k) * lat_rad, lon_rad, Φmax, 0.0))

            n_count += 1
            
            if nb < 2
                τ[i, j + 1, k + 1] = 0
            end

        end
    end

    return spots
end


function active_latitudes(min_lat, max_lat, phase; butterfly = true)
    butterfly ? _exp_lats(min_lat, max_lat, phase) : _rand_lats(min_lat, max_lat, phase)
end

function _exp_lats(min_lat, max_lat, phase)
    phase_scale = 1 / log(max_lat / min_lat)
    lat_avg = max_lat * exp(-phase / phase_scale)
    lat_rms = max_lat / 5 - phase * (max_lat - min_lat) / 7

    return lat_avg, lat_rms
end

function _rand_lats(min_lat, max_lat, phase)
    lat_avg = (max_lat + min_lat) / 2
    lat_rms = (max_lat - min_lat)

    return lat_avg, lat_rms
end