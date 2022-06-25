using LinearAlgebra, CUDA, BenchmarkTools, Krylov, CUDA.CUSPARSE, SparseArrays, Adapt
using Plots

# starting field
function ω_0(x)
    if norm(x) <  0.8
        20*(1 - norm(x)/0.8)
    else
        0
    end
end

# wendland kernel for construction of system interpolation matrix
function wendland(x, y)
    r = norm(x - y) / σ
    if r > 1
        return 0
    end
    return (1-r)^6 * (35*r^2 + 18*r + 3)
end

function dG_dr(r)
    if r == 0 
        return 0
    else
        # @info r
        return -1/r *(2 *r^2 + 4 *r^3 + (25 *r^4)/2 - 4 *r^5 + (5 *r^6)/2 - (6 *r^7)/7 + r^8/8 )
    end
end

# velocity kernel
function K_σ(x)
    if norm(x) == 0
        return [0;0]
    end
    return  [x[2];-x[1]]/norm(x) * dG_dr(norm(x))
end

mutable struct Particle
    x::Vector{Float64}
    ω::Float64
end

# need this function to use Particle struct on GPU
function Adapt.adapt_structure(to, prt::Particle)
    x = Adapt.adapt_structure(to, prt.x)
    ω = Adapt.adapt_structure(to, prt.ω)
    Particle(x, ω)
end

function ω_interpolated(x)
    ω = 0.0
    for (index, particle) in enumerate(particles)
        ω += c[index] * wendland(x, particle.x)
    end
    return ω
end

function u_interpolated(x::Vector{Float64})
    u = [0;0]
    for (index, particle) in enumerate(particles)
        u += c[index] .* K_σ(x .- particle.x)
    end
    return u
end

n = 51
h = 2/(n-1)
domain = Matrix(undef, n,n)

x_lin = LinRange(-1,1,n)
y_lin = LinRange(-1,1,n)

# initialize the field
particles = Vector{Particle}(undef, size(x_lin)[1]*size(y_lin)[1])

counter = 1
for (i,x) in enumerate(x_lin) 
    for (j,y) in enumerate(y_lin)
        domain[i,j] = ω_0([2*x;y])
        particles[counter] = Particle([x,y], ω_0([2*x;y]))
        counter +=1
    end
end

t_domain = [0, 0.05]
Δt = 0.01
t_range = range(t_domain[1], t_domain[2], step=Δt) |> collect

σ = 2*h
μ = 0.1

# memory allocations
# sys_matrix = Matrix{Float64}(undef, size(particles)[1], size(particles)[1])
sys_matrix = spzeros(size(particles)[1], size(particles)[1])
b = Vector{Float64}(undef, size(particles)[1])
c = Vector{Float64}(undef, size(particles)[1])
sys_matrix_gpu = CuSparseMatrixCSC(sys_matrix)
b_gpu = CuVector(b)
c_gpu = CuVector(c)

ω_domain_data = []
push!(ω_domain_data, domain)


# main runtime loop CUDA
@time for (index,t) in enumerate(t_range)
    @info "current Time: \t" t

    
    # construct system interpolation matrix
    for i in  1:size(particles)[1]
        for j in i:size(particles)[1]
            sys_matrix_gpu[i,j] = wendland(particles[i].x, particles[j].x)
            sys_matrix_gpu[j,i] = wendland(particles[j].x, particles[i].x)
        end
    end

    # construct solution Vector
    Threads.@threads for i in 1:size(particles)[1]
        b[i] = particles[i].ω
    end

    # compute interpolation coefficients
    c = b\sys_matrix

    # # evolve particles in time 
    Threads.@threads for particle in particles
        # Use classical 4-Runge-Kutta for time-integration
        x = particle.x

        k1 = u_interpolated(x)
        k2 = u_interpolated(x + 0.5*Δt*k1)
        k3 = u_interpolated(x + 0.5*Δt*k2)
        k4 = u_interpolated(x + Δt*k3)
        particle.x = x + Δt * (k1 + 2*k2 + 2*k3 + k4) / 6.0;

        # l1 = ω_interpolated(x)
        # l2 = ω_interpolated(x + 0.5*Δt*k1)
        # l3 = ω_interpolated(x + 0.5*Δt*k2)
        # l4 = ω_interpolated(x + Δt*k3)
        # particle.ω = particle.ω + μ * Δt * (l1 + 2*l2 + 2*l3 + l4) / 6.0;
        
        # particle.ω = particle.ω + μ * Δt * ω_interpolated(x);
    end

    domain = Matrix(undef, n,n)
    for (i,x) in enumerate(x_lin) 
        for (j,y) in enumerate(y_lin)
            domain[i,j] = ω_interpolated([x;y])
        end
    end
    # @info "Test"
    push!(ω_domain_data, domain)

    
end


# # main runtime loop
# @time for (index,t) in enumerate(t_range)
#     @info "current Time: \t" t

    
#     # construct system interpolation matrix
#     Threads.@threads for i in  1:size(particles)[1]
#         for j in i:size(particles)[1]
#             sys_matrix[i,j] = wendland(particles[i].x, particles[j].x)
#             sys_matrix[j,i] = wendland(particles[j].x, particles[i].x)
#         end
#     end

#     # construct solution Vector
#     Threads.@threads for i in 1:size(particles)[1]
#         b[i] = particles[i].ω
#     end

#     # compute interpolation coefficients
#     c = b\sys_matrix

#     # # evolve particles in time 
#     Threads.@threads for particle in particles
#         # Use classical 4-Runge-Kutta for time-integration
#         x = particle.x

#         k1 = u_interpolated(x)
#         k2 = u_interpolated(x + 0.5*Δt*k1)
#         k3 = u_interpolated(x + 0.5*Δt*k2)
#         k4 = u_interpolated(x + Δt*k3)
#         particle.x = x + Δt * (k1 + 2*k2 + 2*k3 + k4) / 6.0;

#         # l1 = ω_interpolated(x)
#         # l2 = ω_interpolated(x + 0.5*Δt*k1)
#         # l3 = ω_interpolated(x + 0.5*Δt*k2)
#         # l4 = ω_interpolated(x + Δt*k3)
#         # particle.ω = particle.ω + μ * Δt * (l1 + 2*l2 + 2*l3 + l4) / 6.0;
        
#         # particle.ω = particle.ω + μ * Δt * ω_interpolated(x);
#     end

#     domain = Matrix(undef, n,n)
#     for (i,x) in enumerate(x_lin) 
#         for (j,y) in enumerate(y_lin)
#             domain[i,j] = ω_interpolated([x;y])
#         end
#     end
#     # @info "Test"
#     push!(ω_domain_data, domain)

    
# end

# # create animation from the calculated data
# anim = @animate for i in 1:size(ω_domain_data)[1]
#     contour(ω_domain_data[i], fill=true)
#     # heatmap(ω_domain_data[i])
# end
# gif(anim, "test.gif", fps = size(ω_domain_data)[1])