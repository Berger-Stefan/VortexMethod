using LinearAlgebra, CUDA, BenchmarkTools, Krylov, CUDA.CUSPARSE, SparseArrays, CUDA.CUSOLVER, IterativeSolvers, LinearOperators
using Plots

CUDA.allowscalar(false)

# starting field
function ω_0(x)
    if norm(x) <  0.8
        20*(1 - norm(x)/0.8)
    else
        0
    end
end


n = 3
h = 2/(n-1)
domain = Matrix{Float32}(undef, n,n)

x_lin = LinRange(-1,1,n)
y_lin = LinRange(-1,1,n)

# x | y | ω
particles = zeros(size(x_lin)[1]*size(y_lin)[1],3)
particles_update = zeros(size(x_lin)[1]*size(y_lin)[1],3)

global cnt = 1
for (i,x) in enumerate(x_lin) 
    for (j,y) in enumerate(y_lin)
        domain[i,j] = ω_0([2*x;y])
        particles[cnt,:] = vcat([x;y], ω_0([2*x;y]))
        global cnt = cnt + 1
    end
end
ω_domain_data = []
push!(ω_domain_data, domain)

t_domain = [0, .1]
Δt = 0.01
t_range = range(t_domain[1], t_domain[2], step=Δt) |> collect

σ = 2*h
μ = 0.1

# memory allocations
sys_matrix = Matrix{Float32}(undef, size(particles)[1], size(particles)[1])
# sys_matrix = spzeros(size(particles)[1], size(particles)[1])
b = Vector{Float32}(undef, size(particles)[1])
c = Vector{Float32}(undef, size(particles)[1])

# GPU stuff
sys_matrix_gpu = CuMatrix(sys_matrix)
# sys_matrix_gpu = CuSparseMatrixCSC(sys_matrix)
b_gpu = CuVector(b)
c_gpu = CuVector(c)
particles_gpu = CuMatrix(particles)
particles_update_gpu = CuMatrix(particles)
domain_gpu = CuMatrix(domain)

# Kernel function definitions

n_particles = size(particles)[1]
# this function construct the system matrix and the b-vector on the gpu
function build_system!(sys_matrix ,b ,particles, n_particles)
    for i in  1:n_particles
        for j in i:n_particles
            r = sqrt((particles[i,1] - particles[j,1])^2 + (particles[i,2] - particles[j,2])^2)/σ
            if r > 1
                @inbounds sys_matrix[i,j] = 0
                @inbounds sys_matrix[j,i] = 0
            else
                @inbounds sys_matrix[i,j] = (1-r)^6 * (35*r^2 + 18*r + 3)
                @inbounds sys_matrix[j,i] = sys_matrix[i,j]
            end
        end
        @inbounds b[i] = particles[i,3]
    end
    return nothing
end

# @time build_system!(sys_matrix, b,  particles, n_particles)
# CUDA.@time @cuda build_system!(sys_matrix_gpu, b_gpu, particles_gpu, n_particles)

# this function evolves the particles in time using a forward integration scheme TODO implement RK4
function evolve_particles!(particles, particles_update, c, n_particles, Δt)
    for i in 1:n_particles     

        u1_x = 0
        u1_y = 0

        for j in 1:n_particles
            @inbounds r = sqrt((particles[i,1]-particles[j,1])^2 + (particles[i,1]-particles[j,2])^2)
                if r != 0
                    coef = (-1/r *(2 *r^2 + 4 *r^3 + (25 *r^4)/2 - 4 *r^5 + (5 *r^6)/2 - (6 *r^7)/7 + r^8/8 ))
                    @inbounds u1_x += c[j] *  particles[i,2]/r * coef
                    @inbounds u1_y += c[j] * -particles[i,1]/r * coef
                end
        end
        
        @inbounds particles_update[i,1] = particles[i,1] + Δt*u1_x
        @inbounds particles_update[i,2] = particles[i,2] + Δt*u1_y

        @inbounds particles_update[i,3] = particles[i,3]
    end
end

# @time evolve_particles!(particles, particles_update, c, n_particles, Δt)
# CUDA.@time @cuda evolve_particles!(particles_gpu, particles_update_gpu, c_gpu, n_particles, Δt)

#TODO Fix
function compute_ω!(domain, particles ,c ,n_x , Δx, n_y, Δy, n_particles, σ)
    for i in 1:n_x
        for j in 1:n_y
            for k in 1:n_particles
                r = sqrt((((i-1)*Δx- 1) - particles[k,1])^2 + (((j-1)*Δy - 1)- particles[k,2])^2)
                if r/σ < 1
                    domain[i,j] += c[k] * (1-r/σ)^6 * (35*r/σ^2 + 18*r/σ + 3)
                end
            end
        end
    end
end

# @time compute_ω!(domain, particles, c, n, h, n ,h , n_particles, wendland, σ)
# CUDA.@elapsed @cuda compute_ω!(domain_gpu, particles_gpu, c_gpu, n, h, n ,h , n_particles, wendland, σ)


# # compile kernels
# build_system_kernel = @cuda launch=false  build_system!(sys_matrix_gpu, b_gpu, particles_gpu, n_particles,wendland)
# evolve_particles_kernel = @cuda launch=false evolve_particles!(particles_gpu, particles_update_gpu, c_gpu, n_particles, Δt)
# compute_ω_kernel = @cuda launch=false compute_ω!(domain_gpu, particles_gpu, c_gpu, n, h, n ,h , n_particles, wendland)

# main runtime loop CUDA
@time for (index,t) in enumerate(t_range)
    @info "current Time: \t" t

    # # construct system interpolation matrix 
    # begin # cpu version is faster right now
        # config = launch_configuration(build_system_kernel.fun)
        # threads = min(n_particles, config.threads)
        # blocks = cld(n_particles, threads)
        # build_system_kernel(sys_matrix_gpu, b_gpu, particles_gpu, n_particles; threads, blocks) 
    # end
    build_system!(sys_matrix, b,  particles, n_particles)   

    # compute interpolation coefficient  
    CUDA.@time begin 
        sys_matrix_gpu = CuMatrix(sys_matrix)
        b_gpu = CuVector(b)
        c_gpu = sys_matrix_gpu \  b_gpu
    end
    c = Array(c_gpu)

    # c= sys_matrix\b

    # evolve particles
    # begin
    #     # config = launch_configuration(evolve_particles_kernel.fun)
    #     # threads = min(n_particles, config.threads)
    #     # blocks = cld(n_particles, threads)
    #     # CUDA.@elapsed evolve_particles_kernel(particles_gpu, particles_update_gpu, c_gpu, n_particles, Δt)
    #     # particles_gpu = particles_update_gpu
    # end

    particles_update = zeros(n_particles,3)
    evolve_particles!(particles, particles_update, c, n_particles, Δt)
    particles = particles_update 




    begin
        # config = launch_configuration(compute_ω_kernel.fun)
        # threads = min(n_particles, config.threads)
        # blocks = cld(n_particles, threads)
        # CUDA.@elapsed compute_ω_kernel(domain_gpu, particles_gpu, c_gpu, n, h, n ,h , n_particles, wendland)
    end
    domain = zeros(n,n)
    compute_ω!(domain, particles, c, n, h, n ,h , n_particles, σ)

    push!(ω_domain_data, Array(domain))
end

# create animation from the calculated data
anim = @animate for i in 1:size(ω_domain_data)[1]
    contour(ω_domain_data[i], fill=true)
    # heatmap(ω_domain_data[i])
end
gif(anim, "test.gif", fps = 10)