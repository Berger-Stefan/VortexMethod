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

# grid parameter
n = 21  # number of particles
h = 2/(n-1)
σ = 2*h
t_domain = [0,00001]
Δt = 0.00001
t_range = range(t_domain[1], t_domain[2], step=Δt) |> collect


# physical parameter
μ = 0

# plot parameter
domain_scaling = 10 
t_runtime = 5
fps = 50
frames = 5* 50


x_lin = LinRange(-1,1,n)
y_lin = LinRange(-1,1,n)

# x | y | ω
particles = zeros(size(x_lin)[1]*size(y_lin)[1],3)
particles_update = zeros(size(x_lin)[1]*size(y_lin)[1],3)

global cnt = 1
for (i,x) in enumerate(x_lin) 
    for (j,y) in enumerate(y_lin)
        particles[cnt,:] = vcat([x;y], ω_0([x;2*y]))
        global cnt = cnt + 1
    end    
end    


# memory allocations
sys_matrix = Matrix{Float32}(undef, size(particles)[1], size(particles)[1])
b = Vector{Float32}(undef, size(particles)[1])
c = Vector{Float32}(undef, size(particles)[1])
domain = zeros(domain_scaling*n,domain_scaling*n)
ω_domain_data = []

# GPU stuff
sys_matrix_gpu = CuMatrix(sys_matrix)
b_gpu = CuVector(b)
c_gpu = CuVector(c)
particles_gpu = CuMatrix(particles)
particles_update_gpu = CuMatrix(particles)
domain_gpu = CuMatrix(domain)

# Kernel function definitions

n_particles = size(particles)[1]
# this function construct the system matrix and the b-vector on the gpu
function build_system!(sys_matrix, particles, n_particles)
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
    end
    return nothing
end

# @time build_system!(sys_matrix, particles, n_particles
# CUDA.@time @cuda build_system!(sys_matrix_gpu, particles_gpu, n_particles)

# this function evolves the particles in time using a forward integration scheme TODO implement RK4
function evolve_particles!(particles, particles_update, c, n_particles, Δt)
    Threads.@threads for i in 1:n_particles     

        u1_x = 0
        u1_y = 0
        u2_x = 0
        u2_y = 0
        u3_x = 0
        u3_y = 0
        u4_x = 0
        u4_y = 0
    
        # compute coefficients for RK  
        for j in 1:n_particles
            x = particles[i,1]
            y = particles[i,2]
            @inbounds r = sqrt((x-particles[j,1])^2 + (y-particles[j,2])^2)
            if r != 0
                coef = (-1/r *(2 *r^2 + 4 *r^3 + (25 *r^4)/2 - 4 *r^5 + (5 *r^6)/2 - (6 *r^7)/7 + r^8/8 ))
                @inbounds u1_x += c[j] *  particles[i,2]/r * coef
                @inbounds u1_y += c[j] * -particles[i,1]/r * coef
            end
        end
        for j in 1:n_particles
            x = particles[i,1] + 0.5*Δt*u1_x
            y = particles[i,2] + 0.5*Δt*u1_y
            @inbounds r = sqrt((x-particles[j,1])^2 + (y-particles[j,2])^2)
            if r != 0
                coef = (-1/r *(2 *r^2 + 4 *r^3 + (25 *r^4)/2 - 4 *r^5 + (5 *r^6)/2 - (6 *r^7)/7 + r^8/8 ))
                @inbounds u2_x += c[j] *  particles[i,2]/r * coef
                @inbounds u2_y += c[j] * -particles[i,1]/r * coef
            end
        end
        for j in 1:n_particles
            x = particles[i,1] + 0.5*Δt*u2_x
            y = particles[i,2] + 0.5*Δt*u2_y
            @inbounds r = sqrt((x-particles[j,1])^2 + (y-particles[j,2])^2)
            if r != 0
                coef = (-1/r *(2 *r^2 + 4 *r^3 + (25 *r^4)/2 - 4 *r^5 + (5 *r^6)/2 - (6 *r^7)/7 + r^8/8 ))
                @inbounds u3_x += c[j] *  particles[i,2]/r * coef
                @inbounds u3_y += c[j] * -particles[i,1]/r * coef
            end
        end
        for j in 1:n_particles
            x = particles[i,1] + Δt*u3_x
            y = particles[i,2] + Δt*u3_y
            @inbounds r = sqrt((x-particles[j,1])^2 + (y-particles[j,2])^2)
            if r != 0  
                coef = (-1/r *(2 *r^2 + 4 *r^3 + (25 *r^4)/2 - 4 *r^5 + (5 *r^6)/2 - (6 *r^7)/7 + r^8/8 ))
                @inbounds u4_x += c[j] *  particles[i,2]/r * coef
                @inbounds u4_y += c[j] * -particles[i,1]/r * coef
            end
        end

        # evolve the system velocities
        @inbounds particles_update[i,1] = particles[i,1] + Δt/6.0*(u1_x + 2*u2_x + 2*u3_x + u4_x)
        @inbounds particles_update[i,2] = particles[i,2] + Δt/6.0*(u1_y + 2*u2_y + 2*u3_y + u4_y)


        # evolve change of vorticity
        #   | 1 |   
        # 2 | 3 | 4  
        #   | 5 |   
        ω_1 = 0
        ω_2 = 0
        ω_3 = 0
        ω_4 = 0
        ω_5 = 0            
        
        x = particles[i,1]
        y = particles[i,2]

        for k in 1:n_particles
            r = sqrt(( x - particles[k,1])^2 + ( (y+h) - particles[k,2])^2)/σ
            if r < 1
                ω_1 += c[k] * (1-r)^6 * (35*r^2 + 18*r + 3)
            end
        end
           for k in 1:n_particles
            r = sqrt(( x-h - particles[k,1])^2 + ( y - particles[k,2])^2)/σ
            if r < 1
                ω_2 += c[k] * (1-r)^6 * (35*r^2 + 18*r + 3)
            end
        end      
        for k in 1:n_particles
            r = sqrt(( x - particles[k,1])^2 + (y - particles[k,2])^2)/σ
            if r < 1
                ω_3 += c[k] * (1-r)^6 * (35*r^2 + 18*r + 3)
            end
        end    
        for k in 1:n_particles
            r = sqrt(( x+h - particles[k,1])^2 + ( y - particles[k,2])^2)/σ
            if r < 1
                ω_4 += c[k] * (1-r)^6 * (35*r^2 + 18*r + 3)
            end
        end    
        for k in 1:n_particles
            r = sqrt(( x - particles[k,1])^2 + ( (y-h) - particles[k,2])^2)/σ
            if r < 1
                ω_5 += c[k] * (1-r)^6 * (35*r^2 + 18*r + 3)
            end
        end

        @inbounds particles_update[i,3] = particles[i,3] + Δt*μ*(ω_1 + ω_2 -4*ω_3 +ω_4 +ω_5)/(2*h^2)
    end
end

# @time evolve_particles!(particles, particles_update, c, n_particles, Δt)
# CUDA.@time @cuda evolve_particles!(particles_gpu, particles_update_gpu, c_gpu, n_particles, Δt)

function compute_ω!(domain, particles ,c ,n_x , n_y, n_particles, σ)
    Δx = 2/(n_x-1)
    Δy = 2/(n_y-1)
    for i in 1:n_x
        for j in 1:n_y
            for k in 1:n_particles
                r = sqrt((((i-1)*Δx- 1) - particles[k,1])^2 + (((j-1)*Δy - 1)- particles[k,2])^2)/σ
                if r < 1
                    domain[i,j] += c[k] * (1-r)^6 * (35*r^2 + 18*r + 3)
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
@time for (index,t) in enumerate(t_range[1:end-1])
    @info "current time step: \t" t

    # # construct system interpolation matrix 
    # begin # cpu version is faster right now
        # config = launch_configuration(build_system_kernel.fun)
        # threads = min(n_particles, config.threads)
        # blocks = cld(n_particles, threads)
        # build_system_kernel(sys_matrix_gpu, b_gpu, particles_gpu, n_particles; threads, blocks) 
        # end
    @time begin
        build_system!(sys_matrix,  particles, n_particles)   
        b = particles[:,3]
        @info "system build"
    end

    # compute interpolation coefficient  
    CUDA.@time begin 
        sys_matrix_gpu = CuMatrix(sys_matrix)
        b_gpu = CuVector(b)
        c_gpu = sys_matrix_gpu \  b_gpu
        c = Array(c_gpu)
        @info "system solved"
    end

    # evolve particles
    # begin
    #     # config = launch_configuration(evolve_particles_kernel.fun)
    #     # threads = min(n_particles, config.threads)
    #     # blocks = cld(n_particles, threads)
    #     # CUDA.@elapsed evolve_particles_kernel(particles_gpu, particles_update_gpu, c_gpu, n_particles, Δt)
    #     # particles_gpu = particles_update_gpu
    # end

    @time begin 
        particles_update = zeros(n_particles,3)
        evolve_particles!(particles, particles_update, c, n_particles, Δt)
        particles = particles_update 
        @info "particles evolved"
    end

    begin
        # config = launch_configuration(compute_ω_kernel.fun)
        # threads = min(n_particles, config.threads)
        # blocks = cld(n_particles, threads)
        # CUDA.@elapsed compute_ω_kernel(domain_gpu, particles_gpu, c_gpu, n, h, n ,h , n_particles, wendland)
    end
    
    if any(isnan.(particles))
        @warn "Instable, braking"
        break
    end
    
    @time begin 
        domain = zeros(domain_scaling*n,domain_scaling*n)
        compute_ω!(domain, particles, c, domain_scaling*n, domain_scaling*n , n_particles, σ)
        push!(ω_domain_data, Array(domain))
        @info "ω constructed"
    end

    print("\n\n\n")
end

# create animation from the calculated data
anim = @animate for i in 1:round(Int,size(ω_domain_data)[1]/frames+1):size(ω_domain_data)[1]
    # contour(ω_domain_data[], fill=true)
    heatmap(LinRange(-1,1,domain_scaling*n), LinRange(-1,1,domain_scaling*n), ω_domain_data[i],margin = 5Plots.mm)
    title!("t:  " * string(round(i*Δt,digits=4)) *" , h="*string(h)*" , σ="*string(σ)*" , μ="*string(μ)* " , n_particles = "*string(n))
end
gif(anim, "result.gif", fps = 50)
