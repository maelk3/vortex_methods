using LinearAlgebra

# PHYSICAL CONSTANTS
U₀ = 0.1                       # speed
U  = U₀*[1.0, 0.0, 0.0]        # freestream velocity
Ω  = 1.0

ρ = 1.0                        # air density    

# BLADE PROFILE DEFINITION
r_min = 0.2                    # hub radius
r_max = 1.0                    # tip radius
α     = 10*π/180               # blade pitch
c     = 0.1*cos(α)             # chord length
η(x)  = -tan(α)*x              # chord profile

# DISCRETISATION PARAMETERS
rc = 0.00                      # core radius
N = 5                          # chordwise number of panels
M = 20                         # spanwise number of panels

nt = 80                        # number of time steps
Δt = 0.25                      # time intervals

rotation = [1.0 0.0        0.0
            0.0 cos(Ω*Δt) -sin(Ω*Δt)
            0.0 sin(Ω*Δt)  cos(Ω*Δt)]

# RANKINE VORTEX SEGMENT AND RING DEFINITION
function induced_vel_segment(P, A, B, r₀, Γ, rc)
    r₁ = A-P
    r₂ = B-P
    ε = 1e-8
    r = norm(r₁×r₀)+ε
    if r<rc && dot(r₁,r₀)*dot(r₂,r₀) <= 0+ε
        return (Γ/(2π*rc))*(r/rc)*(r₀×r₁)/(norm(r₀×r₁)+ε)
    else
        return (Γ/4π)*(dot(r₀, r₁)/(norm(r₁)+ε) - dot(r₀, r₂)/(norm(r₂)+ε))*(r₁×r₀)/(norm(r₁×r₀)+ε)^2
    end
end

function induced_vel_ring(P, ring, Γ, rc)
    u = zeros(3)
    for i∈1:4
        r₀ = (ring[i,2,:]-ring[i,1,:])/norm(ring[i,2,:]-ring[i,1,:])
        u += induced_vel_segment(P, ring[i,1,:], ring[i,2,:], r₀, Γ, rc)
    end
    return u
end

function induced_vel(P, rings, Γ, rc)
    u = zeros(3)
    for i∈1:(size(rings)[1])
        u += induced_vel_ring(P, rings[i,:,:,:], Γ[i], rc)
            end
    return u
end

spanwise_distribution(x) = r_min + (r_max-r_min)*(cos(π*(1-x))+1)/2 # cosine distribution

# initial lagrangian markers coordinates
x_blade_markers = repeat(η.(range(0,c,length=N+1)), 1, M+1)
y_blade_markers = repeat(spanwise_distribution.(range(0, 1, length=M+1)), 1, N+1)'
z_blade_markers = repeat(range(0, c, length=N+1), 1, M+1)


# blade panel markers
blade_panel_markers = permutedims(cat(x_blade_markers, y_blade_markers, z_blade_markers, dims=3), (3, 1, 2))

# initial collocation points
x_collocation = 0.5*(x_blade_markers[1:end-1,1:end-1] + x_blade_markers[2:end,1:end-1])
y_collocation = 0.5*(y_blade_markers[1:end-1,1:end-1] + y_blade_markers[1:end-1,2:end])
z_collocation = 0.5*(z_blade_markers[1:end-1,1:end-1] + z_blade_markers[2:end,1:end-1])
collocation_points = permutedims(cat(x_collocation, y_collocation, z_collocation, dims=3), (3, 1, 2))

# initial collocation normals
collocation_normals = zeros(3, N, M)
for i∈1:N, j∈1:M
    normal = -(blade_panel_markers[:,i+1,j+1]-blade_panel_markers[:,i,j])×(blade_panel_markers[:,i,j+1]-blade_panel_markers[:,i+1,j])
    collocation_normals[:,i,j] = normal/norm(normal)
end

blade_rings = zeros(N, M, 4, 2, 3)
for i∈1:N, j∈1:M    
    blade_rings[i,j,1,1,:] = blade_panel_markers[:,i,j]
    blade_rings[i,j,1,2,:] = blade_panel_markers[:,i+1,j]
    blade_rings[i,j,2,1,:] = blade_panel_markers[:,i+1,j]
    blade_rings[i,j,2,2,:] = blade_panel_markers[:,i+1,j+1]
    blade_rings[i,j,3,1,:] = blade_panel_markers[:,i+1,j+1]
    blade_rings[i,j,3,2,:] = blade_panel_markers[:,i,j+1]
    blade_rings[i,j,4,1,:] = blade_panel_markers[:,i,j+1]
    blade_rings[i,j,4,2,:] = blade_panel_markers[:,i,j]                    
end

# influence matrix
influence_matrix = zeros(N, M, N, M)
for i∈1:N, j∈1:M, k∈1:N, l∈1:M
    influence_matrix[i,j,k,l] = dot(induced_vel_ring(collocation_points[:,i,j], blade_rings[k,l,:,:,:], 1, rc), collocation_normals[:,i,j])
end

wake_panel_markers = zeros(3, nt+1, M+1)
wake_panel_markers[:,1,:] = blade_panel_markers[:,1,:]

Γ = zeros(N, M)
RHS = zeros(N, M)
wake_Γ = zeros(nt, M)
wake_rings = zeros(nt, M, 4, 2, 3)

for t∈1:nt
    println(t)

    # move the blade with explicit Euler
    for i∈1:N+1, j∈1:M+1
        blade_panel_markers[:,i,j] = rotation*blade_panel_markers[:,i,j]
    end
    
    RHS .= dropdims(mapslices((x -> dot(x, -U)), collocation_normals, dims=(1)), dims=(1))
    for τ∈1:t-1, l∈1:M
        for i∈1:N, j∈1:M
            RHS[i,j] -= dot(collocation_normals[:,i,j], induced_vel_ring(collocation_points[:,i,j], wake_rings[τ,l,:,:,:], wake_Γ[τ,l], rc))
        end
    end

    Γ .= reshape(reshape(influence_matrix, N*M, N*M) \ reshape(RHS, N*M), N, M)
    wake_Γ[t,:] = -Γ[end,:]

    for τ∈1:t-1, j∈1:M+1
        v1 = zeros(3)        
        v2 = zeros(3)        
        v1 = induced_vel(wake_panel_markers[:,τ,j], reshape(blade_rings, N*M, 4, 2, 3), reshape(Γ, N*M), rc)                                     # induced velocity by the blade
        v2 = induced_vel(wake_panel_markers[:,τ,j], reshape(wake_rings[1:t-1,:,:,:,:], (t-1)*M, 4, 2, 3), reshape(wake_Γ[1:t-1,:], (t-1)*M), rc) # induced velocity by the wake
        wake_panel_markers[:, τ, j] += Δt*(v1+v2)
    end

    for τ∈1:t, j∈1:M+1
        wake_panel_markers[:, τ, j] += Δt*U
    end

    # prescribe next trailing edge
    wake_panel_markers[:,t+1,:] = blade_panel_markers[:,1,:]

    # construct trailing edge new wake panels
    for τ∈1:t, j∈1:M
        wake_rings[τ,j,1,1,:] = wake_panel_markers[:,τ+1,j]
        wake_rings[τ,j,1,2,:] = wake_panel_markers[:,τ,j]
        wake_rings[τ,j,2,1,:] = wake_panel_markers[:,τ,j]
        wake_rings[τ,j,2,2,:] = wake_panel_markers[:,τ,j+1]
        wake_rings[τ,j,3,1,:] = wake_panel_markers[:,τ,j+1]
        wake_rings[τ,j,3,2,:] = wake_panel_markers[:,τ+1,j+1]
        wake_rings[τ,j,4,1,:] = wake_panel_markers[:,τ+1,j+1]
        wake_rings[τ,j,4,2,:] = wake_panel_markers[:,τ+1,j]                                
    end

    # update blade rings
    for i∈1:N, j∈1:M    
        blade_rings[i,j,1,1,:] = blade_panel_markers[:,i,j]
        blade_rings[i,j,1,2,:] = blade_panel_markers[:,i+1,j]
        blade_rings[i,j,2,1,:] = blade_panel_markers[:,i+1,j]
        blade_rings[i,j,2,2,:] = blade_panel_markers[:,i+1,j+1]
        blade_rings[i,j,3,1,:] = blade_panel_markers[:,i+1,j+1]
        blade_rings[i,j,3,2,:] = blade_panel_markers[:,i,j+1]
        blade_rings[i,j,4,1,:] = blade_panel_markers[:,i,j+1]
        blade_rings[i,j,4,2,:] = blade_panel_markers[:,i,j]                    
    end

    # update blade normals
    collocation_points .= 0.5*(blade_panel_markers[:,1:end-1,1:end-1] + blade_panel_markers[:,2:end,2:end])

    for i∈1:N, j∈1:M
        normal = (blade_panel_markers[:,i+1,j+1]-blade_panel_markers[:,i,j])×(blade_panel_markers[:,i,j+1]-blade_panel_markers[:,i+1,j])
        collocation_normals[:,i,j] = normal/norm(normal)
    end

    # update influence matrix
    for i∈1:N, j∈1:M, k∈1:N, l∈1:M
        influence_matrix[i,j,k,l] = dot(induced_vel_ring(collocation_points[:,i,j], blade_rings[k,l,:,:,:], 1, rc), collocation_normals[:,i,j])
    end
end

x_max = maximum(wake_panel_markers[1,:,:])

x_range = [-0.4, x_max]
y_range = [-1.2, 1.2]
z_range = [-1.2, 1.2]


a_wake = Int[]
b_wake = Int[]
c_wake = Int[]
intensities = []
for i∈0:M-1, j∈0:nt-1
    push!(a_wake, i*(nt+1)+j)
    push!(b_wake, (i+1)*(nt+1)+j)
    push!(c_wake, (i+1)*(nt+1)+(j+1))
    push!(intensities, wake_Γ[j+1,i+1])
    
    push!(a_wake, i*(nt+1)+j)
    push!(c_wake, i*(nt+1)+(j+1))
    push!(b_wake, (i+1)*(nt+1)+(j+1))
    push!(intensities, wake_Γ[j+1,i+1])
end

a_blade = Int[]
b_blade = Int[]
c_blade = Int[]
for i∈0:M-1, j∈0:N-1
    push!(a_blade, i*(N+1)+j)
    push!(b_blade, (i+1)*(N+1)+j)
    push!(c_blade, (i+1)*(N+1)+(j+1))
    
    push!(a_blade, i*(N+1)+j)
    push!(c_blade, i*(N+1)+(j+1))
    push!(b_blade, (i+1)*(N+1)+(j+1))
end

plotting_backend = "PlotlyJS"
if plotting_backend == "PlotlyJS"
    # PlotlyJS
    using PlotlyJS
    layout = PlotlyJS.Layout(scene=attr(xaxis_range=x_range,
                                        yaxis_range=y_range,
                                        zaxis_range=z_range),
                             scene_camera=attr(
                                 up=attr(x=0, y=0, z=1),
                                 center=attr(x=x_max/4, y=0, z=0),
                                 eye=attr(x=1.25, y=1.25, z=1.25)
                             ),
                             scene_aspectratio=attr(x=x_range[2]-x_range[1], y=y_range[2]-y_range[1], z=z_range[2]-z_range[1]))

    normal_field = PlotlyJS.cone(x=vec(collocation_points[1,:,:]),
                                 y=vec(collocation_points[2,:,:]),
                                 z=vec(collocation_points[3,:,:]),
                                 u=vec(collocation_normals[1,:,:,:]),
                                 v=vec(collocation_normals[2,:,:,:]),
                                 w=vec(collocation_normals[3,:,:,:]),
                                 sizemode="scaled",
                                 sizeref=3,
                                 showscale=false)

    wake_wireframe = GenericTrace{Dict{Symbol,Any}}[]
    for i∈1:nt+1
        line = PlotlyJS.scatter(x=wake_panel_markers[1,i,:],
                                y=wake_panel_markers[2,i,:],
                                z=wake_panel_markers[3,i,:],
                                mode="lines",
                                type="scatter3d",
                                line=attr(color="black", width=2),
                                showlegend=false)
        push!(wake_wireframe, line)
    end

    for j∈1:M+1
        line = PlotlyJS.scatter(x=wake_panel_markers[1,:,j],
                                y=wake_panel_markers[2,:,j],
                                z=wake_panel_markers[3,:,j],
                                mode="lines",
                                type="scatter3d",
                                line=attr(color="black", width=2),
                                showlegend=false)
        push!(wake_wireframe, line)
    end

    blade_wireframe = GenericTrace{Dict{Symbol,Any}}[]
    for i∈1:N+1
        line = PlotlyJS.scatter(x=blade_panel_markers[1,i,:],
                                y=blade_panel_markers[2,i,:],
                                z=blade_panel_markers[3,i,:],
                                mode="lines",
                                type="scatter3d",
                                line=attr(color="blue", width=2),
                                showlegend=false)
        push!(blade_wireframe, line)
    end

    for j∈1:M+1
        line = PlotlyJS.scatter(x=blade_panel_markers[1,:,j],
                                y=blade_panel_markers[2,:,j],
                                z=blade_panel_markers[3,:,j],
                                mode="lines",
                                type="scatter3d",
                                line=attr(color="blue", width=2),
                                showlegend=false)
        push!(blade_wireframe, line)
    end

    blade_surface = PlotlyJS.mesh3d(x=vec(blade_panel_markers[1,:,:]),
                                    y=vec(blade_panel_markers[2,:,:]),
                                    z=vec(blade_panel_markers[3,:,:]),
                                    opacity=1.0,
                                    color="rgb(0, 255, 0)",
                                    i=a_blade,
                                    j=b_blade,
                                    k=c_blade)

    wake_surface = PlotlyJS.mesh3d(x=vec(wake_panel_markers[1,:,:]),
                                   y=vec(wake_panel_markers[2,:,:]),
                                   z=vec(wake_panel_markers[3,:,:]),
                                   opacity=1,
                                   intensity=intensities,                                
                                   intensitymode="cell",
                                   colorbar_title="circulation",
                                   colorscale=[[0, "blue"],
                                               [0.5, "white"],
                                               [1, "red"]],
                                   i=a_wake,
                                   j=b_wake,
                                   k=c_wake)

    
    PlotlyJS.plot(vcat(wake_wireframe, blade_wireframe, [blade_surface, wake_surface]), layout)

elseif plotting_backend == "Plots"
    using Plots
    # pyplot()
    gr()

    plt = Plots.plot(1,
                     type=:path3d,
                     legend=false,
                     xlims=x_range,
                     y_lims=y_range,
                     z_lims=z_range,
                     axis=([], false),
                     camera=(0,0))

    mesh3d!(vec(wake_panel_markers[1,:,:]),
            vec(wake_panel_markers[2,:,:]),
            vec(wake_panel_markers[3,:,:]);
            connections=(a_wake,
                         b_wake,
                         c_wake),
            legend=:none,
            color=:blue,
            fillalpha=0.4)
    
    mesh3d!(vec(blade_panel_markers[1,:,:]),
            vec(blade_panel_markers[2,:,:]),
            vec(blade_panel_markers[3,:,:]);
            connections=(a_blade,
                         b_blade,
                         c_blade),
            legend=:none,
            color=:red)


    display(plt)
end
