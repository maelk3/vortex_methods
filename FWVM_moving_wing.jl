using LinearAlgebra
using PlotlyJS
using PyPlot

# PHYSICAL CONSTANTS
α  = 5*(π/180)                 # angle of attack
U₀ = 0.3                       # speed
U  = U₀*[cos(α), 0.0, sin(α)]  # freestream velocity

ρ = 1.0                        # air density    

# BLADE PROFILE DEFINITION
c = 0.1                        # chord length
s = 1.0                        # span length
η(x) = 0.02*sin((π/c)*x)       # chord profile

# DISCRETISATION PARAMETERS
rc = 0.00                      # core radius
N = 5                          # chordwise number of panels
M = 20                         # spanwise number of panels

nt = 80                        # number of time steps
Δt = 0.085                     # time intervals

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

spanwise_distribution(x) = (cos(π*(1-x))+1)/2-1/2 # cosine distribution

# lagrangian markers coordinates
x_blade_markers = repeat(range(0, c, length=N+1), 1, M+1)
y_blade_markers = repeat(spanwise_distribution.(range(0, s, length=M+1)), 1, N+1)'
z_blade_markers = repeat(η.(range(0,c,length=N+1)), 1, M+1)

# blade panel markers
blade_panel_markers = permutedims(cat(x_blade_markers, y_blade_markers, z_blade_markers, dims=3), (3, 1, 2))
Δy = blade_panel_markers[:,:,2:end,:]-blade_panel_markers[:,:,1:end-1,:]

# collocation points
x_collocation = 0.5*(x_blade_markers[1:end-1,1:end-1] + x_blade_markers[2:end,1:end-1])
y_collocation = 0.5*(y_blade_markers[1:end-1,1:end-1] + y_blade_markers[1:end-1,2:end])
z_collocation = 0.5*(z_blade_markers[1:end-1,1:end-1] + z_blade_markers[2:end,1:end-1])
collocation_points = permutedims(cat(x_collocation, y_collocation, z_collocation, dims=3), (3, 1, 2))

# collocation normals
collocation_normals = zeros(3, N, M)
for i∈1:N
    Δx = c/N
    normal = [-(η(i*Δx)-η((i-1)*Δx)), 0, Δx]
    collocation_normals[:,i,:] = repeat(normal/norm(normal), 1, M)
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

x_wake_markers = c*ones(nt+1, M+1) 
y_wake_markers = repeat(spanwise_distribution.(range(0, s, length=M+1)), 1, nt+1)'
z_wake_markers = zeros(nt+1, M+1)
wake_panel_markers = permutedims(cat(x_wake_markers, y_wake_markers, z_wake_markers, dims=3), (3, 1, 2))

Γ = zeros(N, M)
RHS = zeros(N, M)
wake_Γ = zeros(nt, M)
wake_rings = zeros(nt, M, 4, 2, 3)

for t∈1:nt
    println(t)
    
    RHS .= dropdims(mapslices((x -> dot(x, -U)), collocation_normals, dims=(1)), dims=(1))
    for τ∈1:t-1, l∈1:M
        for i∈1:N, j∈1:M
            RHS[i,j] -= dot(collocation_normals[:,i,j], induced_vel_ring(collocation_points[:,i,j], wake_rings[τ,l,:,:,:], wake_Γ[τ,l], rc))
        end
    end

    Γ .= reshape(reshape(influence_matrix, N*M, N*M) \ reshape(RHS, N*M), N, M)
    wake_Γ[t,:] = Γ[end,:]

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

    # move the blade
    for i∈1:N+1, j∈1:M+1
        blade_panel_markers[3,i,j] = 0.02*sin(t*Δt)+z_blade_markers[i,j]
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
    collocation_points[1,:,:] = 0.5*(blade_panel_markers[1,1:end-1,1:end-1] + blade_panel_markers[1,2:end,1:end-1])
    collocation_points[2,:,:] = 0.5*(blade_panel_markers[2,1:end-1,1:end-1] + blade_panel_markers[2,1:end-1,2:end])
    collocation_points[3,:,:] = 0.5*(blade_panel_markers[3,1:end-1,1:end-1] + blade_panel_markers[3,2:end,1:end-1])

    # prescribe next trailing edge
    wake_panel_markers[:,t+1,:] = blade_panel_markers[:,end,:]

end

# total velocity field computation
nx, ny, nz = 50, 1, 50

xlims = [-0.1, 0.3]
ylims = [0.5,  0.5]
zlims = [-0.1, 0.1]

x = permutedims(repeat(range(xlims[1], xlims[2], length=nx), 1, ny, nz), (1, 2, 3))
y = permutedims(repeat(range(ylims[1], ylims[2], length=ny), 1, nx, nz), (2, 1, 3))
z = permutedims(repeat(range(zlims[1], zlims[2], length=nz), 1, nx, ny), (2, 3, 1))

P = permutedims(cat(x, y, z, dims=4), (4, 1, 2, 3))
V = repeat(U, 1, nx, ny, nz)
for x∈1:nx, y∈1:ny, z∈1:nz
    V[:,x,y,z] += induced_vel(P[:,x,y,z], reshape(blade_rings, N*M, 4, 2, 3), reshape(Γ, N*M), rc)
    V[:,x,y,z] += induced_vel(P[:,x,y,z], reshape(wake_rings, nt*M, 4, 2, 3), reshape(wake_Γ, nt*M), rc)
end

# PlotlyJS
x_max = maximum(wake_panel_markers[1,:,:])

x_range = [0.0, x_max]
y_range = [-0.6, 0.6]
z_range = [-0.4, 0.4]

layout = PlotlyJS.Layout(scene=attr(xaxis_range=x_range,
                                    yaxis_range=y_range,
                                    zaxis_range=z_range),
                         scene_camera=attr(
                             up=attr(x=0, y=0, z=1),
                             center=attr(x=x_max/4, y=0, z=0),
                             eye=attr(x=1.25, y=1.25, z=1.25)
                         ),
                         scene_aspectratio=attr(x=x_range[2]-x_range[1], y=y_range[2]-y_range[1], z=z_range[2]-z_range[1]))

blade_surface = PlotlyJS.mesh3d(x=vec(blade_panel_markers[1,:,:]),
                                y=vec(blade_panel_markers[2,:,:]),
                                z=vec(blade_panel_markers[3,:,:]),
                                opacity=0.8,
                                color="rgb(0,0,255)")

normal_field = PlotlyJS.cone(x=vec(collocation_points[1,:,:]),
                             y=vec(collocation_points[2,:,:]),
                             z=vec(collocation_points[3,:,:]),
                             u=vec(collocation_normals[1,:,:,:]),
                             v=vec(collocation_normals[2,:,:,:]),
                             w=vec(collocation_normals[3,:,:,:]),
                             sizemode="scaled",
                             sizeref=3,
                             showscale=false)

lines = GenericTrace{Dict{Symbol,Any}}[]
for i∈1:nt+1
    line = PlotlyJS.scatter(x=wake_panel_markers[1,i,:],
                            y=wake_panel_markers[2,i,:],
                            z=wake_panel_markers[3,i,:],
                            mode="lines",
                            type="scatter3d",
                            line=attr(color="black", width=2),
                            showlegend=false)
    push!(lines, line)
end

for j∈1:M+1
    line = PlotlyJS.scatter(x=wake_panel_markers[1,:,j],
                            y=wake_panel_markers[2,:,j],
                            z=wake_panel_markers[3,:,j],
                            mode="lines",
                            type="scatter3d",
                            line=attr(color="black", width=2),
                            showlegend=false)
    push!(lines, line)
end
                          
PlotlyJS.plot(vcat(lines, [blade_surface,
                           normal_field]), layout)
