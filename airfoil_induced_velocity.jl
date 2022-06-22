using LinearAlgebra
using PlotlyJS
using PyPlot

# FREESTREAM VELOCITY
α  = 45*(π/180)               # angle of attack
U₀ = 1.0                      # speed
U  = U₀*[cos(α), 0.0, sin(α)] # freestream velocity

# BLADE PROFILE DEFINITION
c = 0.2                       # chord length
s = 1.0                       # span length
η(x) = 0.02*sin((π/c)*x)      # chord profile

# BLADE DISCRETISATION PARAMETERS
rc = 0.01                     # core radius
N = 15                        # chordwise number of panels
M = 7                         # spanwise number of panels

# RANKINE VORTEX SEGMENT AND RING DEFINITION
function induced_vel_segment(P, A, B, r₀, Γ, rc)
    r₁ = A-P
    r₂ = B-P
    r = norm(r₁×r₀)
    if r<rc && dot(r₁,r₀)*dot(r₂,r₀) <= 0
        return (Γ/(2π*rc))*(r/rc)*(r₀×r₁)/norm(r₀×r₁)
    else
        return (Γ/4π)*(dot(r₀, r₁)/norm(r₁) - dot(r₀, r₂)/norm(r₂))*(r₁×r₀)/norm(r₁×r₀)^2
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

# lagrangian markers coordinates
x_markers = repeat(range(0, c, length=N+1), 1, M+1)
y_markers = repeat(range(0, 1, length=M+1), 1, N+1)'
z_markers = repeat(η.(range(0,c,length=N+1)), 1, M+1)

# blade panel markers
blade_panel_markers = permutedims(cat(x_markers, y_markers, z_markers, dims=3), (3, 1, 2))

# collocation points
x_collocation = 0.5*(x_markers[1:end-1,1:end-1] + x_markers[2:end,1:end-1])
y_collocation = 0.5*(y_markers[1:end-1,1:end-1] + y_markers[1:end-1,2:end])
z_collocation = 0.5*(z_markers[1:end-1,1:end-1] + z_markers[2:end,1:end-1])
collocation_points = permutedims(cat(x_collocation, y_collocation, z_collocation, dims=3), (3, 1, 2))

# collocation normals
collocation_normals = zeros(3, N, M)
for i∈1:N
    Δx = c/N
    normal = [-(η(i*Δx)-η((i-1)*Δx)), 0, Δx]
    collocation_normals[:,i,:] = repeat(normal/norm(normal), 1, M)
end

rings = zeros(N, M, 4, 2, 3)
for i∈1:N, j∈1:M    
    rings[i,j,1,1,:] = blade_panel_markers[:,i,j]
    rings[i,j,1,2,:] = blade_panel_markers[:,i+1,j]
    rings[i,j,2,1,:] = blade_panel_markers[:,i+1,j]
    rings[i,j,2,2,:] = blade_panel_markers[:,i+1,j+1]
    rings[i,j,3,1,:] = blade_panel_markers[:,i+1,j+1]
    rings[i,j,3,2,:] = blade_panel_markers[:,i,j+1]
    rings[i,j,4,1,:] = blade_panel_markers[:,i,j+1]
    rings[i,j,4,2,:] = blade_panel_markers[:,i,j]                    
end

# influence matrix
influence_matrix = zeros(N, M, N, M)
for i∈1:N, j∈1:M, k∈1:N, l∈1:M
    influence_matrix[i,j,k,l] = dot(induced_vel_ring(collocation_points[:,i,j], rings[k,l,:,:,:], 1, rc), collocation_normals[:,i,j])
end

RHS = reshape(mapslices((x -> dot(x, -U)), collocation_normals, dims=(1)), N*M)

Γ = reshape(reshape(influence_matrix, N*M, N*M) \ RHS, N, M)

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
    for i∈1:N, j∈1:M
        V[:,x,y,z] += induced_vel_ring(P[:,x,y,z], rings[i,j,:,:,:], Γ[i,j], rc)
    end
end

# PyPlot
x_flat = dropdims(x, dims=(2))
z_flat = dropdims(z, dims=(2))

PyPlot.figure()
PyPlot.plot(collocation_points[2,1,:], Γ[end,:])
PyPlot.title("lift distribution in the spanwise direction")
PyPlot.xlabel("x")
PyPlot.ylabel("L")
PyPlot.show()

PyPlot.figure()
PyPlot.streamplot(x_flat', z_flat', V[1,:,1,:]', V[3,:,1,:]', density=3, color="black", linewidth=0.5, arrowsize=0.5)
PyPlot.plot(range(0,c,length=N+1), z_markers[:,1])
PyPlot.title("velocity field around a wing profile")
PyPlot.savefig("figures/wind_profile.pdf", bbox_inches="tight")
PyPlot.show()

# PlotlyJS
layout = PlotlyJS.Layout(scene=attr(xaxis_range=[-1.0, 1.0],
                                    yaxis_range=[-0.5, 1.5],
                                    zaxis_range=[-1.0, 1.0]))

surface = PlotlyJS.mesh3d(x=vec(x_markers),
                          y=vec(y_markers),
                          z=vec(z_markers))

normal_field = PlotlyJS.cone(x=vec(x_collocation),
                             y=vec(y_collocation),
                             z=vec(z_collocation),
                             u=vec(collocation_normals[1,:,:,:]),
                             v=vec(collocation_normals[2,:,:,:]),
                             w=vec(collocation_normals[3,:,:,:]),
                             sizemode="scaled",
                             sizeref=3)

velocity_field = PlotlyJS.cone(x=vec(x),
                               y=vec(y),
                               z=vec(z),
                               u=vec(V[1,:,:,:]),
                               v=vec(V[2,:,:,:]),
                               w=vec(V[3,:,:,:]),
                               sizemode="scaled",
                               sizeref=3)

PlotlyJS.plot([surface, velocity_field, normal_field], layout)
