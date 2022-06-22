using LinearAlgebra
using PlotlyJS

# velocity fied induced by the segment AB with core size rc using Biot-Savart law and rigid rotation

nx = 30
ny = 30
nz = 30

xlims = [-1.0, 2.0]
ylims = [-1.0, 1.0]
zlims = [-1.0, 1.0]

x = permutedims(repeat(range(xlims[1], xlims[2], length=nx), 1, ny, nz), (1, 2, 3))
y = permutedims(repeat(range(ylims[1], ylims[2], length=ny), 1, nx, nz), (2, 1, 3))
z = permutedims(repeat(range(zlims[1], zlims[2], length=nz), 1, nx, ny), (2, 3, 1))

P = permutedims(cat(x, y, z, dims=4), (4, 1, 2, 3))
A = [0.0, 0.0, 0.0]
B = [1.0, 0.0, 0.0]
r₀ = (B-A)/norm(B-A)

Γ = 1.0
rc = 0.3

function induced_vel(P, A, B, r₀, Γ, rc)
    r₁ = A-P
    r₂ = B-P
    r = norm(r₁×r₀)
    if r<rc && dot(r₁,r₀)*dot(r₂,r₀) <= 0
        return (Γ/(2π*rc))*(r/rc)*(r₀×r₁)/norm(r₀×r₁)
    else
        return (Γ/4π)*(dot(r₀, r₁)/norm(r₁) - dot(r₀, r₂)/norm(r₂))*(r₁×r₀)/norm(r₁×r₀)^2
    end
end

v_induced = zeros(3, nx, ny, nz)
for i∈1:nx, j∈1:ny, k∈1:nz
    v_induced[:,i,j,k] = induced_vel(P[:,i,j,k], A, B, r₀, Γ, rc)
end

# 3D quiver plot
trace = cone(x=vec(x),
         y=vec(y),
         z=vec(z),
         u=vec(v_induced[1,:,:,:]),
         v=vec(v_induced[2,:,:,:]),
         w=vec(v_induced[3,:,:,:]),
         sizemode="scaled",
         sizeref=2)
plot(trace,
     Layout(width=600, height=600,
            scene_camera_eye=attr(x=1.55, y=1.55, z=0.6)))
         

