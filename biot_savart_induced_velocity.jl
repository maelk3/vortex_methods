using LinearAlgebra
using PlotlyJS

# velocity fied induced by the segment AB using the Biot-Savart law

nx = 30
ny = 30
nz = 30

xlims = [-1.0, 2.0]
ylims = [-4.0, 4.0]
zlims = [-4.0, 4.0]

x = permutedims(repeat(range(xlims[1], xlims[2], length=nx), 1, ny, nz), (1, 2, 3))
y = permutedims(repeat(range(ylims[1], ylims[2], length=ny), 1, nx, nz), (2, 1, 3))
z = permutedims(repeat(range(zlims[1], zlims[2], length=nz), 1, nx, ny), (2, 3, 1))

P = permutedims(cat(x, y, z, dims=4), (4, 1, 2, 3))
A = [0.0, 0.0, 0.0]
B = [1.0, 0.0, 0.0]
r₀ = (B-A)/norm(B-A)
r₁ = broadcast(-, A, P)
r₂ = broadcast(-, B, P)

Γ = 1.0

t1 = sum(broadcast(*, r₁, r₀), dims=1) ./ mapslices(norm, r₁, dims=[1])
t2 = sum(broadcast(*, r₂, r₀), dims=1) ./ mapslices(norm, r₂, dims=[1])
t3 = t1 - t2

direction = zeros(3, nx, ny, nz)
for i∈1:nx, j∈1:ny, k∈1:nz
    direction[:,i,j,k] = cross(r₁[:,i,j,k], r₀)
end

norm_direction = dropdims(mapslices(norm, direction, dims=(1)) .^ 2, dims=(1))

scalar = (Γ/4π) ./ norm_direction .* dropdims(t3, dims=1)

v_induced = direction
for i∈1:nx, j∈1:ny, k∈1:nz
    v_induced[:,i,j,k] = v_induced[:,i,j,k] .* scalar[i,j,k]
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
         

