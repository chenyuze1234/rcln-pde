"""
Create 3D isosurface visualization from synthetic cylinder wake data
Generates VTK files for ParaView and matplotlib 3D plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import os

print("[INIT] Loading data...")

# Load step 500 (fully developed flow)
data = np.load("paper/synthetic_3d_cylinder/gt_data_step0500.npz")
u = data['u']
v = data['v']
w = data['w']
Q = data['Q']
x = data['x']
y = data['y']
z = data['z']

print(f"[DATA] Shape: {u.shape}, Q range: [{Q.min():.2f}, {Q.max():.2f}]")

# Create output directory
os.makedirs("paper/synthetic_3d_cylinder/3d_viz", exist_ok=True)

# ==============================================================================
# Q-criterion Isosurface
# ==============================================================================
print("\n[3D PLOT] Generating Q-criterion isosurface...")

# Use a lower threshold for better visualization
Q_threshold = 10.0  # Adjust based on Q distribution
print(f"[THRESHOLD] Q > {Q_threshold}")

# Find isosurface
verts, faces, normals, values = measure.marching_cubes(Q, level=Q_threshold)

# Scale vertices to physical coordinates
verts[:, 0] = verts[:, 0] * (x[-1] - x[0]) / len(x) + x[0]
verts[:, 1] = verts[:, 1] * (y[-1] - y[0]) / len(y) + y[0]
verts[:, 2] = verts[:, 2] * (z[-1] - z[0]) / len(z) + z[0]

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create mesh
mesh = Poly3DCollection(verts[faces], alpha=0.7, facecolor='cyan', edgecolor='blue', linewidth=0.1)
ax.add_collection3d(mesh)

# Set limits
ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
ax.set_zlim(z[0], z[-1])

ax.set_xlabel('X (streamwise)')
ax.set_ylabel('Y (cross-stream)')
ax.set_zlabel('Z (spanwise)')
ax.set_title(f'Q-criterion Isosurface (Q > {Q_threshold})\nSynthetic 3D Cylinder Wake Re=5000, Step 500')

# Add cylinder
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
theta = np.linspace(0, 2*np.pi, 50)
cyl_x = 2.0 + 0.5 * np.cos(theta)
cyl_y = 4.0 + 0.5 * np.sin(theta)
cyl_z_bottom = np.zeros_like(theta)
cyl_z_top = np.ones_like(theta) * z[-1]

# Plot cylinder
ax.plot(cyl_x, cyl_y, cyl_z_bottom, 'k-', linewidth=2)
ax.plot(cyl_x, cyl_y, cyl_z_top, 'k-', linewidth=2)
for i in range(0, len(theta), 5):
    ax.plot([cyl_x[i], cyl_x[i]], [cyl_y[i], cyl_y[i]], [cyl_z_bottom[i], cyl_z_top[i]], 'k-', alpha=0.3)

plt.tight_layout()
plt.savefig("paper/synthetic_3d_cylinder/3d_viz/q_isosurface_3d.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: q_isosurface_3d.png ({len(faces)} faces, {len(verts)} vertices)")

# ==============================================================================
# Vorticity Magnitude Isosurface
# ==============================================================================
print("\n[3D PLOT] Generating vorticity magnitude isosurface...")

# Compute vorticity magnitude
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

dwdy = np.gradient(w, dy, axis=1)
dvdz = np.gradient(v, dz, axis=2)
dudz = np.gradient(u, dz, axis=2)
dwdx = np.gradient(w, dx, axis=0)
dvdx = np.gradient(v, dx, axis=0)
dudy = np.gradient(u, dy, axis=1)

wx = dwdy - dvdz
wy = dudz - dwdx
wz = dvdx - dudy

vort_mag = np.sqrt(wx**2 + wy**2 + wz**2)
print(f"[VORTICITY] Range: [{vort_mag.min():.2f}, {vort_mag.max():.2f}]")

vort_threshold = np.percentile(vort_mag, 95)  # Top 5%
print(f"[THRESHOLD] |ω| > {vort_threshold:.2f}")

verts_vort, faces_vort, _, _ = measure.marching_cubes(vort_mag, level=vort_threshold)

verts_vort[:, 0] = verts_vort[:, 0] * (x[-1] - x[0]) / len(x) + x[0]
verts_vort[:, 1] = verts_vort[:, 1] * (y[-1] - y[0]) / len(y) + y[0]
verts_vort[:, 2] = verts_vort[:, 2] * (z[-1] - z[0]) / len(z) + z[0]

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

mesh_vort = Poly3DCollection(verts_vort[faces_vort], alpha=0.6, facecolor='magenta', edgecolor='purple', linewidth=0.1)
ax.add_collection3d(mesh_vort)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
ax.set_zlim(z[0], z[-1])

ax.set_xlabel('X (streamwise)')
ax.set_ylabel('Y (cross-stream)')
ax.set_zlabel('Z (spanwise)')
ax.set_title(f'Vorticity Magnitude Isosurface (|ω| > {vort_threshold:.1f})\nSynthetic 3D Cylinder Wake Re=5000, Step 500')

plt.tight_layout()
plt.savefig("paper/synthetic_3d_cylinder/3d_viz/vorticity_isosurface_3d.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: vorticity_isosurface_3d.png ({len(faces_vort)} faces)")

# ==============================================================================
# Velocity Magnitude Isosurface
# ==============================================================================
print("\n[3D PLOT] Generating velocity magnitude isosurface...")

vel_mag = np.sqrt(u**2 + v**2 + w**2)
vel_threshold = np.percentile(vel_mag, 90)
print(f"[VELOCITY] Range: [{vel_mag.min():.2f}, {vel_mag.max():.2f}], threshold: {vel_threshold:.2f}")

verts_vel, faces_vel, _, _ = measure.marching_cubes(vel_mag, level=vel_threshold)

verts_vel[:, 0] = verts_vel[:, 0] * (x[-1] - x[0]) / len(x) + x[0]
verts_vel[:, 1] = verts_vel[:, 1] * (y[-1] - y[0]) / len(y) + y[0]
verts_vel[:, 2] = verts_vel[:, 2] * (z[-1] - z[0]) / len(z) + z[0]

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

mesh_vel = Poly3DCollection(verts_vel[faces_vel], alpha=0.5, facecolor='yellow', edgecolor='orange', linewidth=0.1)
ax.add_collection3d(mesh_vel)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
ax.set_zlim(z[0], z[-1])

ax.set_xlabel('X (streamwise)')
ax.set_ylabel('Y (cross-stream)')
ax.set_zlabel('Z (spanwise)')
ax.set_title(f'Velocity Magnitude Isosurface (|U| > {vel_threshold:.2f})\nSynthetic 3D Cylinder Wake Re=5000, Step 500')

plt.tight_layout()
plt.savefig("paper/synthetic_3d_cylinder/3d_viz/velocity_isosurface_3d.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: velocity_isosurface_3d.png ({len(faces_vel)} faces)")

# ==============================================================================
# Multi-view Comparison
# ==============================================================================
print("\n[3D PLOT] Generating multi-view comparison...")

fig = plt.figure(figsize=(18, 12))

# Q-criterion - side view
ax1 = fig.add_subplot(231, projection='3d')
ax1.add_collection3d(Poly3DCollection(verts[faces], alpha=0.5, facecolor='cyan', edgecolor='blue', linewidth=0.1))
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(y[0], y[-1])
ax1.set_zlim(z[0], z[-1])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'Q-criterion (Q>{Q_threshold})')
ax1.view_init(elev=0, azim=0)

# Q-criterion - top view
ax2 = fig.add_subplot(232, projection='3d')
ax2.add_collection3d(Poly3DCollection(verts[faces], alpha=0.5, facecolor='cyan', edgecolor='blue', linewidth=0.1))
ax2.set_xlim(x[0], x[-1])
ax2.set_ylim(y[0], y[-1])
ax2.set_zlim(z[0], z[-1])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title(f'Q-criterion (top view)')
ax2.view_init(elev=90, azim=0)

# Q-criterion - front view
ax3 = fig.add_subplot(233, projection='3d')
ax3.add_collection3d(Poly3DCollection(verts[faces], alpha=0.5, facecolor='cyan', edgecolor='blue', linewidth=0.1))
ax3.set_xlim(x[0], x[-1])
ax3.set_ylim(y[0], y[-1])
ax3.set_zlim(z[0], z[-1])
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title(f'Q-criterion (front view)')
ax3.view_init(elev=0, azim=90)

# Vorticity - combined view
ax4 = fig.add_subplot(234, projection='3d')
ax4.add_collection3d(Poly3DCollection(verts_vort[faces_vort], alpha=0.4, facecolor='magenta', edgecolor='purple', linewidth=0.1))
ax4.set_xlim(x[0], x[-1])
ax4.set_ylim(y[0], y[-1])
ax4.set_zlim(z[0], z[-1])
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title(f'Vorticity |ω|>{vort_threshold:.1f}')

# Velocity - combined view
ax5 = fig.add_subplot(235, projection='3d')
ax5.add_collection3d(Poly3DCollection(verts_vel[faces_vel], alpha=0.4, facecolor='yellow', edgecolor='orange', linewidth=0.1))
ax5.set_xlim(x[0], x[-1])
ax5.set_ylim(y[0], y[-1])
ax5.set_zlim(z[0], z[-1])
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.set_title(f'Velocity |U|>{vel_threshold:.2f}')

# Combined - all three
ax6 = fig.add_subplot(236, projection='3d')
ax6.add_collection3d(Poly3DCollection(verts[faces], alpha=0.3, facecolor='cyan', edgecolor='blue', linewidth=0.1))
ax6.add_collection3d(Poly3DCollection(verts_vort[faces_vort], alpha=0.3, facecolor='magenta', edgecolor='purple', linewidth=0.1))
ax6.set_xlim(x[0], x[-1])
ax6.set_ylim(y[0], y[-1])
ax6.set_zlim(z[0], z[-1])
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.set_title(f'Combined View')

plt.suptitle(f'Synthetic 3D Cylinder Wake Re=5000 - Step 500\nMultiple Isosurface Views', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("paper/synthetic_3d_cylinder/3d_viz/multi_view_isosurfaces.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: multi_view_isosurfaces.png")

# ==============================================================================
# Write VTK file for ParaView
# ==============================================================================
print("\n[VTK] Writing VTK file for ParaView...")

# Simple VTK writer
def write_vtk_structured_grid(filename, x, y, z, u, v, w, Q, vort_mag):
    nx, ny, nz = len(x), len(y), len(z)
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Synthetic 3D Cylinder Wake\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"POINTS {nx*ny*nz} float\n")
        
        # Write points
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{x[i]:.6f} {y[j]:.6f} {z[k]:.6f}\n")
        
        # Write point data
        f.write(f"POINT_DATA {nx*ny*nz}\n")
        
        # Velocity vector
        f.write("VECTORS velocity float\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{u[i,j,k]:.6f} {v[i,j,k]:.6f} {w[i,j,k]:.6f}\n")
        
        # Q-criterion
        f.write("SCALARS Q_criterion float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{Q[i,j,k]:.6f}\n")
        
        # Vorticity magnitude
        f.write("SCALARS vorticity_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{vort_mag[i,j,k]:.6f}\n")
        
        # Velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        f.write("SCALARS velocity_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{vel_mag[i,j,k]:.6f}\n")

write_vtk_structured_grid(
    "paper/synthetic_3d_cylinder/3d_viz/cylinder_wake_step500.vtk",
    x, y, z, u, v, w, Q, vort_mag
)

print(f"  Saved: cylinder_wake_step500.vtk")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*70)
print("3D VISUALIZATION COMPLETE")
print("="*70)
print(f"Output directory: paper/synthetic_3d_cylinder/3d_viz/")
print(f"\nGenerated files:")
print(f"  - q_isosurface_3d.png: Q-criterion isosurface")
print(f"  - vorticity_isosurface_3d.png: Vorticity magnitude isosurface")
print(f"  - velocity_isosurface_3d.png: Velocity magnitude isosurface")
print(f"  - multi_view_isosurfaces.png: Multiple views comparison")
print(f"  - cylinder_wake_step500.vtk: Full 3D data for ParaView")
print(f"\nTo visualize in ParaView:")
print(f"  1. Open cylinder_wake_step500.vtk")
print(f"  2. Apply 'Contour' filter for Q-criterion (threshold: {Q_threshold})")
print(f"  3. Apply 'Glyph' filter for velocity vectors")
print(f"  4. Use 'Clip' to show cross-sections")
print("="*70)
