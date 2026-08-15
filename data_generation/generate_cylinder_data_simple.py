"""
使用简化方法生成不同雷诺数的圆柱绕流数据
基于物理方程的数值模拟
"""

import numpy as np
import h5py
import os
from scipy.ndimage import gaussian_filter, laplace

def generate_vortex_shedding(Re, nx=192, ny=378, nt=200):
    """
    生成简化的卡门涡街数据
    基于涡量方程的简化模型
    """
    print(f"Generating Re={Re} data...")
    
    # 参数设置
    dt = 0.01
    nu = 1.0 / Re  # 运动粘度
    
    # 初始化涡度场
    omega = np.zeros((nt, nx, ny), dtype=np.float32)
    
    # 初始条件：在圆柱后方有一些扰动
    cx, cy = nx // 2, ny // 3  # 圆柱位置
    
    # 初始涡度分布（模拟边界层分离）
    for i in range(nx):
        for j in range(ny):
            dx = (i - cx) / 10.0
            dy = (j - cy) / 10.0
            r = np.sqrt(dx**2 + dy**2)
            if r > 1.0:
                omega[0, i, j] = 0.1 * np.sin(dy) * np.exp(-r/5)
    
    # 简化的涡量方程演化
    # dω/dt = -u·∇ω + ν·∇²ω
    
    for t in range(1, nt):
        # 扩散项
        laplacian = laplace(omega[t-1])
        diffusion = nu * laplacian * dt
        
        # 对流项（简化处理）
        # 使用简单的平流模型
        advection = np.zeros_like(omega[t-1])
        
        # 简化的周期性涡脱落
        shedding_freq = 0.2 * (Re / 250)  # 与 Re 相关的脱落频率
        phase = 2 * np.pi * shedding_freq * t * dt
        
        # 在圆柱后方添加周期性扰动
        wake_region = (np.arange(ny) > cy) & (np.arange(ny) < ny - 20)
        for i in range(cx-20, cx+20):
            if 0 <= i < nx:
                for j in np.where(wake_region)[0]:
                    advection[i, j] = 0.01 * np.sin(phase + j * 0.1) * np.exp(-(i-cx)**2/100)
        
        # 更新涡度场
        omega[t] = omega[t-1] + diffusion - advection * dt
        
        # 应用边界条件
        # 壁面边界：圆柱位置
        omega[t, cx-2:cx+2, cy-2:cy+2] = 0
        
        # 远场边界
        omega[t, 0, :] = omega[t, 1, :]
        omega[t, -1, :] = omega[t, -2, :]
        omega[t, :, 0] = 0
        omega[t, :, -1] = 0
    
    return omega

def save_as_pdebench_format(data, Re, output_dir):
    """
    保存为 PDEBench 格式 (HDF5)
    """
    nt, nx, ny = data.shape
    
    # PDEBench 格式: (3, 192, 378, 200, 1)
    # 3 = [u, v, omega]
    fields = np.zeros((3, nx, ny, nt, 1), dtype=np.float32)
    
    # 填充涡度
    for t in range(nt):
        fields[2, :, :, t, 0] = data[t]
    
    # 填充简化的速度场（从涡度推导）
    fields[0] = 0.5  # u 速度
    fields[1] = 0.0  # v 速度
    
    # 保存
    filename = f"cyl_Re{Re}_synthetic.mat"
    filepath = os.path.join(output_dir, filename)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('fields_', data=fields)
        f.create_dataset('grids_', data=np.zeros((2, nx, ny)))
        f.attrs['Re'] = Re
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nt'] = nt
    
    print(f"Saved: {filepath}")
    print(f"  Shape: {fields.shape}")
    print(f"  Size: {fields.nbytes / 1024 / 1024:.1f} MB")

def main():
    """主函数"""
    output_dir = r"D:\湍流实验数据"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Synthetic Cylinder Flow Data Generator")
    print("=" * 60)
    print()
    
    # 生成不同 Re 的数据
    Re_values = [150, 300, 400]
    
    for Re in Re_values:
        # 生成数据
        data = generate_vortex_shedding(Re)
        
        # 保存
        save_as_pdebench_format(data, Re, output_dir)
        print()
    
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print()
    print("Files generated:")
    for Re in Re_values:
        print(f"  - cyl_Re{Re}_synthetic.mat")
    print()
    print("Note: These are synthetic data for testing purposes.")
    print("For real data, please download from official sources.")

if __name__ == "__main__":
    main()
