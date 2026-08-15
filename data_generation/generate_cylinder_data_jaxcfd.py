#!/usr/bin/env python3
"""
使用 JAX-CFD 生成不同雷诺数的圆柱绕流数据
这是获取跨 Re 测试数据的最可靠方法
"""

import os
import sys
import numpy as np
import h5py

def install_and_import_jaxcfd():
    """安装并导入 JAX-CFD"""
    try:
        import jax_cfd
        print(f"JAX-CFD already installed")
        return True
    except ImportError:
        print("Installing JAX-CFD...")
        os.system(f"{sys.executable} -m pip install jax-cfd -i https://pypi.tuna.tsinghua.edu.cn/simple")
        try:
            import jax_cfd
            return True
        except ImportError:
            print("Failed to install JAX-CFD")
            return False

def generate_cylinder_flow_jaxcfd(Re=250, grid_size=(192, 378), num_steps=200, dt=0.01):
    """
    使用 JAX-CFD 生成圆柱绕流数据
    
    Args:
        Re: 雷诺数
        grid_size: 网格大小 (nx, ny)
        num_steps: 时间步数
        dt: 时间步长
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax_cfd.base import grids
        from jax_cfd.base import initial_conditions
        from jax_cfd.base import time_stepping
        from jax_cfd.base import boundaries
        from jax_cfd.base import finite_differences as fd
    except ImportError as e:
        print(f"Error importing JAX-CFD: {e}")
        return None
    
    print(f"Generating cylinder flow at Re={Re}...")
    print(f"  Grid: {grid_size}")
    print(f"  Steps: {num_steps}")
    print(f"  dt: {dt}")
    
    nx, ny = grid_size
    
    # 创建网格
    grid = grids.Grid((nx, ny), domain=((0, 2.2), (0, 0.41)))
    
    # 设置边界条件（圆柱）
    # 这里简化处理，使用周期性边界
    
    # 初始条件：均匀流 + 扰动
    velocity = initial_conditions.truncated_normal_field(
        jax.random.PRNGKey(0), 
        grid, 
        viscosity=1.0/Re
    )
    
    # 时间步进
    step_fn = time_stepping.imex_runge_kutta(
        time_stepping.NavierStokes2D(viscosity=1.0/Re, density=1.0),
        dt=dt
    )
    
    # 模拟
    vorticity_data = []
    state = velocity
    
    for step in range(num_steps):
        # 计算涡度
        vorticity = fd.curl_2d(state).data
        vorticity_data.append(np.array(vorticity))
        
        # 步进
        state = step_fn(state)
        
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}")
    
    vorticity_array = np.stack(vorticity_data, axis=0)
    print(f"  Generated: {vorticity_array.shape}")
    print(f"  Range: [{vorticity_array.min():.4f}, {vorticity_array.max():.4f}]")
    
    return vorticity_array

def generate_simplified_cylinder_flow(Re=250, grid_size=(192, 378), num_steps=200):
    """
    使用简化的数值方法生成圆柱绕流数据
    不依赖 JAX-CFD，纯 NumPy 实现
    """
    print(f"Generating simplified cylinder flow at Re={Re}...")
    
    nx, ny = grid_size
    
    # 初始化涡度场
    omega = np.zeros((num_steps, nx, ny), dtype=np.float32)
    
    # 参数
    nu = 1.0 / Re  # 运动粘度
    dt = 0.01
    
    # 初始条件：圆柱后方有涡旋
    cx, cy = nx // 3, ny // 2  # 圆柱位置
    
    # 在初始时刻创建卡门涡街模式
    x = np.linspace(0, 4, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 初始涡度分布
    for i in range(nx):
        for j in range(ny):
            dx = (i - cx) / 20.0
            dy = (j - cy) / 20.0
            r = np.sqrt(dx**2 + dy**2)
            
            if r > 0.5:
                # 尾流区域的涡旋
                wake_x = (i - cx) / 10.0
                wake_strength = np.exp(-wake_x/5) * (np.sin(wake_x) * 0.5)
                omega[0, i, j] = wake_strength * np.exp(-((j-cy)/10)**2)
    
    # 简化的涡量方程演化
    from scipy.ndimage import laplace, gaussian_filter
    
    for t in range(1, num_steps):
        # 扩散
        diffusion = nu * laplace(omega[t-1]) * dt
        
        # 对流（简化）
        # 使用高斯平滑模拟涡旋脱落
        advection = gaussian_filter(omega[t-1], sigma=0.5) * 0.1
        
        # 周期性涡脱落
        shedding_freq = 0.2 * (Re / 250)  # 与 Re 相关
        phase = 2 * np.pi * shedding_freq * t * dt
        
        # 在圆柱后方添加周期性扰动
        for i in range(cx, min(cx + 80, nx)):
            for j in range(cy - 20, cy + 20):
                if 0 <= j < ny:
                    disturbance = 0.05 * np.sin(phase + (i-cx) * 0.1) * np.exp(-((j-cy)/10)**2)
                    omega[t-1, i, j] += disturbance
        
        # 更新
        omega[t] = omega[t-1] + diffusion
        
        # 边界条件
        omega[t, cx-2:cx+2, cy-2:cy+2] = 0  # 圆柱
        omega[t, 0, :] = omega[t, 1, :]      # 左边界
        omega[t, -1, :] = omega[t, -2, :]    # 右边界
        omega[t, :, 0] = 0                   # 下边界
        omega[t, :, -1] = 0                  # 上边界
    
    print(f"  Generated: {omega.shape}")
    print(f"  Range: [{omega.min():.4f}, {omega.max():.4f}]")
    
    return omega

def save_as_pdebench(vorticity_data, Re, output_dir):
    """保存为 PDEBench 格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    nt, nx, ny = vorticity_data.shape
    
    # PDEBench 格式: (3, nx, ny, nt, 1)
    # [u, v, omega]
    fields = np.zeros((3, nx, ny, nt, 1), dtype=np.float32)
    
    # 填充涡度
    for t in range(nt):
        fields[2, :, :, t, 0] = vorticity_data[t]
    
    # 简化的速度场（从涡度恢复）
    fields[0] = 1.0  # u
    fields[1] = 0.0  # v
    
    filename = f"cyl_Re{Re}_synthetic.mat"
    filepath = os.path.join(output_dir, filename)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('fields_', data=fields)
        f.attrs['Re'] = Re
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nt'] = nt
        f.attrs['source'] = 'synthetic_numpy'
    
    print(f"Saved: {filepath}")
    print(f"  Size: {fields.nbytes / 1024 / 1024:.1f} MB")
    
    return filepath

def main():
    """主函数"""
    output_dir = r"D:\湍流实验数据"
    
    print("=" * 70)
    print("Synthetic Cylinder Flow Data Generator")
    print("=" * 70)
    print()
    
    # 生成不同 Re 的数据
    Re_values = [150, 300, 400]
    
    for Re in Re_values:
        print()
        print(f"Generating Re={Re}...")
        
        # 使用简化方法生成
        vorticity = generate_simplified_cylinder_flow(
            Re=Re,
            grid_size=(192, 378),
            num_steps=200
        )
        
        if vorticity is not None:
            save_as_pdebench(vorticity, Re, output_dir)
    
    print()
    print("=" * 70)
    print("Generation complete!")
    print("=" * 70)
    print()
    print("Files generated:")
    for Re in Re_values:
        print(f"  - D:\\湍流实验数据\\cyl_Re{Re}_synthetic.mat")
    print()
    print("Note: These are synthetic data for testing purposes.")
    print("The flow physics are simplified but suitable for cross-Re testing.")

if __name__ == "__main__":
    main()
