"""
使用 FluidSim 在线求解生成 TGV 数据
=====================================
手动控制时间步进，直接读取 state_phys 保存为 HDF5 [T, 3, nx, ny, nz]
"""

import numpy as np
import h5py
import os
import sys
import argparse
from pathlib import Path


def generate_tgv(Re: float, N: int = 64, T: float = 5.0, dt_save: float = 0.02,
                 output_dir: str = 'data_generated'):
    """
    使用 FluidSim 生成 TGV 数据
    
    Args:
        Re: Reynolds number
        N:  空间分辨率
        T:  总模拟时间
        dt_save: 保存时间间隔
    """
    try:
        from fluidsim.solvers.ns3d.solver import Simul
    except ImportError as e:
        print(f"Error importing fluidsim: {e}")
        print("Please ensure fluidsim is installed: pip install fluidsim")
        sys.exit(1)
    
    params = Simul.create_default_params()
    params.oper.nx = N
    params.oper.ny = N
    params.oper.nz = N
    params.oper.Lx = 2 * np.pi
    params.oper.Ly = 2 * np.pi
    params.oper.Lz = 2 * np.pi
    
    params.time_stepping.t_end = T
    params.time_stepping.USE_CFL = True
    params.nu_2 = 1.0 / Re
    
    params.init_fields.type = 'in_script'
    
    # 禁用文件输出，我们手动保存
    params.output.HAS_TO_SAVE = False
    params.output.ONLINE_PLOT_OK = False
    
    sim = Simul(params)
    
    # 设置 TGV 初始条件
    # FluidSim axes: ('z', 'y', 'x'), get_XYZ_loc returns arrays of shape [nz, ny, nx]
    X, Y, Z = sim.oper.get_XYZ_loc()
    sim.state.state_phys[0][:] = np.sin(X) * np.cos(Y) * np.cos(Z)
    sim.state.state_phys[1][:] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    sim.state.state_phys[2][:] = 0.0
    sim.state.statespect_from_statephys()
    
    print(f"Running FluidSim TGV: Re={Re}, N={N}, T={T}, dt_save={dt_save}")
    print(f"Initial energy: {sim.state.compute_energy_phys():.6e}")
    
    # 手动时间步进
    ts = sim.time_stepping
    ts.prepare_main_loop()
    
    times_list = [0.0]
    fields_list = []
    
    # 保存初始场（转置到 [nx, ny, nz]）
    vx = sim.state.state_phys[0].transpose(2, 1, 0)
    vy = sim.state.state_phys[1].transpose(2, 1, 0)
    vz = sim.state.state_phys[2].transpose(2, 1, 0)
    fields_list.append(np.array([vx, vy, vz]))
    
    next_save_time = dt_save
    step_count = 0
    
    while not ts.is_simul_completed():
        ts.one_time_step()
        step_count += 1
        
        if ts.t >= next_save_time or ts.is_simul_completed():
            vx = sim.state.state_phys[0].transpose(2, 1, 0)
            vy = sim.state.state_phys[1].transpose(2, 1, 0)
            vz = sim.state.state_phys[2].transpose(2, 1, 0)
            fields_list.append(np.array([vx, vy, vz]))
            times_list.append(float(ts.t))
            
            if len(times_list) % 50 == 0 or ts.is_simul_completed():
                print(f"  t={ts.t:.3f}, it={ts.it}, energy={sim.state.compute_energy_phys():.4e}")
            
            next_save_time += dt_save
    
    ts.finalize_main_loop()
    
    times = np.array(times_list)
    fields = np.array(fields_list)
    
    print(f"Extracted {len(times)} snapshots, fields shape: {fields.shape}")
    print(f"Final energy: {sim.state.compute_energy_phys():.6e}")
    
    # 保存为标准格式
    n_total = len(times)
    n_train = int(n_total * 0.8)
    
    train_inputs = fields[:n_train - 1]
    train_targets = fields[1:n_train]
    val_inputs = fields[n_train:-1]
    val_targets = fields[n_train + 1:]
    
    output_path = os.path.join(
        output_dir,
        f'tgv_re{int(Re)}_N{N}_T{T}_fluidsim.h5'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('times', data=times)
        f.create_dataset('fields', data=fields)
        f.create_dataset('train_inputs', data=train_inputs)
        f.create_dataset('train_targets', data=train_targets)
        f.create_dataset('val_inputs', data=val_inputs)
        f.create_dataset('val_targets', data=val_targets)
    
    print(f"Saved: {output_path}")
    print(f"  Total snapshots: {n_total}")
    print(f"  Train pairs: {len(train_inputs)}")
    print(f"  Val pairs: {len(val_inputs)}")
    
    return times, fields


def main():
    parser = argparse.ArgumentParser(description='Generate TGV data via FluidSim')
    parser.add_argument('--Re', type=float, default=1600, help='Reynolds number')
    parser.add_argument('--N', type=int, default=64, help='Spatial resolution')
    parser.add_argument('--T', type=float, default=5.0, help='Total simulation time')
    parser.add_argument('--dt_save', type=float, default=0.02, help='Save time interval')
    parser.add_argument('--output_dir', type=str, default='data_generated', help='Output directory')
    args = parser.parse_args()
    
    generate_tgv(args.Re, args.N, args.T, args.dt_save, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
