"""
3D流场动态可视化与预测精度时间序列分析

功能:
1. 3D流场True vs Predicted对比图 (静态+动态GIF)
2. 随时刻变化的预测精度曲线
3. 多视角3D渲染

结果保存: D:\AxiomOS_Project\experiments_results\3d_visualization\
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import json
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = 'D:\\AxiomOS_Project\\experiments_results\\3d_visualization'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("3D FLOW FIELD VISUALIZATION & TEMPORAL ACCURACY ANALYSIS")
print("=" * 80)
print(f"Results: {RESULTS_DIR}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")


# ============================================
# 模型定义 (最优配置)
# ============================================
class HardCore3D(nn.Module):
    """3D Hard Core - 物理扩散"""
    def __init__(self, dt=0.01):
        super().__init__()
        self.dt = dt
        self.nu = nn.Parameter(torch.tensor([0.001]))
    
    def laplacian_3d(self, x):
        """3D拉普拉斯算子"""
        return (torch.roll(x, 1, 2) + torch.roll(x, -1, 2) +
                torch.roll(x, 1, 3) + torch.roll(x, -1, 3) +
                torch.roll(x, 1, 4) + torch.roll(x, -1, 4) - 6*x)
    
    def forward(self, x):
        return x + self.dt * torch.abs(self.nu) * self.laplacian_3d(x)


class SoftShell3D(nn.Module):
    """3D Soft Shell - 简化版"""
    def __init__(self, in_ch=3, out_ch=3, base_ch=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_ch, base_ch*2, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_ch*2, base_ch, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_ch, out_ch, 3, padding=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class OptimizedRCLN3D(nn.Module):
    """最优配置3D RCLN"""
    def __init__(self, upi_freq=5):
        super().__init__()
        self.upi_freq = upi_freq
        self.hard_core = HardCore3D()
        self.soft_shell = SoftShell3D()
        
        alpha_logit = np.log(0.9 / 0.1)
        self.alpha_param = nn.Parameter(torch.tensor([alpha_logit]))
    
    def get_alpha(self):
        return torch.sigmoid(self.alpha_param).item()
    
    def forward(self, x, target=None, step_idx=0):
        hard = self.hard_core(x)
        soft = self.soft_shell(x)
        alpha = self.get_alpha()
        pred = alpha * hard + (1 - alpha) * soft
        return pred


# ============================================
# 3D流场数据生成
# ============================================
def generate_3d_taylor_green_vortex(nx=32, ny=32, nz=32, n_steps=50, Re=400):
    """生成3D Taylor-Green涡数据"""
    print("\n[Generating 3D Taylor-Green Vortex data...]")
    
    # 3D网格
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)
    z = np.linspace(0, 2*np.pi, nz, endpoint=False)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 初始速度场 (Taylor-Green)
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)
    
    # 转换为涡量场
    def velocity_to_vorticity(u, v, w):
        dwdy = np.gradient(w, y, axis=1)
        dvdz = np.gradient(v, z, axis=2)
        dudz = np.gradient(u, z, axis=2)
        dwdx = np.gradient(w, x, axis=0)
        dvdx = np.gradient(v, x, axis=0)
        dudy = np.gradient(u, y, axis=1)
        
        omega_x = dwdy - dvdz
        omega_y = dudz - dwdx
        omega_z = dvdx - dudy
        
        return np.stack([omega_x, omega_y, omega_z], axis=0)
    
    # 生成时间序列
    sequence = []
    dt = 0.01
    nu = 1.0 / Re
    
    for t in range(n_steps):
        # 计算涡量
        vorticity = velocity_to_vorticity(u, v, w)
        sequence.append(vorticity)
        
        # 时间演化 (简化)
        decay = np.exp(-2 * nu * dt)
        u *= decay
        v *= decay
        w *= decay
    
    sequence = np.array(sequence)  # [n_steps, 3, nx, ny, nz]
    print(f"  Generated: {sequence.shape}")
    
    return torch.from_numpy(sequence).float()


# ============================================
# 1. 3D流场True vs Predicted对比图
# ============================================
def create_3d_flow_comparison():
    """创建3D流场对比图"""
    print("\n" + "=" * 80)
    print("CREATING 3D FLOW COMPARISON VISUALIZATION")
    print("=" * 80)
    
    # 生成数据
    true_data = generate_3d_taylor_green_vortex(nx=24, ny=24, nz=24, n_steps=20, Re=400)
    
    # 创建模型
    model = OptimizedRCLN3D(upi_freq=5).to(device)
    model.eval()
    
    # 生成预测序列
    print("\n[Generating predictions...]")
    pred_data = []
    with torch.no_grad():
        current = true_data[0:1].to(device)
        pred_data.append(current.cpu())
        
        for t in range(1, len(true_data)):
            current = model(current)
            pred_data.append(current.cpu())
    
    pred_data = torch.cat(pred_data, dim=0)
    
    # 选择关键时间步进行可视化
    timesteps = [0, 5, 10, 15, 19]
    
    for idx, t in enumerate(timesteps):
        print(f"\n  [Creating visualization for t={t}...]")
        
        true_frame = true_data[t].numpy()
        pred_frame = pred_data[t].numpy()
        
        # 计算误差
        error_frame = np.abs(true_frame - pred_frame)
        
        # 创建3D可视化图
        fig = plt.figure(figsize=(20, 5))
        
        # 提取Z方向涡量分量用于可视化
        true_z = true_frame[2]  # [nx, ny, nz]
        pred_z = pred_frame[2]
        error_z = error_frame[2]
        
        nx, ny, nz = true_z.shape
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        z = np.linspace(0, 2*np.pi, nz)
        
        # True
        ax1 = fig.add_subplot(141, projection='3d')
        plot_3d_vorticity(ax1, true_z, x, y, z, title=f'True (t={t})', cmap='RdBu_r')
        
        # Predicted
        ax2 = fig.add_subplot(142, projection='3d')
        plot_3d_vorticity(ax2, pred_z, x, y, z, title=f'Predicted (t={t})', cmap='RdBu_r')
        
        # Error
        ax3 = fig.add_subplot(143, projection='3d')
        plot_3d_vorticity(ax3, error_z, x, y, z, title=f'Error (t={t})', cmap='hot', vmin=0, vmax=np.max(error_z)*0.5)
        
        # 切片对比 (XY平面)
        ax4 = fig.add_subplot(144)
        mid_z = nz // 2
        im = ax4.imshow(np.abs(true_z[:, :, mid_z] - pred_z[:, :, mid_z]), 
                       cmap='hot', origin='lower')
        ax4.set_title(f'|True - Pred| at z=mid (t={t})')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/3d_comparison_t{t}.png', dpi=150, bbox_inches='tight')
        print(f"    Saved: {RESULTS_DIR}/3d_comparison_t{t}.png")
        plt.close()
    
    return true_data, pred_data


def plot_3d_vorticity(ax, vorticity, x, y, z, title, cmap='RdBu_r', vmin=None, vmax=None):
    """绘制3D涡量场的等值面"""
    nx, ny, nz = vorticity.shape
    
    # 创建网格
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 绘制等值面 (使用contour3D的简化版本 - 切片)
    mid_x, mid_y, mid_z = nx//2, ny//2, nz//2
    
    # 绘制三个正交切片
    vmin = vmin if vmin is not None else np.min(vorticity)
    vmax = vmax if vmax is not None else np.max(vorticity)
    
    # X=mid切片
    ax.contourf(X[mid_x, :, :], Y[mid_x, :, :], vorticity[mid_x, :, :], 
               zdir='x', offset=x[mid_x], levels=20, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
    
    # Y=mid切片
    ax.contourf(X[:, mid_y, :], Z[:, mid_y, :], vorticity[:, mid_y, :], 
               zdir='y', offset=y[mid_y], levels=20, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
    
    # Z=mid切片
    ax.contourf(X[:, :, mid_z], Y[:, :, mid_z], vorticity[:, :, mid_z], 
               zdir='z', offset=z[mid_z], levels=20, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([y[0], y[-1]])
    ax.set_zlim([z[0], z[-1]])


# ============================================
# 2. 随时刻变化的预测精度曲线
# ============================================
def create_temporal_accuracy_analysis(true_data, pred_data):
    """创建时间序列精度分析"""
    print("\n" + "=" * 80)
    print("CREATING TEMPORAL ACCURACY ANALYSIS")
    print("=" * 80)
    
    n_steps = len(true_data)
    
    # 计算各时间步的精度指标
    mse_history = []
    mae_history = []
    rmse_history = []
    corr_history = []
    drift_history = []
    
    for t in range(n_steps):
        true_frame = true_data[t].numpy()
        pred_frame = pred_data[t].numpy()
        
        # MSE
        mse = np.mean((true_frame - pred_frame)**2)
        mse_history.append(mse)
        
        # MAE
        mae = np.mean(np.abs(true_frame - pred_frame))
        mae_history.append(mae)
        
        # RMSE
        rmse = np.sqrt(mse)
        rmse_history.append(rmse)
        
        # Correlation
        true_flat = true_frame.flatten()
        pred_flat = pred_frame.flatten()
        corr = np.corrcoef(true_flat, pred_flat)[0, 1]
        corr_history.append(corr)
        
        # Enstrophy drift
        true_enstrophy = np.sum(true_frame**2)
        pred_enstrophy = np.sum(pred_frame**2)
        drift = abs(pred_enstrophy - true_enstrophy) / (true_enstrophy + 1e-10)
        drift_history.append(drift)
    
    # 创建综合精度图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    time_steps = np.arange(n_steps)
    
    # MSE
    axes[0, 0].plot(time_steps, mse_history, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Mean Squared Error vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Threshold=0.1')
    axes[0, 0].legend()
    
    # RMSE
    axes[0, 1].plot(time_steps, rmse_history, 'g-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Root Mean Squared Error vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 2].plot(time_steps, mae_history, 'm-', linewidth=2, marker='^', markersize=4)
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].set_title('Mean Absolute Error vs Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Correlation
    axes[1, 0].plot(time_steps, corr_history, 'c-', linewidth=2, marker='d', markersize=4)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Correlation Coefficient')
    axes[1, 0].set_title('Pearson Correlation vs Time')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='High Correlation')
    axes[1, 0].legend()
    
    # Enstrophy Drift
    axes[1, 1].plot(time_steps, drift_history, 'r-', linewidth=2, marker='v', markersize=4)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Drift Ratio')
    axes[1, 1].set_title('Enstrophy Drift vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='10% Threshold')
    axes[1, 1].legend()
    
    # 综合指标对比 (对数尺度)
    axes[1, 2].semilogy(time_steps, mse_history, 'b-', linewidth=2, label='MSE', marker='o', markersize=4)
    axes[1, 2].semilogy(time_steps, rmse_history, 'g-', linewidth=2, label='RMSE', marker='s', markersize=4)
    axes[1, 2].semilogy(time_steps, mae_history, 'm-', linewidth=2, label='MAE', marker='^', markersize=4)
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Error (log scale)')
    axes[1, 2].set_title('All Error Metrics (Log Scale)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/temporal_accuracy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {RESULTS_DIR}/temporal_accuracy_analysis.png")
    plt.close()
    
    # 保存数据 (转换为Python原生类型)
    accuracy_data = {
        'time_steps': [int(t) for t in time_steps],
        'mse': [float(m) for m in mse_history],
        'mae': [float(m) for m in mae_history],
        'rmse': [float(r) for r in rmse_history],
        'correlation': [float(c) for c in corr_history],
        'drift': [float(d) for d in drift_history],
        'summary': {
            'mean_mse': float(np.mean(mse_history)),
            'max_mse': float(np.max(mse_history)),
            'final_mse': float(mse_history[-1]),
            'mean_correlation': float(np.mean(corr_history)),
            'mean_drift': float(np.mean(drift_history))
        }
    }
    
    with open(f'{RESULTS_DIR}/temporal_accuracy_data.json', 'w') as f:
        json.dump(accuracy_data, f, indent=2)
    
    print(f"  Saved: {RESULTS_DIR}/temporal_accuracy_data.json")
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("TEMPORAL ACCURACY SUMMARY")
    print("=" * 80)
    print(f"Mean MSE: {np.mean(mse_history):.6f}")
    print(f"Max MSE: {np.max(mse_history):.6f}")
    print(f"Final MSE (t={n_steps-1}): {mse_history[-1]:.6f}")
    print(f"Mean Correlation: {np.mean(corr_history):.4f}")
    print(f"Mean Drift: {np.mean(drift_history):.4f}")
    print("=" * 80)
    
    return accuracy_data


# ============================================
# 3. 动态GIF生成
# ============================================
def create_animated_comparison(true_data, pred_data):
    """创建动态对比GIF"""
    print("\n" + "=" * 80)
    print("CREATING ANIMATED COMPARISON (This may take a while...)")
    print("=" * 80)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        n_frames = min(20, len(true_data))  # 限制帧数
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 准备数据
        nx, ny, nz = true_data[0].shape[1:]
        mid_z = nz // 2
        
        def update(frame):
            # 清除之前的内容
            for ax in axes:
                ax.clear()
            
            true_frame = true_data[frame, 2, :, :, mid_z].numpy()  # Z分量，中间切片
            pred_frame = pred_data[frame, 2, :, :, mid_z].numpy()
            error_frame = np.abs(true_frame - pred_frame)
            
            # True
            im0 = axes[0].imshow(true_frame, cmap='RdBu_r', origin='lower', 
                                vmin=-np.max(np.abs(true_frame)), vmax=np.max(np.abs(true_frame)))
            axes[0].set_title(f'True (t={frame})')
            axes[0].axis('off')
            
            # Predicted
            im1 = axes[1].imshow(pred_frame, cmap='RdBu_r', origin='lower',
                                vmin=-np.max(np.abs(true_frame)), vmax=np.max(np.abs(true_frame)))
            axes[1].set_title(f'Predicted (t={frame})')
            axes[1].axis('off')
            
            # Error
            im2 = axes[2].imshow(error_frame, cmap='hot', origin='lower', vmin=0)
            axes[2].set_title(f'Error (t={frame})')
            axes[2].axis('off')
            
            return [im0, im1, im2]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
        
        # 保存为GIF
        anim.save(f'{RESULTS_DIR}/flow_evolution_animation.gif', writer='pillow', fps=5)
        print(f"  Saved: {RESULTS_DIR}/flow_evolution_animation.gif")
        
        plt.close()
        
    except Exception as e:
        print(f"  Animation creation failed: {e}")
        print("  (This is normal if pillow is not installed)")


# ============================================
# 4. 3D流线可视化
# ============================================
def create_3d_streamlines(true_data, pred_data):
    """创建3D流线可视化"""
    print("\n" + "=" * 80)
    print("CREATING 3D STREAMLINE VISUALIZATION")
    print("=" * 80)
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # 选择特定时间步
        t = 10
        true_frame = true_data[t].numpy()
        pred_frame = pred_data[t].numpy()
        
        nx, ny, nz = true_frame.shape[1:]
        
        # 创建流线图 (XY平面切片)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        mid_z = nz // 2
        
        # True流场
        u_true = true_frame[0, :, :, mid_z]  # X速度
        v_true = true_frame[1, :, :, mid_z]  # Y速度
        
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # 下采样以便可视化
        step = 2
        axes[0].quiver(X[::step, ::step], Y[::step, ::step], 
                      u_true[::step, ::step], v_true[::step, ::step],
                      scale=50, alpha=0.7)
        axes[0].set_title(f'True Flow Field (t={t}, z=mid)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        # Predicted流场
        u_pred = pred_frame[0, :, :, mid_z]
        v_pred = pred_frame[1, :, :, mid_z]
        
        axes[1].quiver(X[::step, ::step], Y[::step, ::step],
                      u_pred[::step, ::step], v_pred[::step, ::step],
                      scale=50, alpha=0.7)
        axes[1].set_title(f'Predicted Flow Field (t={t}, z=mid)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/streamline_comparison_t{t}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {RESULTS_DIR}/streamline_comparison_t{t}.png")
        plt.close()
        
    except Exception as e:
        print(f"  Streamline creation failed: {e}")


# ============================================
# 主程序
# ============================================
def main():
    start_time = datetime.now()
    
    # 1. 创建3D流场对比
    true_data, pred_data = create_3d_flow_comparison()
    
    # 2. 时间序列精度分析
    accuracy_data = create_temporal_accuracy_analysis(true_data, pred_data)
    
    # 3. 动态GIF (可选)
    create_animated_comparison(true_data, pred_data)
    
    # 4. 3D流线可视化
    create_3d_streamlines(true_data, pred_data)
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 80)
    print("3D VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed}")
    print("\nGenerated files:")
    print(f"  - {RESULTS_DIR}/3d_comparison_t*.png (5 time steps)")
    print(f"  - {RESULTS_DIR}/temporal_accuracy_analysis.png")
    print(f"  - {RESULTS_DIR}/temporal_accuracy_data.json")
    print(f"  - {RESULTS_DIR}/flow_evolution_animation.gif (if pillow available)")
    print(f"  - {RESULTS_DIR}/streamline_comparison_t10.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
