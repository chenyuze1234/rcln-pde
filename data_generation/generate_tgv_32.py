"""TGV N=32 — EXACT rfftn solver from generate_v5_stable.py, just N=32."""
import numpy as np, torch, torch.fft as fft, h5py, os

N,Re,dt,steps,warmup = 32, 1000, 0.001, 500, 50
nu=1.0/Re; device='cuda'
print(f'TGV N={N} dt={dt}')

kx1=fft.fftfreq(N,d=1.0/N)*2*np.pi; kz1=fft.rfftfreq(N,d=1.0/N)*2*np.pi
kx,ky,kz=torch.meshgrid(torch.tensor(kx1,dtype=torch.float32,device=device),
                         torch.tensor(kx1,dtype=torch.float32,device=device),
                         torch.tensor(kz1,dtype=torch.float32,device=device),indexing='ij')
k2=kx**2+ky**2+kz**2; k2_s=k2.clone(); k2_s[0,0,0]=1.0

torch.manual_seed(42)
km=torch.sqrt(k2); k0=N/4; Ek=(km/k0)**4*torch.exp(-(km/k0)**2); Ek[0,0,0]=0
amp=torch.sqrt(Ek/(km**2+1e-10))
uh=amp*torch.exp(1j*torch.rand_like(amp)*2*np.pi)
vh=amp*torch.exp(1j*torch.rand_like(amp)*2*np.pi)
wh=amp*torch.exp(1j*torch.rand_like(amp)*2*np.pi)
kd=kx*uh+ky*vh+kz*wh; uh-=kd*kx/k2_s; vh-=kd*ky/k2_s; wh-=kd*kz/k2_s
uhs=torch.stack([uh,vh,wh],dim=0)
u=fft.irfftn(uhs[0],s=(N,N,N)); v=fft.irfftn(uhs[1],s=(N,N,N)); w=fft.irfftn(uhs[2],s=(N,N,N))
sc=1.0/(torch.sqrt(torch.mean(u**2+v**2+w**2))+1e-8); uhs*=sc

def step(uh):
    u=fft.irfftn(uh[0],s=(N,N,N)); v=fft.irfftn(uh[1],s=(N,N,N)); w=fft.irfftn(uh[2],s=(N,N,N))
    du=fft.irfftn(1j*kx*uh[0],s=(N,N,N)); dy=fft.irfftn(1j*ky*uh[0],s=(N,N,N)); dz=fft.irfftn(1j*kz*uh[0],s=(N,N,N))
    dv=fft.irfftn(1j*kx*uh[1],s=(N,N,N)); de=fft.irfftn(1j*ky*uh[1],s=(N,N,N)); df=fft.irfftn(1j*kz*uh[1],s=(N,N,N))
    dw=fft.irfftn(1j*kx*uh[2],s=(N,N,N)); dg=fft.irfftn(1j*ky*uh[2],s=(N,N,N)); dh=fft.irfftn(1j*kz*uh[2],s=(N,N,N))
    cu=fft.rfftn(u*du+v*dy+w*dz); cv=fft.rfftn(u*dv+v*de+w*df); cw=fft.rfftn(u*dw+v*dg+w*dh)
    un=uh[0]-dt*(cu+nu*k2*uh[0]); vn=uh[1]-dt*(cv+nu*k2*uh[1]); wn=uh[2]-dt*(cw+nu*k2*uh[2])
    ns=torch.stack([un,vn,wn],dim=0)
    div=kx*ns[0]+ky*ns[1]+kz*ns[2]; phi=div/k2_s
    ns[0]-=kx*phi; ns[1]-=ky*phi; ns[2]-=kz*phi
    return ns

cur=uhs; fields=[]
for s in range(warmup+steps):
    cur=step(cur)
    if s>=warmup:
        ut=fft.irfftn(cur[0],s=(N,N,N)).cpu().numpy(); vt=fft.irfftn(cur[1],s=(N,N,N)).cpu().numpy(); wt=fft.irfftn(cur[2],s=(N,N,N)).cpu().numpy()
        fields.append(np.stack([ut,vt,wt],0).astype(np.float32))
    if s%50==0:
        uu=fft.irfftn(cur[0],s=(N,N,N)); ke=0.5*(uu**2).mean().item()
        print(f'  {s}: KE={ke:.6f}');

fields=np.stack(fields,0)
print(f'{fields.shape} {fields.nbytes/1e6:.1f}MB OK' if np.isfinite(fields).all() else 'NaN!')
if np.isfinite(fields).all():
    os.makedirs('data_generated',exist_ok=True)
    p=f'data_generated/tgv_re{Re}_N32_T{steps*dt:.1f}.h5'
    with h5py.File(p,'w') as f:
        f.create_dataset('fields',data=fields,compression='gzip',compression_opts=4)
        f.create_dataset('times',data=np.arange(steps)*dt)
        f.attrs['flow_type']='TGV';f.attrs['Re']=Re;f.attrs['N']=32
    print(f'Saved: {p}')
