import numpy as np, csv
from pipeline.ingest import load_capture
b = load_capture('../recordings-1')
ts_imu=[]; acc=[]
with open('../recordings-1/imu.csv') as f:
    r=csv.reader(f); next(r)
    for row in r:
        if not row or not row[0].strip(): continue
        ts_imu.append(float(row[0])); acc.append([float(row[1]),float(row[2]),float(row[3])])
ts_imu=np.array(ts_imu); acc=np.array(acc)
print('imu samples', len(acc), '|a| mean %.3f std %.3f (g units => gravity dominates)' % (np.linalg.norm(acc,1 if False else None,axis=1).mean(), np.linalg.norm(acc,axis=1).std()))

# nearest pose per imu sample
j = np.clip(np.searchsorted(b.timestamps, ts_imu), 0, len(b)-1)
R = b.poses[j][:,:3,:3]
# device-frame accel -> world. Try identity mapping device->camera first.
world = np.einsum('nij,nj->ni', R, acc)
g = world.mean(0); g/=np.linalg.norm(g)
print('mean accel in world (device axes == camera axes):', np.round(g,3))
print('  consistency: mean |unit dot g| = %.3f (1.0 = perfectly consistent)' %
      np.abs((world/np.linalg.norm(world,axis=1,keepdims=True))@g).mean())

# iPhone device frame vs OpenCV camera frame differ by a rotation; try the 4
# 90-degree rotations about Z plus the standard portrait mapping.
cands = {
 'identity': np.eye(3),
 'devY->camY_flip': np.diag([1.,-1,-1]),
 'rotZ90': np.array([[0.,-1,0],[1,0,0],[0,0,1]]),
 'rotZ-90': np.array([[0.,1,0],[-1,0,0],[0,0,1]]),
 'swapXY': np.array([[0.,1,0],[1,0,0],[0,0,-1]]),
}
print()
for name,M in cands.items():
    w = np.einsum('nij,nj->ni', R, acc@M.T)
    m = w.mean(0); mn=m/np.linalg.norm(m)
    cons = np.abs((w/np.linalg.norm(w,axis=1,keepdims=True))@mn).mean()
    print(f'{name:16s} world-gravity {np.round(mn,3)}  |mean|={np.linalg.norm(m):.3f} consistency={cons:.3f}')
