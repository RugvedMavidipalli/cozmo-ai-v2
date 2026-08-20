import numpy as np, open3d as o3d, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline.ingest import load_capture
from pipeline.geometry import estimate_gravity

pc = o3d.io.read_point_cloud('out/raw_cloud.ply')
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
p=np.asarray(pc.points); n=np.asarray(pc.normals)
b = load_capture('../recordings-1'); t=b.poses[:,:3,3]
print('IMU up %s  consistency %.3f' % (np.round(b.gravity_up,4), b.gravity_consistency))
g = estimate_gravity(p, hint=b.gravity_up, normals=n)
tilt = np.degrees(np.arccos(np.clip(g.up@b.gravity_up,-1,1)))
print('refined up %s  (%.2f deg from IMU)' % (np.round(g.up,4), tilt))
print('floor %.3f  ceiling %.3f  ROOM HEIGHT %.3f m  inliers %.1f%%' %
      (g.floor_height, g.ceiling_height, g.room_height, 100*g.inlier_fraction))
print('camera %.2f m above floor' % ((t@g.up).mean()-g.floor_height))

up=g.up; e1=np.array([1.,0,0]); e1-=up*(up@e1); e1/=np.linalg.norm(e1); e2=np.cross(up,e1)
xy=np.stack([p@e1,p@e2],1); txy=np.stack([t@e1,t@e2],1); h=p@up
fig,ax=plt.subplots(1,3,figsize=(19,5.5))
hh,e=np.histogram(h,bins=400); ax[0].plot((e[:-1]+e[1:])/2,hh)
ax[0].axvline(g.floor_height,color='g',ls='--',label='floor'); ax[0].axvline(g.ceiling_height,color='r',ls='--',label='ceiling')
ax[0].legend(); ax[0].set_title('height along refined up'); ax[0].set_xlabel('m')
ax[1].hexbin(xy[:,0],xy[:,1],gridsize=340,bins='log',cmap='inferno'); ax[1].plot(txy[:,0],txy[:,1],'c-',lw=1)
ax[1].set_aspect('equal'); ax[1].set_title('top-down ALL')
band=(h>g.floor_height+0.4)&(h<g.floor_height+2.1)
ax[2].hexbin(xy[band,0],xy[band,1],gridsize=340,bins='log',cmap='inferno'); ax[2].plot(txy[:,0],txy[:,1],'c-',lw=1.2)
ax[2].set_aspect('equal'); ax[2].set_title('WALL BAND 0.4-2.1m above floor')
plt.tight_layout(); plt.savefig('out/diag.png',dpi=85)
