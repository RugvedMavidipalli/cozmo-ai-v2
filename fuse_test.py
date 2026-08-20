import time, numpy as np, open3d as o3d
from pipeline.ingest import load_capture
from pipeline.fuse import fuse

b = load_capture('../recordings-1')
idx = np.arange(0, len(b), 4)
t0=time.time()
r = fuse(b, idx, voxel_size=0.03, min_confidence=1, max_depth=5.0)
print('fused %d frames in %.1fs' % (r.frame_count, time.time()-t0))
print('mesh: %d verts %d tris | cloud: %d pts' % (len(r.mesh.vertices), len(r.mesh.triangles), len(r.cloud.points)))
o3d.io.write_triangle_mesh('out/raw_mesh.ply', r.mesh)
o3d.io.write_point_cloud('out/raw_cloud.ply', r.cloud)
p = np.asarray(r.cloud.points)
print('extent', np.round(p.max(0)-p.min(0),2))
np.save('/tmp/pts.npy', p)
