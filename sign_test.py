import numpy as np, open3d as o3d
from pipeline.ingest import load_capture
pc = o3d.io.read_point_cloud('out/raw_cloud.ply'); p=np.asarray(pc.points)
b = load_capture('../recordings-1'); cam = b.poses[:,:3,3]
y = p[:,1]; cy = cam[:,1]
print('scene Y: %.2f .. %.2f' % (y.min(), y.max()))
print('camera Y: %.2f .. %.2f (mean %.2f, std %.3f)' % (cy.min(), cy.max(), cy.mean(), cy.std()))
h,e = np.histogram(y, bins=500); c=(e[:-1]+e[1:])/2
peaks = np.argsort(h)[-12:][::-1]
print('\ntop density bands in Y:')
for i in sorted(peaks, key=lambda i:-h[i])[:8]:
    print('   Y=%+.3f  n=%6d   (camera_mean - Y = %+.2f m)' % (c[i], h[i], cy.mean()-c[i]))
# A walkthrough camera sits ~1.1-1.7m above the floor. Which sign puts a strong
# horizontal surface at that offset?
for name,up in (('up=+Y', np.array([0.,1,0])), ('up=-Y', np.array([0.,-1,0]))):
    hh = p@up; ch=cam@up
    strong = c[h > 0.3*h.max()]
    proj = strong if up[1]>0 else -strong[::-1]
    floor = proj.min()
    print('\n%s -> lowest strong band %.2f, camera %.2f above it' % (name, floor, ch.mean()-floor))
