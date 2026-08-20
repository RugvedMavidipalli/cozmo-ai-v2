"""Which pose convention actually aligns consecutive frames?"""
import numpy as np, itertools, cv2
from pipeline.ingest import load_capture, _read_odometry, ARKIT_TO_CV, _quaternion_to_matrix
from pathlib import Path

b = load_capture('../recordings-1')
ts, poses_cv, K = _read_odometry(Path('../recordings-1/odometry.csv'))
# raw ARKit poses (undo the flip we applied)
raw = poses_cv @ np.linalg.inv(ARKIT_TO_CV)

def cam_pts(i, stride=3):
    d = cv2.imread(f'../recordings-1/depth/{i:06d}.png', cv2.IMREAD_UNCHANGED).astype(np.float32)/1000.
    c = cv2.imread(f'../recordings-1/confidence/{i:06d}.png', cv2.IMREAD_UNCHANGED)
    d[(c<1)|(d>5)] = 0
    d = d[::stride,::stride]
    Kk = b.intrinsics
    hh,ww = d.shape
    u,v = np.meshgrid(np.arange(ww)*stride, np.arange(hh)*stride)
    m = d>0; z=d[m]
    return np.stack([(u[m]-Kk[0,2])*z/Kk[0,0], (v[m]-Kk[1,2])*z/Kk[1,1], z],1)

# Candidate conventions: how to map the stored rotation+translation to camera->world OpenCV
def variants(i):
    R = raw[i][:3,:3]; t = raw[i][:3,3]
    P = np.eye(4); P[:3,:3]=R; P[:3,3]=t
    Pinv = np.linalg.inv(P)
    F = np.diag([1.,-1,-1,1])
    return {
      'c2w_flip'   : P @ F,          # current assumption
      'c2w_noflip' : P,
      'w2c_flip'   : Pinv @ F,
      'w2c_noflip' : Pinv,
      'flip_c2w'   : F @ P,
    }

# For each variant, transform two nearby frames to world and measure how well
# they overlap (median nearest-neighbour distance). Correct convention => small.
from scipy.spatial import cKDTree
pairs = [(300,315),(1200,1215),(2400,2415)]
scores = {}
for name in variants(0):
    ds=[]
    for i,j in pairs:
        A = cam_pts(i); B = cam_pts(j)
        Ta = variants(i)[name]; Tb = variants(j)[name]
        Aw = A@Ta[:3,:3].T + Ta[:3,3]; Bw = B@Tb[:3,:3].T + Tb[:3,3]
        d,_ = cKDTree(Aw).query(Bw)
        ds.append(np.median(d))
    scores[name]=np.mean(ds)
for k,v in sorted(scores.items(), key=lambda x:x[1]):
    print(f'{k:14s} median NN dist {v*100:7.2f} cm')
