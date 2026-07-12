import sys
import glob
import numpy as np
import cv2
import fix

def main():
    samples = glob.glob("samples/*.[jJ][pP]*[gG]*")
    for path in sorted(samples):
        if "out" in path or "fixed" in path: continue
        
        bgr, icc, exif = fix.load_image(path)
        h, w = bgr.shape[:2]
        gravity_info = fix.apple_acceleration_from_exif(exif)
        
        if gravity_info is None:
            print(f"{path}: No gravity info")
            continue
            
        acc = gravity_info["vector"]
        orientation = gravity_info["orientation"]
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        verticals, horizontals = fix._detect_and_cluster(gray, w, h)
        vp_v = fix.ransac_vanishing_point(verticals)
        
        f = max(w, h)
        cx, cy = w / 2, h / 2
        Kinv = np.linalg.inv(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64))
        
        print(f"--- {path} ---")
        print(f"Accel (raw): {acc}")
        if vp_v is not None:
            v_visual = Kinv @ np.array([vp_v[0], vp_v[1], 1.0])
            v_visual = v_visual / np.linalg.norm(v_visual)
            if v_visual[1] < 0:
                v_visual = -v_visual
            print(f"Visual VP ray: {v_visual}")
        else:
            print("Visual VP: None")

if __name__ == "__main__":
    main()
