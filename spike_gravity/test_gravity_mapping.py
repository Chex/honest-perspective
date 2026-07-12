import sys
import glob
import numpy as np
import cv2
import fix

def get_image_gravity(acc):
    Ax, Ay, Az = acc
    if abs(Ax) > abs(Ay):
        if Ax < 0:
            Ix, Iy, Iz = Ay, -Ax, -Az
        else:
            Ix, Iy, Iz = -Ay, Ax, -Az
    else:
        if Ay < 0:
            Ix, Iy, Iz = Ax, -Ay, -Az
        else:
            Ix, Iy, Iz = -Ax, Ay, -Az
            
    v = np.array([Ix, Iy, Iz], dtype=np.float64)
    return v / np.linalg.norm(v)

def main():
    samples = glob.glob("samples/*.[jJ][pP]*[gG]*")
    for path in sorted(samples):
        if "out" in path or "fixed" in path: continue
        
        bgr, icc, exif = fix.load_image(path)
        h, w = bgr.shape[:2]
        gravity_info = fix.apple_acceleration_from_exif(exif)
        
        if gravity_info is None:
            continue
            
        acc = gravity_info["vector"]
        g_img = get_image_gravity(acc)
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        verticals, horizontals = fix._detect_and_cluster(gray, w, h)
        vp_v = fix.ransac_vanishing_point(verticals)
        
        f = max(w, h)
        cx, cy = w / 2, h / 2
        Kinv = np.linalg.inv(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64))
        
        print(f"--- {path} ---")
        print(f"Mapped Gravity: {g_img}")
        
        if vp_v is not None:
            v_visual = Kinv @ np.array([vp_v[0], vp_v[1], 1.0])
            v_visual = v_visual / np.linalg.norm(v_visual)
            if v_visual[1] < 0:
                v_visual = -v_visual
                
            dot = np.clip(np.dot(g_img, v_visual), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))
            print(f"Visual VP ray : {v_visual}")
            print(f"Angle Diff    : {angle:.2f} degrees")
        else:
            print("Visual VP: None")

if __name__ == "__main__":
    main()
