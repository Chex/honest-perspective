"""Smoke test the post-fix gravity integration end-to-end on the 9 samples."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import fix  # noqa: E402

SAMPLES = [
    ("IMG_5587.JPG", "house"), ("IMG_5606.JPG", "shelf"), ("IMG_5792.JPG", "street"),
    ("IMG_5980.jpeg", "bread"), ("IMG_5984.jpeg", "wall"),
    ("IMG_5302.JPG", "desk"), ("IMG_5362.JPG", "pagoda"), ("IMG_5893.JPG", "forest"),
    ("IMG_4501.jpeg", "yogurt"),
]


def main():
    print(f"{'image':<18} {'scene':<8} {'norm':>6} {'trust':>5}  "
          f"{'chosen':<10} {'area':>5}  {'g_used':>6} {'phys':>5}  "
          f"{'both_ang':>8} {'both_reason':<10}")
    print("-" * 110)
    for name, scene in SAMPLES:
        bgr, _, exif = fix.load_image(str(ROOT / "samples" / name))
        grav = fix.apple_acceleration_from_exif(exif)
        norm = grav["norm"] if grav else None
        trusted = grav["trusted"] if grav else None
        results = fix.auto_correct_all_modes(bgr, gravity=grav, gravity_mode="auto")
        chosen, _ = fix.choose_auto_mode(results)
        r = results.get(chosen, {}) or {}
        meta = r.get("meta", {}) or {}
        g_used = meta.get("gravity_used")
        physics = (meta.get("physics") or {}).get("accepted")
        area = r.get("area_ratio") or 0
        both = results.get("both") or {}
        both_meta = (both.get("meta") if isinstance(both, dict) else None) or {}
        both_ang = both_meta.get("gravity_visual_angle_deg")
        both_reason = both.get("reason") if isinstance(both, dict) else "?"
        print(f"{name:<18} {scene:<8} "
              f"{norm if norm is not None else 0:>6.3f} "
              f"{str(trusted)[:5]:>5}  "
              f"{str(chosen):<10} {area:>5.2f}  "
              f"{str(g_used)[:5]:>6} {str(physics)[:5]:>5}  "
              f"{(f'{both_ang:.1f}°' if both_ang is not None else '-'):>8} "
              f"{str(both_reason)[:10]:<10}")


if __name__ == "__main__":
    main()
