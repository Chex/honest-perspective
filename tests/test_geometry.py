import json
import math
from pathlib import Path
import subprocess
import unittest

import cv2
import numpy as np

import fix
from geometry import (
    compute_view,
    interpolate_rotation,
    make_correction_state,
    make_intrinsics,
    rotation_angle_deg,
    rotation_from_vector,
    validate_correction_state,
)


ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_exif_focal_length_uses_full_frame_diagonal(self):
        intrinsics = make_intrinsics(5712, 4284, focal_35mm=26)
        expected = 26 * math.hypot(5712, 4284) / math.hypot(36, 24)
        self.assertEqual(intrinsics["source"], "exif_35mm")
        self.assertAlmostEqual(intrinsics["fx"], expected, places=9)

    def test_strength_interpolates_on_rotation_manifold(self):
        rotation = rotation_from_vector([0.0, math.radians(30), 0.0])
        halfway = interpolate_rotation(rotation, 0.5)
        self.assertAlmostEqual(rotation_angle_deg(halfway), 15.0, places=9)
        np.testing.assert_allclose(halfway.T @ halfway, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(halfway)), 1.0, places=12)

    def test_state_rejects_non_rotation(self):
        intrinsics = make_intrinsics(800, 600, focal_35mm=26)
        state = make_correction_state(np.eye(3), intrinsics, [800, 600])
        state["rotation"][0] = 1.1
        validation = validate_correction_state(state, [800, 600])
        self.assertFalse(validation["accepted"])
        self.assertIn("SO(3)", validation["reason"])

    def test_state_crop_contains_no_black_pixels(self):
        size = [800, 600]
        intrinsics = make_intrinsics(*size, focal_35mm=26)
        rotation = rotation_from_vector(np.radians([8.0, -12.0, 3.0]))
        state = make_correction_state(rotation, intrinsics, size)
        validation = validate_correction_state(state, size)
        self.assertTrue(validation["accepted"], validation["reason"])

        source = np.full((size[1], size[0]), 255, dtype=np.uint8)
        output = cv2.warpPerspective(source, validation["view"]["matrix"], tuple(size))
        self.assertGreaterEqual(int(output.min()), 250)

    def test_browser_and_backend_geometry_match(self):
        rng = np.random.default_rng(20260710)
        fixtures = []
        python_results = []
        for _ in range(24):
            width, height = 800, 600
            intrinsics = make_intrinsics(width, height, focal_35mm=float(rng.choice([24, 26, 35, 52])))
            vector = np.radians(rng.uniform(-12, 12, size=3))
            rotation = rotation_from_vector(vector)
            fixtures.append({
                "rotation": rotation.reshape(-1).tolist(),
                "intrinsics": intrinsics,
                "imageSize": [width, height],
            })
            result = compute_view(rotation, intrinsics, [width, height])
            python_results.append(result)

        script = """
const fs = require('fs');
const geometry = require('./webapp/geometry.js');
const fixtures = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = fixtures.map(f => geometry.computeView(
  f.rotation, f.intrinsics, f.imageSize
));
process.stdout.write(JSON.stringify(results));
"""
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            input=json.dumps(fixtures),
            text=True,
            capture_output=True,
            check=True,
        )
        browser_results = json.loads(completed.stdout)
        for python_result, browser_result in zip(python_results, browser_results):
            np.testing.assert_allclose(
                python_result["matrix"].reshape(-1), browser_result["matrix"],
                rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                python_result["crop"], browser_result["crop"], rtol=1e-10, atol=1e-10
            )
            self.assertAlmostEqual(
                python_result["theta_deg"], browser_result["thetaDeg"], places=9
            )

    def test_manual_drag_maps_screen_axes_to_camera_pitch_and_yaw(self):
        script = r"""
const geometry = require('./webapp/geometry.js');
const intrinsics = {fx: 600.925, fy: 600.925, cx: 400, cy: 300};
const starts = {
  identity: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  sample5606: [
    0.9990729478240148, -0.0024003169030011916, 0.04298236155679296,
    -0.0033504919349792326, 0.9910802568364584, 0.13322424221129425,
    -0.04291875033161317, -0.13324474844340045, 0.9901534314853571,
  ],
  sample5980: [
    0.9994562317620974, -0.02126261882953526, 0.025202020403504194,
    0.020069813471978535, 0.9987085724638712, 0.046673224383756214,
    -0.02616186880000621, -0.04614204511816043, 0.9985922432570787,
  ],
};

function transpose(rotation) {
  return [
    rotation[0], rotation[3], rotation[6],
    rotation[1], rotation[4], rotation[7],
    rotation[2], rotation[5], rotation[8],
  ];
}

const results = [];
for (const [name, startRotation] of Object.entries(starts)) {
  for (const displacement of [[-120, 0], [0, 90], [-120, 90]]) {
    const expectedDelta = geometry.rotationFromVector([
      -Math.atan(displacement[1] / intrinsics.fy),
      Math.atan(displacement[0] / intrinsics.fx),
      0,
    ]);
    const rotation = geometry.cameraRotationForDrag(
      startRotation, displacement, intrinsics
    );
    const actualDelta = geometry.multiply(rotation, transpose(startRotation));
    const maxMatrixError = Math.max(...actualDelta.map(
      (value, index) => Math.abs(value - expectedDelta[index])
    ));

    let previous = startRotation;
    let maxStepDeg = 0;
    for (let step = 1; step <= 80; step++) {
      const progress = step / 80;
      const next = geometry.cameraRotationForDrag(
        startRotation,
        [displacement[0] * progress, displacement[1] * progress],
        intrinsics
      );
      const stepDelta = geometry.multiply(next, transpose(previous));
      maxStepDeg = Math.max(maxStepDeg, geometry.rotationAngleDeg(stepDelta));
      previous = next;
    }
    results.push({name, displacement, maxMatrixError, maxStepDeg});
  }
}
process.stdout.write(JSON.stringify(results));
"""
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        results = json.loads(completed.stdout)
        self.assertEqual(len(results), 9)
        for result in results:
            self.assertLess(result["maxMatrixError"], 1e-12)
            self.assertLess(
                result["maxStepDeg"],
                1.0,
                f'{result["name"]} {result["displacement"]} was discontinuous',
            )


class AutoSelectionTests(unittest.TestCase):
    @staticmethod
    def results(mode="vertical", improvement=1.0, weighted_support=0.8,
                gravity_used=False, gravity_angle=None):
        none = {
            "cropped": object(), "corners": [[0, 0]] * 4,
            "area_ratio": 1.0, "reason": None,
            "meta": {"identity": True},
        }
        candidate = {
            "cropped": object(), "corners": [[0, 0]] * 4,
            "area_ratio": 0.9, "reason": None,
            "meta": {
                "alignment": {"improvement_deg": improvement},
                "gravity_used": gravity_used,
                "gravity_visual_angle_deg": gravity_angle,
                "vp_quality": {
                    "vertical": {"weighted_inlier_ratio": weighted_support},
                    "horizontal": {"weighted_inlier_ratio": weighted_support},
                },
            },
        }
        return {"none": none, mode: candidate}

    def test_no_op_wins_without_measurable_improvement(self):
        mode, _result = fix.choose_auto_mode(self.results(improvement=0.1))
        self.assertEqual(mode, "none")

    def test_horizontal_requires_supported_vanishing_point(self):
        mode, _result = fix.choose_auto_mode(
            self.results(mode="horizontal", weighted_support=0.4)
        )
        self.assertEqual(mode, "none")

    def test_gravity_conflict_with_strong_visual_evidence_abstains(self):
        mode, _result = fix.choose_auto_mode(self.results(
            gravity_used=True, gravity_angle=12.0, weighted_support=0.8
        ))
        self.assertEqual(mode, "none")

    def test_supported_improving_candidate_can_win(self):
        mode, _result = fix.choose_auto_mode(self.results())
        self.assertEqual(mode, "vertical")


if __name__ == "__main__":
    unittest.main()
