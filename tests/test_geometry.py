import json
import math
from pathlib import Path
import subprocess
import sys
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
FIXTURES_PATH = ROOT / "tests" / "HonestGeometryTests" / "Fixtures" / "geometry_contract.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_geometry_fixtures  # noqa: E402


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


class GeometryContractFixtureTests(unittest.TestCase):
    """Python, JS, and Swift all verify against the same checked-in fixtures.

    The Swift side runs via `swift test`; regenerate the fixtures with
    `python tests/generate_geometry_fixtures.py` after contract changes.
    """

    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(FIXTURES_PATH.read_text())

    def assert_deep_close(self, actual, expected, path="$"):
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertEqual(sorted(actual), sorted(expected), path)
            for key in expected:
                self.assert_deep_close(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, list):
            self.assertIsInstance(actual, list, path)
            self.assertEqual(len(actual), len(expected), path)
            for index, (a, b) in enumerate(zip(actual, expected)):
                self.assert_deep_close(a, b, f"{path}[{index}]")
        elif isinstance(expected, float):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, err_msg=path)
        else:
            self.assertEqual(actual, expected, path)

    def test_fixture_file_matches_generator(self):
        # Catches contract edits whose fixtures were not regenerated.
        self.assert_deep_close(generate_geometry_fixtures.build_fixtures(), self.fixtures)

    def test_python_reference_matches_fixtures(self):
        for case in self.fixtures["makeIntrinsics"]:
            intrinsics = make_intrinsics(case["width"], case["height"], focal_35mm=case["focal35mm"])
            expected = dict(case["expected"])
            self.assertEqual(intrinsics["source"], expected.pop("source"))
            self.assertEqual(intrinsics["focal_35mm"], expected.pop("focal35mm"))
            for key, value in expected.items():
                self.assertAlmostEqual(intrinsics[key], value, places=9, msg=key)
        for case in self.fixtures["computeView"]:
            view = compute_view(
                np.asarray(case["rotation"]).reshape(3, 3),
                case["intrinsics"],
                case["imageSize"],
                crop=case.get("crop"),
            )
            expected = case["expected"]
            np.testing.assert_allclose(
                view["matrix"].reshape(-1), expected["matrix"], rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(view["crop"], expected["crop"], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                view["canonical_crop"], expected["canonicalCrop"], rtol=1e-10, atol=1e-10
            )
            self.assertAlmostEqual(view["theta_deg"], expected["thetaDeg"], places=9)
            self.assertAlmostEqual(
                view["projective_w_ratio"], expected["projectiveWRatio"], places=9
            )
        for case in self.fixtures["interpolate"]:
            interpolated = interpolate_rotation(
                np.asarray(case["rotation"]).reshape(3, 3), case["strength"]
            )
            np.testing.assert_allclose(
                interpolated.reshape(-1), case["expectedRotation"], rtol=1e-10, atol=1e-10
            )

    def test_browser_matches_fixtures(self):
        completed = subprocess.run(
            ["node", str(ROOT / "tests" / "browser_contract_runner.js"), str(FIXTURES_PATH)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        results = json.loads(completed.stdout)
        for case, view in zip(self.fixtures["computeView"], results["computeView"]):
            expected = case["expected"]
            np.testing.assert_allclose(view["matrix"], expected["matrix"], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                view["homography"], expected["homography"], rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(view["crop"], expected["crop"], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                view["canonicalCrop"], expected["canonicalCrop"], rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                view["transformedCorners"], expected["transformedCorners"],
                rtol=1e-10, atol=1e-10,
            )
            self.assertAlmostEqual(view["thetaDeg"], expected["thetaDeg"], places=9)
            self.assertAlmostEqual(
                view["projectiveWRatio"], expected["projectiveWRatio"], places=9
            )
        for case, rotation in zip(self.fixtures["drag"], results["drag"]):
            np.testing.assert_allclose(
                rotation, case["expectedRotation"], rtol=1e-10, atol=1e-10
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
