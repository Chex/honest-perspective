(function (root) {
  'use strict';

  const STATE_VERSION = 1;
  const CROP_POLICY = 'conservative_same_aspect_v1';

  function multiply(a, b) {
    const result = new Array(9).fill(0);
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        for (let k = 0; k < 3; k++) result[row * 3 + col] += a[row * 3 + k] * b[k * 3 + col];
      }
    }
    return result;
  }

  function inverse(m) {
    const adj = [
      m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
      m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
      m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3],
    ];
    const det = m[0] * adj[0] + m[1] * adj[3] + m[2] * adj[6];
    if (!Number.isFinite(det) || Math.abs(det) < 1e-12) throw new Error('matrix is singular');
    return adj.map(value => value / det);
  }

  function apply(matrix, point) {
    const x = matrix[0] * point[0] + matrix[1] * point[1] + matrix[2];
    const y = matrix[3] * point[0] + matrix[4] * point[1] + matrix[5];
    const w = matrix[6] * point[0] + matrix[7] * point[1] + matrix[8];
    if (!Number.isFinite(w) || Math.abs(w) < 1e-9) throw new Error('projection crosses infinity');
    return [x / w, y / w];
  }

  function rotationAngleDeg(rotation) {
    const cosine = Math.max(-1, Math.min(1, (rotation[0] + rotation[4] + rotation[8] - 1) / 2));
    return Math.acos(cosine) * 180 / Math.PI;
  }

  function rotationFromVector(vector) {
    const theta = Math.hypot(vector[0], vector[1], vector[2]);
    if (theta < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const x = vector[0] / theta, y = vector[1] / theta, z = vector[2] / theta;
    const c = Math.cos(theta), s = Math.sin(theta), t = 1 - c;
    return [
      t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
      t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
      t*x*z - s*y, t*y*z + s*x, t*z*z + c,
    ];
  }

  function composeRotation(rotationVector, rotation) {
    return multiply(rotationFromVector(rotationVector), rotation);
  }

  function intrinsicsMatrix(intrinsics) {
    return [
      Number(intrinsics.fx), 0, Number(intrinsics.cx),
      0, Number(intrinsics.fy), Number(intrinsics.cy),
      0, 0, 1,
    ];
  }

  function rotationHomography(rotation, intrinsics) {
    const k = intrinsicsMatrix(intrinsics);
    return multiply(k, multiply(rotation, inverse(k)));
  }

  function cameraRotationForDrag(initialRotation, displacement, intrinsics) {
    const dx = Number(displacement[0]), dy = Number(displacement[1]);
    const fx = Number(intrinsics.fx), fy = Number(intrinsics.fy);
    if (![dx, dy, fx, fy].every(Number.isFinite) || !(fx > 0 && fy > 0)) {
      throw new Error('drag and focal lengths must be finite');
    }
    const pitch = -Math.atan(dy / fy);
    const yaw = Math.atan(dx / fx);
    return composeRotation([pitch, yaw, 0], initialRotation);
  }

  function fitAspect(crop, targetAspect) {
    let [x0, y0, x1, y1] = crop.map(Number);
    const width = x1 - x0, height = y1 - y0;
    if (!(width > 0 && height > 0)) throw new Error('crop is empty');
    if (width / height > targetAspect) {
      const newWidth = height * targetAspect, center = (x0 + x1) / 2;
      x0 = center - newWidth / 2; x1 = center + newWidth / 2;
    } else {
      const newHeight = width / targetAspect, center = (y0 + y1) / 2;
      y0 = center - newHeight / 2; y1 = center + newHeight / 2;
    }
    return [x0, y0, x1, y1];
  }

  function conservativeCrop(corners, targetAspect) {
    const [tl, tr, br, bl] = corners;
    return fitAspect([
      Math.max(tl[0], bl[0]),
      Math.max(tl[1], tr[1]),
      Math.min(tr[0], br[0]),
      Math.min(br[1], bl[1]),
    ], targetAspect);
  }

  function computeView(rotation, intrinsics, imageSize, crop) {
    const [width, height] = imageSize.map(Number);
    const homography = rotationHomography(rotation, intrinsics);
    const sourceCorners = [[0, 0], [width, 0], [width, height], [0, height]];
    const denominators = sourceCorners.map(([x, y]) => homography[6] * x + homography[7] * y + homography[8]);
    const sameSide = denominators.every(value => value > 0) || denominators.every(value => value < 0);
    if (!sameSide) throw new Error('projection flips across the camera plane');
    const absW = denominators.map(Math.abs);
    const projectiveWRatio = Math.min(...absW) / Math.max(...absW);
    const transformedCorners = sourceCorners.map(point => apply(homography, point));
    const canonicalCrop = conservativeCrop(transformedCorners, width / height);
    const selectedCrop = crop ? fitAspect(crop, width / height) : canonicalCrop;
    const [x0, y0, x1, y1] = selectedCrop;
    const sx = width / (x1 - x0), sy = height / (y1 - y0);
    if (Math.abs(sx - sy) > 1e-6 * Math.max(1, sx, sy)) throw new Error('crop requires non-uniform scaling');
    const framing = [sx, 0, -sx*x0, 0, sy, -sy*y0, 0, 0, 1];
    return {
      matrix: multiply(framing, homography),
      homography,
      crop: selectedCrop,
      canonicalCrop,
      transformedCorners,
      thetaDeg: rotationAngleDeg(rotation),
      projectiveWRatio,
    };
  }

  function makeState(rotation, intrinsics, imageSize, crop) {
    const view = computeView(rotation, intrinsics, imageSize, crop);
    return {
      version: STATE_VERSION,
      rotation: rotation.slice(),
      intrinsics: { ...intrinsics },
      crop: view.crop.slice(),
      crop_policy: CROP_POLICY,
    };
  }

  function toCssMatrix3d(matrix) {
    return `matrix3d(${[
      matrix[0], matrix[3], 0, matrix[6],
      matrix[1], matrix[4], 0, matrix[7],
      0, 0, 1, 0,
      matrix[2], matrix[5], 0, matrix[8],
    ].join(',')})`;
  }

  const api = {
    STATE_VERSION,
    CROP_POLICY,
    apply,
    cameraRotationForDrag,
    composeRotation,
    computeView,
    makeState,
    multiply,
    rotationAngleDeg,
    rotationFromVector,
    rotationHomography,
    toCssMatrix3d,
  };
  root.PerspectiveGeometry = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
