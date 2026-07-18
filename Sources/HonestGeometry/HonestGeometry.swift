// Swift implementation of the shared physical camera model. The Python
// implementation (geometry.py) is the reference; webapp/geometry.js and this
// file must stay bit-compatible with it within the tolerances pinned by
// tests/HonestGeometryTests/Fixtures/geometry_contract.json.

import Foundation
import simd

public enum GeometryError: Error, CustomStringConvertible, Equatable {
    case invalidInput(String)
    case degenerate(String)

    public var description: String {
        switch self {
        case .invalidInput(let message), .degenerate(let message):
            return message
        }
    }
}

public let stateVersion = 1
public let cropPolicy = "conservative_same_aspect_v1"
public let fullFrameDiagonalMM = hypot(36.0, 24.0)

public typealias Matrix3 = simd_double3x3

// MARK: - Intrinsics

/// Extensible enum: unknown future sources (new estimators on the Python
/// side) must decode and round-trip losslessly instead of failing the whole
/// correction state, so this is a raw-value struct rather than a Swift enum.
public struct IntrinsicsSource: RawRepresentable, Codable, Sendable, Equatable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public static let exif35mm = IntrinsicsSource(rawValue: "exif_35mm")
    public static let fallbackMaxDimension = IntrinsicsSource(rawValue: "fallback_max_dimension")
}

public struct Intrinsics: Codable, Sendable, Equatable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double
    /// Where the focal length came from; nil when the state predates it.
    public var source: IntrinsicsSource?
    public var focal35mm: Double?

    enum CodingKeys: String, CodingKey {
        case fx, fy, cx, cy, source
        case focal35mm = "focal_35mm"
    }

    public init(
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        source: IntrinsicsSource? = nil,
        focal35mm: Double? = nil
    ) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.source = source
        self.focal35mm = focal35mm
    }
}

/// Centered square-pixel intrinsics, preferring the EXIF 35mm focal length.
public func makeIntrinsics(width: Double, height: Double, focal35mm: Double? = nil) -> Intrinsics {
    let focalPx: Double
    let source: IntrinsicsSource
    if let focal = focal35mm, focal.isFinite, focal > 0 {
        focalPx = focal * hypot(width, height) / fullFrameDiagonalMM
        source = .exif35mm
    } else {
        focalPx = Swift.max(width, height)
        source = .fallbackMaxDimension
    }
    return Intrinsics(
        fx: focalPx,
        fy: focalPx,
        cx: width / 2.0,
        cy: height / 2.0,
        source: source,
        focal35mm: focal35mm
    )
}

public func intrinsicsMatrix(_ intrinsics: Intrinsics) -> Matrix3 {
    Matrix3(rows: [
        SIMD3(intrinsics.fx, 0.0, intrinsics.cx),
        SIMD3(0.0, intrinsics.fy, intrinsics.cy),
        SIMD3(0.0, 0.0, 1.0),
    ])
}

// MARK: - Matrix helpers

public func matrix3(rowMajor values: [Double]) throws -> Matrix3 {
    guard values.count == 9, values.allSatisfy(\.isFinite) else {
        throw GeometryError.invalidInput("matrix needs 9 finite values")
    }
    return Matrix3(rows: [
        SIMD3(values[0], values[1], values[2]),
        SIMD3(values[3], values[4], values[5]),
        SIMD3(values[6], values[7], values[8]),
    ])
}

public func rowMajor(_ matrix: Matrix3) -> [Double] {
    // simd matrices subscript as matrix[column][row].
    [
        matrix[0][0], matrix[1][0], matrix[2][0],
        matrix[0][1], matrix[1][1], matrix[2][1],
        matrix[0][2], matrix[1][2], matrix[2][2],
    ]
}

private func trace(_ matrix: Matrix3) -> Double {
    matrix[0][0] + matrix[1][1] + matrix[2][2]
}

private func isFinite(_ matrix: Matrix3) -> Bool {
    rowMajor(matrix).allSatisfy(\.isFinite)
}

private func maxAbsDifference(_ a: Matrix3, _ b: Matrix3) -> Double {
    zip(rowMajor(a), rowMajor(b)).map { abs($0 - $1) }.max() ?? 0
}

private func frobeniusNorm(_ matrix: Matrix3) -> Double {
    rowMajor(matrix).map { $0 * $0 }.reduce(0, +).squareRoot()
}

// MARK: - Rotations

/// Nearest rotation matrix (polar factor). The Python reference uses SVD and
/// flips a sign for det < 0 inputs; those never pass `validateCorrectionState`,
/// so here anything not clearly on the det > 0 side is an error.
public func projectToRotation(_ matrix: Matrix3) throws -> Matrix3 {
    guard isFinite(matrix) else {
        throw GeometryError.invalidInput("rotation contains non-finite values")
    }
    guard matrix.determinant > 1e-12 else {
        throw GeometryError.invalidInput("matrix is not near a rotation")
    }
    var current = matrix
    for _ in 0..<64 {
        let next = 0.5 * (current + current.inverse.transpose)
        let delta = maxAbsDifference(next, current)
        current = next
        if delta < 1e-15 { break }
    }
    return current
}

public func rotationAngleDeg(_ rotation: Matrix3) -> Double {
    let cosine = Swift.max(-1.0, Swift.min(1.0, (trace(rotation) - 1.0) / 2.0))
    return acos(cosine) * 180.0 / .pi
}

/// Rodrigues formula: axis-angle vector -> rotation matrix.
public func rotationFromVector(_ vector: SIMD3<Double>) throws -> Matrix3 {
    guard vector.x.isFinite, vector.y.isFinite, vector.z.isFinite else {
        throw GeometryError.invalidInput("rotation vector contains non-finite values")
    }
    let theta = (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).squareRoot()
    if theta < 1e-12 {
        return matrix_identity_double3x3
    }
    let x = vector.x / theta, y = vector.y / theta, z = vector.z / theta
    let c = cos(theta), s = sin(theta), t = 1.0 - c
    return Matrix3(rows: [
        SIMD3(t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        SIMD3(t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        SIMD3(t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    ])
}

/// Log map: rotation matrix -> axis-angle vector.
public func rotationVector(from matrix: Matrix3) throws -> SIMD3<Double> {
    let rotation = try projectToRotation(matrix)
    let cosine = Swift.max(-1.0, Swift.min(1.0, (trace(rotation) - 1.0) / 2.0))
    let theta = acos(cosine)
    if theta < 1e-12 {
        return .zero
    }
    // Row-major R[i][j] is rotation[j][i] in simd's column-major subscripting.
    if theta < .pi - 1e-6 {
        let axis = SIMD3(
            rotation[1][2] - rotation[2][1],
            rotation[2][0] - rotation[0][2],
            rotation[0][1] - rotation[1][0]
        ) / (2.0 * sin(theta))
        return axis * theta
    }
    // theta ~ pi: off-diagonal differences vanish, recover the axis from the
    // diagonal and fix signs from off-diagonal sums.
    var axis = SIMD3(
        Swift.max(0.0, (rotation[0][0] + 1.0) / 2.0).squareRoot(),
        Swift.max(0.0, (rotation[1][1] + 1.0) / 2.0).squareRoot(),
        Swift.max(0.0, (rotation[2][2] + 1.0) / 2.0).squareRoot()
    )
    let xy = rotation[1][0] + rotation[0][1]
    let xz = rotation[2][0] + rotation[0][2]
    let yz = rotation[2][1] + rotation[1][2]
    if axis.x >= axis.y && axis.x >= axis.z {
        axis.y = copysign(axis.y, xy)
        axis.z = copysign(axis.z, xz)
    } else if axis.y >= axis.z {
        axis.x = copysign(axis.x, xy)
        axis.z = copysign(axis.z, yz)
    } else {
        axis.x = copysign(axis.x, xz)
        axis.y = copysign(axis.y, yz)
    }
    return simd_normalize(axis) * theta
}

/// Interpolate identity -> rotation on SO(3), never in homography space.
public func interpolateRotation(_ rotation: Matrix3, strength: Double) throws -> Matrix3 {
    guard strength.isFinite else {
        throw GeometryError.invalidInput("strength must be finite")
    }
    let vector = try rotationVector(from: rotation)
    return try rotationFromVector(vector * strength)
}

public func composeRotation(_ vector: SIMD3<Double>, _ rotation: Matrix3) throws -> Matrix3 {
    try projectToRotation(rotationFromVector(vector) * rotation)
}

public func rotationFromHomography(_ homography: Matrix3, intrinsics: Intrinsics) throws -> Matrix3 {
    let k = intrinsicsMatrix(intrinsics)
    let candidate = k.inverse * homography * k
    let scale = cbrt(abs(candidate.determinant))
    guard scale.isFinite, scale >= 1e-12 else {
        throw GeometryError.degenerate("homography is singular")
    }
    return try projectToRotation((1.0 / scale) * candidate)
}

/// One-finger drag: horizontal displacement is pure yaw, vertical is pure
/// pitch, from the pose at pointer-down. No roll, position-independent.
public func cameraRotationForDrag(
    initialRotation: Matrix3,
    displacement: SIMD2<Double>,
    intrinsics: Intrinsics
) throws -> Matrix3 {
    guard displacement.x.isFinite, displacement.y.isFinite,
          intrinsics.fx.isFinite, intrinsics.fy.isFinite,
          intrinsics.fx > 0, intrinsics.fy > 0 else {
        throw GeometryError.invalidInput("drag and focal lengths must be finite")
    }
    let pitch = -atan(displacement.y / intrinsics.fy)
    let yaw = atan(displacement.x / intrinsics.fx)
    return try composeRotation(SIMD3(pitch, yaw, 0.0), initialRotation)
}

// MARK: - Crop

public struct Crop: Sendable, Equatable {
    public var x0: Double
    public var y0: Double
    public var x1: Double
    public var y1: Double

    public init(x0: Double, y0: Double, x1: Double, y1: Double) {
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    }

    public init(array: [Double]) throws {
        guard array.count == 4, array.allSatisfy(\.isFinite) else {
            throw GeometryError.invalidInput("crop must contain four finite values")
        }
        self.init(x0: array[0], y0: array[1], x1: array[2], y1: array[3])
    }

    public var asArray: [Double] { [x0, y0, x1, y1] }
    public var width: Double { x1 - x0 }
    public var height: Double { y1 - y0 }
}

public func fitAspect(crop: Crop, targetAspect: Double) throws -> Crop {
    var result = crop
    guard result.width > 0, result.height > 0 else {
        throw GeometryError.invalidInput("crop is empty")
    }
    if result.width / result.height > targetAspect {
        let newWidth = result.height * targetAspect
        let center = (result.x0 + result.x1) / 2.0
        result.x0 = center - newWidth / 2.0
        result.x1 = center + newWidth / 2.0
    } else {
        let newHeight = result.width / targetAspect
        let center = (result.y0 + result.y1) / 2.0
        result.y0 = center - newHeight / 2.0
        result.y1 = center + newHeight / 2.0
    }
    return result
}

public func conservativeCrop(corners: [SIMD2<Double>], targetAspect: Double) throws -> Crop {
    guard corners.count == 4 else {
        throw GeometryError.invalidInput("expected four corners")
    }
    let (tl, tr, br, bl) = (corners[0], corners[1], corners[2], corners[3])
    return try fitAspect(
        crop: Crop(
            x0: Swift.max(tl.x, bl.x),
            y0: Swift.max(tl.y, tr.y),
            x1: Swift.min(tr.x, br.x),
            y1: Swift.min(br.y, bl.y)
        ),
        targetAspect: targetAspect
    )
}

/// All four crop corners inside (or on the edge of) the convex source quad.
func cropInsideQuad(_ crop: Crop, quad: [SIMD2<Double>], tolerance: Double = 1e-9) -> Bool {
    let corners = [
        SIMD2(crop.x0, crop.y0), SIMD2(crop.x1, crop.y0),
        SIMD2(crop.x1, crop.y1), SIMD2(crop.x0, crop.y1),
    ]
    for point in corners {
        var hasPositive = false
        var hasNegative = false
        for index in 0..<4 {
            let a = quad[index]
            let b = quad[(index + 1) % 4]
            let cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
            let scale = Swift.max(1.0, abs(b.x - a.x), abs(b.y - a.y), abs(point.x - a.x), abs(point.y - a.y))
            if cross > tolerance * scale {
                hasPositive = true
            } else if cross < -tolerance * scale {
                hasNegative = true
            }
        }
        if hasPositive && hasNegative { return false }
    }
    return true
}

// MARK: - View

/// Apply a homography to a point (JS `apply`, Python `_apply_homography`).
public func applyHomography(_ matrix: Matrix3, to point: SIMD2<Double>) throws -> SIMD2<Double> {
    let mapped = matrix * SIMD3(point.x, point.y, 1.0)
    guard abs(mapped.z) >= 1e-9 else {
        throw GeometryError.degenerate("projection crosses infinity")
    }
    return SIMD2(mapped.x / mapped.z, mapped.y / mapped.z)
}

public struct ViewResult: Sendable {
    /// Fixed-size warp: framing * homography.
    public let matrix: Matrix3
    /// K * R * K^-1.
    public let homography: Matrix3
    public let crop: Crop
    public let canonicalCrop: Crop
    public let transformedCorners: [SIMD2<Double>]
    public let thetaDeg: Double
    public let projectiveWRatio: Double
}

/// Build a fixed-size warp and a crop containing only source pixels.
public func computeView(
    rotation: Matrix3,
    intrinsics: Intrinsics,
    imageSize: SIMD2<Double>,
    crop: Crop? = nil
) throws -> ViewResult {
    let width = imageSize.x
    let height = imageSize.y
    guard width > 1, height > 1 else {
        throw GeometryError.invalidInput("image_size must be positive")
    }
    let projected = try projectToRotation(rotation)
    let k = intrinsicsMatrix(intrinsics)
    let homography = k * projected * k.inverse

    let sourceCorners: [SIMD2<Double>] = [
        SIMD2(0.0, 0.0), SIMD2(width, 0.0), SIMD2(width, height), SIMD2(0.0, height),
    ]
    var denominators: [Double] = []
    var transformed: [SIMD2<Double>] = []
    for corner in sourceCorners {
        let mapped = homography * SIMD3(corner.x, corner.y, 1.0)
        guard abs(mapped.z) >= 1e-9 else {
            throw GeometryError.degenerate("projection crosses infinity")
        }
        denominators.append(mapped.z)
        transformed.append(SIMD2(mapped.x / mapped.z, mapped.y / mapped.z))
    }
    guard denominators.allSatisfy({ $0 > 0 }) || denominators.allSatisfy({ $0 < 0 }) else {
        throw GeometryError.degenerate("projection flips across the camera plane")
    }
    let absW = denominators.map(abs)
    let wRatio = absW.min()! / absW.max()!

    let aspect = width / height
    let canonicalCrop = try conservativeCrop(corners: transformed, targetAspect: aspect)
    let selectedCrop: Crop
    if let crop {
        let fitted = try fitAspect(crop: crop, targetAspect: aspect)
        guard cropInsideQuad(fitted, quad: transformed) else {
            throw GeometryError.invalidInput("crop extends outside transformed source pixels")
        }
        selectedCrop = fitted
    } else {
        selectedCrop = canonicalCrop
    }

    let sx = width / selectedCrop.width
    let sy = height / selectedCrop.height
    guard abs(sx - sy) <= 1e-6 * Swift.max(1.0, sx, sy) else {
        throw GeometryError.invalidInput("crop would require non-uniform scaling")
    }
    let framing = Matrix3(rows: [
        SIMD3(sx, 0.0, -sx * selectedCrop.x0),
        SIMD3(0.0, sy, -sy * selectedCrop.y0),
        SIMD3(0.0, 0.0, 1.0),
    ])
    return ViewResult(
        matrix: framing * homography,
        homography: homography,
        crop: selectedCrop,
        canonicalCrop: canonicalCrop,
        transformedCorners: transformed,
        thetaDeg: rotationAngleDeg(projected),
        projectiveWRatio: wRatio
    )
}

// MARK: - Correction state

public struct CorrectionState: Codable, Sendable, Equatable {
    public var version: Int
    /// Row-major 3x3 rotation.
    public var rotation: [Double]
    public var intrinsics: Intrinsics
    /// [x0, y0, x1, y1] in source pixels after rotation.
    public var crop: [Double]
    public var cropPolicy: String

    enum CodingKeys: String, CodingKey {
        case version, rotation, intrinsics, crop
        case cropPolicy = "crop_policy"
    }

    public init(version: Int, rotation: [Double], intrinsics: Intrinsics, crop: [Double], cropPolicy: String) {
        self.version = version
        self.rotation = rotation
        self.intrinsics = intrinsics
        self.crop = crop
        self.cropPolicy = cropPolicy
    }
}

public func makeCorrectionState(
    rotation: Matrix3,
    intrinsics: Intrinsics,
    imageSize: SIMD2<Double>,
    crop: Crop? = nil
) throws -> CorrectionState {
    let projected = try projectToRotation(rotation)
    let view = try computeView(rotation: projected, intrinsics: intrinsics, imageSize: imageSize, crop: crop)
    return CorrectionState(
        version: stateVersion,
        rotation: rowMajor(projected),
        intrinsics: intrinsics,
        crop: view.crop.asArray,
        cropPolicy: cropPolicy
    )
}

public struct Validation: Sendable {
    public let accepted: Bool
    public let reason: String?
    public let thetaDeg: Double?
    public let orthogonalityError: Double?
    public let determinant: Double?
    public let projectiveWRatio: Double?
    public let view: ViewResult?
}

public func validateCorrectionState(
    _ state: CorrectionState,
    imageSize: SIMD2<Double>,
    maxRotationDeg: Double = 45.0,
    minProjectiveWRatio: Double = 0.22
) -> Validation {
    do {
        guard state.version == stateVersion else {
            throw GeometryError.invalidInput("unsupported correction state version")
        }
        let input = try matrix3(rowMajor: state.rotation)
        let orthogonalityError = frobeniusNorm(input.transpose * input - matrix_identity_double3x3)
        let determinant = input.determinant
        guard orthogonalityError <= 1e-5, abs(determinant - 1.0) <= 1e-5 else {
            throw GeometryError.invalidInput("rotation is not in SO(3)")
        }
        let rotation = try projectToRotation(input)
        let thetaDeg = rotationAngleDeg(rotation)
        guard thetaDeg <= maxRotationDeg else {
            throw GeometryError.invalidInput("rotation exceeds the safety envelope")
        }
        let view = try computeView(
            rotation: rotation,
            intrinsics: state.intrinsics,
            imageSize: imageSize,
            crop: Crop(array: state.crop)
        )
        guard view.projectiveWRatio >= minProjectiveWRatio else {
            throw GeometryError.degenerate("projection is too close to degeneracy")
        }
        return Validation(
            accepted: true,
            reason: nil,
            thetaDeg: thetaDeg,
            orthogonalityError: orthogonalityError,
            determinant: determinant,
            projectiveWRatio: view.projectiveWRatio,
            view: view
        )
    } catch {
        return Validation(
            accepted: false,
            reason: "\(error)",
            thetaDeg: nil,
            orthogonalityError: nil,
            determinant: nil,
            projectiveWRatio: nil,
            view: nil
        )
    }
}
