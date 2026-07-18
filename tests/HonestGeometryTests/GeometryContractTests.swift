import Foundation
import Testing
import simd
@testable import HonestGeometry

// MARK: - Fixture decoding

private struct Fixtures: Decodable {
    let version: Int
    let makeIntrinsics: [IntrinsicsCase]
    let computeView: [ComputeViewCase]
    let drag: [DragCase]
    let interpolate: [InterpolateCase]
}

private struct IntrinsicsCase: Decodable {
    let width: Double
    let height: Double
    let focal35mm: Double?
    let expected: ExpectedIntrinsics
}

private struct ExpectedIntrinsics: Decodable {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let source: String
    let focal35mm: Double?
}

private struct FixtureIntrinsics: Decodable {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double

    var intrinsics: Intrinsics { Intrinsics(fx: fx, fy: fy, cx: cx, cy: cy) }
}

private struct ComputeViewCase: Decodable {
    let rotation: [Double]
    let intrinsics: FixtureIntrinsics
    let imageSize: [Double]
    let crop: [Double]?
    let expected: ExpectedView
}

private struct ExpectedView: Decodable {
    let matrix: [Double]
    let homography: [Double]
    let crop: [Double]
    let canonicalCrop: [Double]
    let transformedCorners: [[Double]]
    let thetaDeg: Double
    let projectiveWRatio: Double
}

private struct DragCase: Decodable {
    let startRotation: [Double]
    let displacement: [Double]
    let intrinsics: FixtureIntrinsics
    let expectedRotation: [Double]
}

private struct InterpolateCase: Decodable {
    let rotation: [Double]
    let strength: Double
    let expectedRotation: [Double]
}

private func loadFixtures() throws -> Fixtures {
    let url = try #require(
        Bundle.module.url(
            forResource: "geometry_contract",
            withExtension: "json",
            subdirectory: "Fixtures"
        ),
        "fixture missing — run: python tests/generate_geometry_fixtures.py"
    )
    return try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: url))
}

private func expectClose(
    _ actual: [Double],
    _ expected: [Double],
    tolerance: Double = 1e-9,
    _ label: @autoclosure () -> String = "",
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(actual.count == expected.count, "\(label()): count mismatch", sourceLocation: sourceLocation)
    for (index, (value, reference)) in zip(actual, expected).enumerated() {
        let slack = tolerance * Swift.max(1.0, abs(reference))
        #expect(
            abs(value - reference) <= slack,
            "\(label())[\(index)]: \(value) vs \(reference)",
            sourceLocation: sourceLocation
        )
    }
}

// MARK: - Contract parity with the Python reference

@Suite struct GeometryContractTests {
    @Test func makeIntrinsicsMatchesReference() throws {
        for testCase in try loadFixtures().makeIntrinsics {
            let intrinsics = makeIntrinsics(
                width: testCase.width,
                height: testCase.height,
                focal35mm: testCase.focal35mm
            )
            expectClose(
                [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
                [testCase.expected.fx, testCase.expected.fy, testCase.expected.cx, testCase.expected.cy],
                "makeIntrinsics(\(testCase.width)x\(testCase.height), \(String(describing: testCase.focal35mm)))"
            )
            #expect(intrinsics.source?.rawValue == testCase.expected.source)
            #expect(intrinsics.focal35mm == testCase.expected.focal35mm)
        }
    }

    @Test func computeViewMatchesReference() throws {
        for (index, testCase) in try loadFixtures().computeView.enumerated() {
            let view = try computeView(
                rotation: try matrix3(rowMajor: testCase.rotation),
                intrinsics: testCase.intrinsics.intrinsics,
                imageSize: SIMD2(testCase.imageSize[0], testCase.imageSize[1]),
                crop: testCase.crop.map { try! Crop(array: $0) }
            )
            expectClose(rowMajor(view.matrix), testCase.expected.matrix, "case \(index) matrix")
            expectClose(rowMajor(view.homography), testCase.expected.homography, "case \(index) homography")
            expectClose(view.crop.asArray, testCase.expected.crop, "case \(index) crop")
            expectClose(view.canonicalCrop.asArray, testCase.expected.canonicalCrop, "case \(index) canonicalCrop")
            expectClose(
                view.transformedCorners.flatMap { [$0.x, $0.y] },
                testCase.expected.transformedCorners.flatMap { $0 },
                "case \(index) corners"
            )
            expectClose([view.thetaDeg], [testCase.expected.thetaDeg], "case \(index) thetaDeg")
            expectClose([view.projectiveWRatio], [testCase.expected.projectiveWRatio], "case \(index) wRatio")
        }
    }

    @Test func dragMatchesReference() throws {
        for (index, testCase) in try loadFixtures().drag.enumerated() {
            let rotation = try cameraRotationForDrag(
                initialRotation: try matrix3(rowMajor: testCase.startRotation),
                displacement: SIMD2(testCase.displacement[0], testCase.displacement[1]),
                intrinsics: testCase.intrinsics.intrinsics
            )
            expectClose(rowMajor(rotation), testCase.expectedRotation, "drag case \(index)")
        }
    }

    @Test func interpolateMatchesReference() throws {
        for (index, testCase) in try loadFixtures().interpolate.enumerated() {
            let rotation = try interpolateRotation(
                try matrix3(rowMajor: testCase.rotation),
                strength: testCase.strength
            )
            expectClose(rowMajor(rotation), testCase.expectedRotation, "interpolate case \(index)")
        }
    }
}

// MARK: - Behavior mirrored from tests/test_geometry.py

@Suite struct CorrectionStateTests {
    private let size = SIMD2(800.0, 600.0)

    @Test func strengthInterpolatesOnRotationManifold() throws {
        let rotation = try rotationFromVector(SIMD3(0.0, 30.0 * .pi / 180.0, 0.0))
        let halfway = try interpolateRotation(rotation, strength: 0.5)
        #expect(abs(rotationAngleDeg(halfway) - 15.0) < 1e-9)
        let orthogonality = rowMajor(halfway.transpose * halfway - matrix_identity_double3x3)
        #expect(orthogonality.allSatisfy { abs($0) < 1e-12 })
        #expect(abs(halfway.determinant - 1.0) < 1e-12)
    }

    @Test func stateRejectsNonRotation() throws {
        let intrinsics = makeIntrinsics(width: 800, height: 600, focal35mm: 26)
        var state = try makeCorrectionState(
            rotation: matrix_identity_double3x3,
            intrinsics: intrinsics,
            imageSize: size
        )
        state.rotation[0] = 1.1
        let validation = validateCorrectionState(state, imageSize: size)
        #expect(!validation.accepted)
        #expect(validation.reason?.contains("SO(3)") == true)
    }

    @Test func stateRejectsRotationBeyondSafetyEnvelope() throws {
        let intrinsics = makeIntrinsics(width: 800, height: 600, focal35mm: 26)
        let rotation = try rotationFromVector(SIMD3(50.0 * .pi / 180.0, 0.0, 0.0))
        let view = try computeView(rotation: rotation, intrinsics: intrinsics, imageSize: size)
        let state = CorrectionState(
            version: stateVersion,
            rotation: rowMajor(rotation),
            intrinsics: intrinsics,
            crop: view.crop.asArray,
            cropPolicy: cropPolicy
        )
        let validation = validateCorrectionState(state, imageSize: size)
        #expect(!validation.accepted)
        #expect(validation.reason?.contains("safety envelope") == true)
    }

    @Test func unknownIntrinsicsSourceSurvivesRoundTrip() throws {
        // A future Python-side estimator must not brick old app versions.
        let json = """
        {"fx": 1000.0, "fy": 1000.0, "cx": 400.0, "cy": 300.0, \
        "source": "apple_maker_note_gravity", "focal_35mm": null}
        """
        let decoded = try JSONDecoder().decode(Intrinsics.self, from: Data(json.utf8))
        #expect(decoded.source?.rawValue == "apple_maker_note_gravity")
        let reEncoded = try JSONEncoder().encode(decoded)
        #expect(String(data: reEncoded, encoding: .utf8)?.contains("apple_maker_note_gravity") == true)
    }

    @Test func validStateRoundTripsThroughJSON() throws {
        let intrinsics = makeIntrinsics(width: 800, height: 600, focal35mm: 26)
        let rotation = try rotationFromVector(SIMD3(
            8.0 * .pi / 180.0, -12.0 * .pi / 180.0, 3.0 * .pi / 180.0
        ))
        let state = try makeCorrectionState(rotation: rotation, intrinsics: intrinsics, imageSize: size)
        let encoded = try JSONEncoder().encode(state)
        let json = try #require(String(data: encoded, encoding: .utf8))
        // Same key shape as the Python state dict.
        #expect(json.contains("crop_policy"))
        #expect(json.contains("\"exif_35mm\""))
        #expect(json.contains("focal_35mm"))
        let decoded = try JSONDecoder().decode(CorrectionState.self, from: encoded)
        let validation = validateCorrectionState(decoded, imageSize: size)
        #expect(validation.accepted, "\(validation.reason ?? "")")
    }
}
