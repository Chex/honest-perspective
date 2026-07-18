// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HonestGeometry",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
    ],
    products: [
        .library(name: "HonestGeometry", targets: ["HonestGeometry"]),
    ],
    targets: [
        .target(
            name: "HonestGeometry",
            path: "Sources/HonestGeometry"
        ),
        .testTarget(
            name: "HonestGeometryTests",
            dependencies: ["HonestGeometry"],
            path: "tests/HonestGeometryTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
