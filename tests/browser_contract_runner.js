// Run the browser geometry implementation against the shared contract
// fixtures and print its results as JSON for tests/test_geometry.py.
'use strict';

const fs = require('fs');
const path = require('path');
const geometry = require(path.join(__dirname, '..', 'webapp', 'geometry.js'));

const fixtures = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

const results = {
  computeView: fixtures.computeView.map(testCase => {
    const view = geometry.computeView(
      testCase.rotation, testCase.intrinsics, testCase.imageSize, testCase.crop || undefined
    );
    return {
      matrix: view.matrix,
      homography: view.homography,
      crop: view.crop,
      canonicalCrop: view.canonicalCrop,
      transformedCorners: view.transformedCorners,
      thetaDeg: view.thetaDeg,
      projectiveWRatio: view.projectiveWRatio,
    };
  }),
  drag: fixtures.drag.map(testCase => geometry.cameraRotationForDrag(
    testCase.startRotation, testCase.displacement, testCase.intrinsics
  )),
};

process.stdout.write(JSON.stringify(results));
