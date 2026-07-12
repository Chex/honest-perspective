import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WebRuntimeTests(unittest.TestCase):
    def test_static_host_and_missing_backend_detection(self):
        script = r"""
const runtime = require('./webapp/runtime.js');
const result = {
  githubPages: runtime.isStaticHostingLocation({
    protocol: 'https:', hostname: 'chex.github.io',
  }),
  fileUrl: runtime.isStaticHostingLocation({
    protocol: 'file:', hostname: '',
  }),
  localBackend: runtime.isStaticHostingLocation({
    protocol: 'http:', hostname: '127.0.0.1',
  }),
  pages404: runtime.responseIndicatesMissingBackend(404, 'text/html'),
  staticHtml200: runtime.responseIndicatesMissingBackend(200, 'text/html'),
  staticHtml403: runtime.responseIndicatesMissingBackend(403, 'text/html'),
  backendSuccess: runtime.responseIndicatesMissingBackend(200, 'application/json'),
  backendError: runtime.responseIndicatesMissingBackend(500, 'application/json; charset=utf-8'),
  problemJson: runtime.responseIndicatesMissingBackend(422, 'application/problem+json'),
};
process.stdout.write(JSON.stringify(result));
"""
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["githubPages"])
        self.assertTrue(result["fileUrl"])
        self.assertFalse(result["localBackend"])
        self.assertTrue(result["pages404"])
        self.assertTrue(result["staticHtml200"])
        self.assertTrue(result["staticHtml403"])
        self.assertFalse(result["backendSuccess"])
        self.assertFalse(result["backendError"])
        self.assertFalse(result["problemJson"])


if __name__ == "__main__":
    unittest.main()
