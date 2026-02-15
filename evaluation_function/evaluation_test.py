import os
import unittest
from pathlib import Path

from lf_toolkit.evaluation import Params
from .evaluation import evaluation_function


def _as_file_uri(path: str) -> str:
    """Return a proper file:// URI (works on Windows/macOS/Linux)."""
    return Path(os.path.abspath(path)).as_uri()


def _feedback_as_text(feedback):
    """Normalize feedback to plain text for assertions."""
    if isinstance(feedback, list):
        # list of tuples or list of strings -> flatten
        parts = []
        for item in feedback:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                parts.append(f"{item[0]}: {item[1]}")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(feedback)


class TestEvaluationFunction(unittest.TestCase):
    """
    Tests for evaluation_function().

    - test_evaluation_with_local_file_url:
        Simulates Lambda Feedback image upload format using file:// URL.
    - test_evaluation_missing_image:
        Empty response should not crash, should return a valid Result dict.
    - test_evaluation_bad_url:
        Invalid URL should not crash, should return a valid Result dict with error-like feedback.
    """

    def test_evaluation_with_local_file_url(self):
        here = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(here, "evaluation_test.jpg")

        if not os.path.exists(img_path):
            self.fail(
                "Missing test image: evaluation_function/evaluation_test.jpg\n"
                "Please add a small jpg file with that exact name."
            )

        response = [{
            "comment": "",
            "name": "evaluation_test.jpg",
            "size": os.path.getsize(img_path),
            "type": "image/jpeg",
            "url": _as_file_uri(img_path),
        }]

        answer = {}
        params = Params(return_images=False, debug=True, show_target=False)

        result = evaluation_function(response, answer, params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        self.assertIn("feedback", result)

        fb = result["feedback"]
        self.assertTrue(isinstance(fb, (list, str)))

        fb_text = _feedback_as_text(fb)
        # For a valid image, it should NOT be a load fail message
        self.assertNotIn("LOAD_FAIL", fb_text)

    def test_evaluation_missing_image(self):
        response = []
        answer = {}
        params = Params(return_images=False, debug=True)

        result = evaluation_function(response, answer, params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        self.assertIn("feedback", result)
        self.assertFalse(result["is_correct"])

    def test_evaluation_bad_url(self):
        response = [{
            "name": "nope.jpg",
            "size": 0,
            "type": "image/jpeg",
            "url": "file:///THIS/PATH/DOES/NOT/EXIST.jpg",
        }]
        answer = {}
        params = Params(return_images=False, debug=True)

        result = evaluation_function(response, answer, params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        self.assertIn("feedback", result)

        fb = result["feedback"]
        self.assertTrue(isinstance(fb, (list, str)))

        fb_text = _feedback_as_text(fb)
        # Here we EXPECT an error-like message
        self.assertIn("LOAD_FAIL", fb_text)


if __name__ == "__main__":
    unittest.main()
