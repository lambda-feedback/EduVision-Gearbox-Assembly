import os
import unittest
from pathlib import Path

import cv2
import numpy as np
from lf_toolkit.evaluation import Params
from .evaluation import evaluation_function
from .yolo_pipeline import compute_image_quality_metrics


def _as_file_uri(path: str) -> str:
    """Return a proper file:// URI (works on Windows/macOS/Linux)."""
    return Path(os.path.abspath(path)).as_uri()


def _feedback_as_text(feedback):
    """Normalize feedback to plain text for assertions."""
    if isinstance(feedback, list):
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
        Invalid URL should not crash, should return a valid Result dict with final user-facing feedback.
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
        params = Params(return_images=False, show_target=False)

        result = evaluation_function(response, answer, params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        self.assertIn("feedback", result)

        fb = result["feedback"]
        self.assertTrue(isinstance(fb, (list, str)))

        fb_text = _feedback_as_text(fb)

        # For a valid image, it should not be a load/read failure message.
        self.assertNotIn("could not be loaded", fb_text.lower())
        self.assertNotIn("please upload at least one image", fb_text.lower())

    def test_evaluation_missing_image(self):
        response = []
        answer = {}
        params = Params(return_images=False)

        result = evaluation_function(response, answer, params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        self.assertIn("feedback", result)
        self.assertFalse(result["is_correct"])

        fb = result["feedback"]
        self.assertTrue(isinstance(fb, (list, str)))

        fb_text = _feedback_as_text(fb)
        self.assertIn("please upload at least one image", fb_text.lower())

    def test_evaluation_bad_url(self):
        response = [{
            "name": "nope.jpg",
            "size": 0,
            "type": "image/jpeg",
            "url": "file:///THIS/PATH/DOES/NOT/EXIST.jpg",
        }]
        answer = {}
        params = Params(return_images=False)

        result = evaluation_function(response, answer, params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        self.assertIn("feedback", result)
        self.assertFalse(result["is_correct"])

        fb = result["feedback"]
        self.assertTrue(isinstance(fb, (list, str)))

        fb_text = _feedback_as_text(fb)
        self.assertIn("could not be loaded", fb_text.lower())

    def test_image_quality_score_uses_student_friendly_threshold(self):
        sharp = np.zeros((120, 120, 3), dtype=np.uint8)
        sharp[:, :60] = 30
        sharp[:, 60:] = 230
        cv2.line(sharp, (0, 0), (119, 119), (255, 255, 255), 3)

        blurry = np.full((120, 120, 3), 120, dtype=np.uint8)
        blurry = cv2.GaussianBlur(blurry, (31, 31), 0)

        sharp_quality = compute_image_quality_metrics(sharp)
        blurry_quality = compute_image_quality_metrics(blurry)

        self.assertIn("quality_score", sharp_quality)
        self.assertIn("quality_accept_score", sharp_quality)
        self.assertEqual(sharp_quality["quality_score_max"], 100)
        self.assertIn("quality_advice", sharp_quality)
        self.assertGreaterEqual(
            sharp_quality["quality_score"],
            sharp_quality["quality_accept_score"],
        )
        self.assertLess(
            blurry_quality["quality_score"],
            blurry_quality["quality_accept_score"],
        )

    def test_image_quality_rejects_low_single_component_score(self):
        gradient = np.tile(np.linspace(30, 230, 160, dtype=np.uint8), (120, 1))
        image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

        quality = compute_image_quality_metrics(image)

        self.assertGreaterEqual(
            quality["quality_score"],
            quality["quality_accept_score"],
        )
        self.assertLessEqual(
            quality["sharpness_score_100"],
            quality["quality_min_component_score"],
        )
        self.assertFalse(quality["quality_pass"])


if __name__ == "__main__":
    unittest.main()
