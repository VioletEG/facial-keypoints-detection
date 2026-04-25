import unittest

import numpy as np

from facial_keypoints.data import apply_horizontal_flip_to_targets, build_horizontal_flip_mappings


class FlipUtilsTests(unittest.TestCase):
    def test_build_horizontal_flip_mappings(self) -> None:
        cols = [
            "left_eye_center_x",
            "left_eye_center_y",
            "right_eye_center_x",
            "right_eye_center_y",
            "nose_tip_x",
            "nose_tip_y",
        ]
        flip_indices, x_mask = build_horizontal_flip_mappings(cols)
        self.assertListEqual(flip_indices.tolist(), [2, 3, 0, 1, 4, 5])
        self.assertListEqual(x_mask.tolist(), [True, False, True, False, True, False])

    def test_apply_horizontal_flip_is_involution(self) -> None:
        cols = [
            "left_eye_center_x",
            "left_eye_center_y",
            "right_eye_center_x",
            "right_eye_center_y",
            "nose_tip_x",
            "nose_tip_y",
        ]
        flip_indices, x_mask = build_horizontal_flip_mappings(cols)
        targets = np.array(
            [
                [0.1, 0.2, 0.8, 0.21, 0.5, 0.6],
                [0.2, 0.3, 0.7, 0.35, 0.45, 0.5],
            ],
            dtype=np.float32,
        )
        flipped = apply_horizontal_flip_to_targets(targets, flip_indices, x_mask)
        restored = apply_horizontal_flip_to_targets(flipped, flip_indices, x_mask)
        np.testing.assert_allclose(restored, targets, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
