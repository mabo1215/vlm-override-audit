import unittest

from .generate_owlv2_candidates import _normalized_image_name, _text_max_length


class ImageNameNormalizationTests(unittest.TestCase):
    def test_removes_any_expression_numeric_suffix(self):
        self.assertEqual(
            _normalized_image_name("COCO_train2014_000000000110_8.jpg"),
            "COCO_train2014_000000000110.jpg",
        )
        self.assertEqual(
            _normalized_image_name("COCO_train2014_000000000110_0.jpg"),
            "COCO_train2014_000000000110.jpg",
        )

    def test_preserves_names_without_expression_suffix(self):
        self.assertEqual(
            _normalized_image_name("COCO_train2014_000000000110.jpg"),
            "COCO_train2014_000000000110.jpg",
        )


class TextLimitTests(unittest.TestCase):
    def test_reads_owlv2_text_context_limit(self):
        class TextConfig:
            max_position_embeddings = 16

        class Config:
            text_config = TextConfig()

        class Model:
            config = Config()

        self.assertEqual(_text_max_length(Model()), 16)

    def test_missing_text_context_limit_is_explicit(self):
        class Model:
            config = object()

        self.assertIsNone(_text_max_length(Model()))


if __name__ == "__main__":
    unittest.main()
