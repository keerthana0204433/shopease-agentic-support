
import json
import unittest
from pathlib import Path


class GoldenDatasetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        dataset_path = project_root / "tests" / "golden_dataset.json"

        with dataset_path.open("r", encoding="utf-8") as file:
            cls.cases = json.load(file)

    def test_dataset_has_at_least_10_cases(self):
        self.assertGreaterEqual(len(self.cases), 10)

    def test_every_case_has_required_fields(self):
        required_fields = {
            "id",
            "category",
            "message",
            "expected_behavior",
        }

        for case in self.cases:
            with self.subTest(case=case.get("id")):
                self.assertTrue(required_fields.issubset(case.keys()))

    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_messages_are_not_empty(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["message"].strip())

    def test_expected_behaviors_are_not_empty(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["expected_behavior"].strip())


if __name__ == "__main__":
    unittest.main()
