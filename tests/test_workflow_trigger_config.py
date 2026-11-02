import unittest
from pathlib import Path


class WorkflowTriggerConfigTests(unittest.TestCase):
    def test_robust_research_has_one_push_trigger_for_both_branches(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "robust-research.yml"
        ).read_text(encoding="utf-8")

        self.assertEqual(workflow.count("\n  push:\n"), 1)
        self.assertIn(
            'branches: [main, "research/robust-backtest-audit"]',
            workflow,
        )

    def test_robust_research_push_filter_includes_workflow_and_code(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "robust-research.yml"
        ).read_text(encoding="utf-8")

        push_section = workflow.split("\n  push:\n", 1)[1].split(
            "\n  workflow_dispatch:", 1
        )[0]
        self.assertIn('"robust_research.py"', push_section)
        self.assertIn('"tests/test_robust_research.py"', push_section)
        self.assertIn('".github/workflows/robust-research.yml"', push_section)


if __name__ == "__main__":
    unittest.main()
