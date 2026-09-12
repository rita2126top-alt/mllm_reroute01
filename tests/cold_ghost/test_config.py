import unittest
from cold_ghost.config import ROOT, load_experiment, original_config_names, gc_config_names, resolve_tasks

class ReleaseConfigurationTests(unittest.TestCase):
    def test_complete_release_and_gc_matrix(self):
        original = original_config_names()
        self.assertEqual(len(original), 38)
        self.assertEqual(len(set(original)), 38)
        self.assertEqual(len(gc_config_names()), 24)
        self.assertEqual(len({load_experiment(name).checkpoint_key for name in gc_config_names()}), 12)
        for name in original:
            self.assertEqual(load_experiment(name).config_name, name)

    def test_exact_reroute_schedules_and_rounding(self):
        reference = {"avg192": ([152] * 4, [259, 259, 129, 64]),
                     "avg128": ([81] * 4, [152, 152, 76, 38]),
                     "avg64": ([11] * 4, [80, 80, 40, 20])}
        for model, middle in (("llava15", [7, 15, 23]), ("qwen25vl", [6, 13, 20])):
            for tier, keeps in reference.items():
                for index, variant in enumerate(("ours_vs_fastv", "ours_vs_pdrop")):
                    spec = load_experiment(f"{model}/{tier}/{model}_{variant}_{tier}")
                    self.assertEqual(spec.routing["drop_layers"], [3 if index == 0 else 2] + middle)
                    self.assertEqual([int(576 * ratio) for ratio in spec.routing["keep_ratios"]], keeps[index])
                    self.assertFalse(spec.routing["monotonic"])

    def test_tasks_preserve_paper_main_and_vqa_distinction(self):
        self.assertEqual(len(resolve_tasks("all")), 12)
        self.assertEqual(len(resolve_tasks("paper_main")), 9)
        self.assertEqual(set(resolve_tasks("paper_main,vqa")), set(resolve_tasks("all")))
        self.assertEqual(resolve_tasks("ablation"), ["gqa", "mmbench_en_dev",
                         "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB"])
        self.assertEqual(resolve_tasks("gqa,gqa,mmbench"), ["gqa", "mmbench_en_dev"])

    def test_only_original_config_directory_allowed(self):
        with self.assertRaises(ValueError):
            load_experiment("../model/qwen25vl_7b")
        with self.assertRaises(ValueError):
            load_experiment("invented/config")

    def test_config_spellings_are_equivalent(self):
        name = "llava15/avg192/llava15_ours_vs_fastv_avg192"
        for spelling in (name, "experiment/" + name, "configs/experiment/" + name + ".yaml",
                         ROOT / "configs" / "experiment" / (name + ".yaml")):
            self.assertEqual(load_experiment(spelling).config_name, name)

if __name__ == "__main__":
    unittest.main()
