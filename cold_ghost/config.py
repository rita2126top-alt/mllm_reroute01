"""Read the release experiment matrix without changing any release YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("avg192", "avg128", "avg64")
MODELS = ("llava15", "qwen25vl")
VARIANTS = ("fastv_K3", "pdrop_earlyL2", "ours_vs_fastv", "ours_vs_pdrop",
            "ours_vs_fastv_stagewise", "ours_vs_pdrop_stagewise")
GC_VARIANTS = VARIANTS[2:]
ABLATIONS = ("full", "self", "context", "score", "no_fresh")
GROUNDING = ("refcoco_bbox_rec_val", "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB",
             "refcoco+_bbox_rec_val", "refcoco+_bbox_rec_testA", "refcoco+_bbox_rec_testB",
             "refcocog_bbox_rec_val", "refcocog_bbox_rec_test")
TASK_GROUPS = {
    "pope": ("pope",), "vqa": ("gqa", "mmbench_en_dev", "mme"),
    "refcoco": GROUNDING[:3], "grounding": GROUNDING,
    "paper_main": ("pope",) + GROUNDING,
    "ablation": ("gqa", "mmbench_en_dev", GROUNDING[1], GROUNDING[2]),
    "all": ("pope", "gqa", "mmbench_en_dev", "mme") + GROUNDING,
}
TASK_ALIASES = {"mmbench": "mmbench_en_dev", **{
    name.replace("_bbox_rec", ""): name for name in GROUNDING}}


def _yaml(path: Path) -> dict:
    import yaml
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


@dataclass(frozen=True)
class ExperimentSpec:
    config_name: str
    path: Path
    model_name: str
    model_family: str
    model: dict
    routing: dict
    eval: dict

    @property
    def checkpoint_key(self) -> str:
        if not self.is_reroute:
            raise ValueError("Only recoverable reroute experiments have Ghost checkpoints")
        model, tier, filename = self.config_name.split("/")
        variant = filename[len(model) + 1:-(len(tier) + 1)].replace("_stagewise", "")
        return f"{model}__{variant}__{tier}"

    @property
    def is_reroute(self) -> bool:
        return (self.routing.get("method") == "pdrop"
                and self.routing.get("action") in ("compact_route", "compact_route_stagewise")
                and self.routing.get("monotonic") is False)

    def as_omegaconf(self):
        from omegaconf import OmegaConf
        return OmegaConf.create({"model": self.model, "routing": self.routing, "eval": self.eval})

    def checkpoint_metadata(self) -> dict:
        return {"backbone": self.model["pretrained"], "model_family": self.model_family,
                "num_layers": self.model["num_layers"],
                "drop_layers": list(self.routing.get("drop_layers", [])),
                "keep_ratios": list(self.routing.get("keep_ratios", [])),
                "checkpoint_key": self.checkpoint_key}


def load_experiment(config: str | Path) -> ExperimentSpec:
    """Resolve an ORIGINAL release config; external/modified configs are rejected."""
    base = (ROOT / "configs" / "experiment").resolve()
    raw = str(config).replace("\\", "/")
    candidate = Path(config)
    if not candidate.is_absolute():
        if raw.startswith("configs/experiment/"):
            raw = raw[len("configs/experiment/"):]
        elif raw.startswith("experiment/"):
            raw = raw[len("experiment/"):]
        candidate = base / raw
    if candidate.suffix != ".yaml":
        candidate = candidate.with_suffix(".yaml")
    candidate = candidate.resolve()
    try:
        name = candidate.relative_to(base).with_suffix("").as_posix()
    except ValueError as exc:
        raise ValueError("Config must be inside the original configs/experiment directory") from exc
    if name not in original_config_names():
        raise ValueError(f"Not one of the 38 release configurations: {name}")
    raw_cfg = _yaml(candidate)
    defaults = {key: value for item in raw_cfg.get("defaults", [])
                if isinstance(item, dict) for key, value in item.items()}
    model_name = defaults["/model"]
    model = _yaml(ROOT / "configs" / "model" / f"{model_name}.yaml")
    eval_cfg = _yaml(ROOT / "configs" / "eval" / f"{defaults.get('/eval', 'pope')}.yaml")
    eval_cfg.update(raw_cfg.get("eval", {}))
    family = "llava" if model_name == "llava15_7b_hf" else "qwen25vl"
    return ExperimentSpec(name, candidate, model_name, family, model, raw_cfg["routing"], eval_cfg)


def original_config_names(models=MODELS, tiers=TIERS) -> list[str]:
    return [name for model in models for name in
            [f"baseline/{model}"] + [f"{model}/{tier}/{model}_{variant}_{tier}"
             for tier in tiers for variant in VARIANTS]]


def gc_config_names(models=MODELS, tiers=TIERS) -> list[str]:
    return [f"{model}/{tier}/{model}_{variant}_{tier}"
            for model in models for tier in tiers for variant in GC_VARIANTS]


def resolve_tasks(selection: str) -> list[str]:
    tasks = []
    for item in selection.split(","):
        item = item.strip()
        if not item:
            raise ValueError("Empty task in selection")
        for task in TASK_GROUPS.get(item, (TASK_ALIASES.get(item, item),)):
            if not re.fullmatch(r"[A-Za-z0-9_+.-]+", task):
                raise ValueError(f"Invalid task name: {task}")
            if task not in tasks:
                tasks.append(task)
    return tasks
