import json
from pathlib import Path

from PIL import Image
import pytest

from cold_ghost.data import (PromptDataset, build_exclusions, canonical_sha256,
                             image_content_sha256, load_manifest, prepare_gqa)


def inputs(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index in range(6):
        Image.new("RGB", (5, 4), (index * 30, 20, 40)).save(image_root / f"{index}.png")
    questions = {}
    for index in range(6):
        questions[str(100 + index)] = {"imageId": str(index), "question": "later question", "answer": "unused"}
        questions[str(index)] = {"imageId": str(index), "question": f"question {index}", "answer": "unused"}
    source = tmp_path / "train_balanced_questions.json"
    source.write_text(json.dumps(questions), encoding="utf-8")
    excluded = tmp_path / "exclusions.json"
    build_exclusions({"gqa": [str(image_root / "5.png")]}, excluded, smoke=True)
    return source, image_root, excluded


def test_manifest_determinism_answers_exclusions_and_holdout(tmp_path):
    source, image_root, excluded = inputs(tmp_path)
    path = tmp_path / "manifest.json"
    first = prepare_gqa(source, image_root, excluded, path, train_count=3, val_count=1, smoke=True)
    second = prepare_gqa(source, image_root, excluded, tmp_path / "repeat.json", train_count=3, val_count=1, smoke=True)
    assert first == second
    rows = first["train"] + first["validation"]
    assert all(row["question_id"] == row["image_id"] and row["image_id"] != "5" for row in rows)
    assert all("answer" not in row for row in rows)
    assert len({row["content_sha256"] for row in rows}) == 4
    with pytest.raises(ValueError, match="explicit --smoke"):
        load_manifest(path)
    loaded = load_manifest(path, allow_smoke=True)
    sample = PromptDataset(loaded, image_root)[0]
    assert set(sample) == {"image", "question", "image_id"}
    assert sample["question"].startswith("question ")


def test_changed_image_and_tampered_manifest_fail(tmp_path):
    source, image_root, excluded = inputs(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = prepare_gqa(source, image_root, excluded, path, train_count=2, val_count=1, smoke=True)
    Image.new("RGB", (5, 4), (0, 0, 0)).save(image_root / manifest["train"][0]["image"])
    with pytest.raises(ValueError, match="changed"):
        PromptDataset(manifest, image_root)[0]
    manifest["train"][0]["question"] = "tampered"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_manifest(path, allow_smoke=True)


def test_formal_scope_and_count_requirements(tmp_path):
    source, image_root, excluded = inputs(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        build_exclusions({"gqa": [str(image_root)]}, tmp_path / "bad.json")
    with pytest.raises(ValueError, match="8192"):
        prepare_gqa(source, image_root, excluded, tmp_path / "bad.json", train_count=2, val_count=1)
    with pytest.raises(ValueError, match="eligible distinct"):
        prepare_gqa(source, image_root, excluded, tmp_path / "bad.json", train_count=6, val_count=1, smoke=True)


def test_identical_decoded_content_hash_across_formats(tmp_path):
    image = Image.new("RGB", (3, 4), (45, 16, 90))
    image.save(tmp_path / "a.png")
    image.save(tmp_path / "b.bmp")
    assert image_content_sha256(tmp_path / "a.png") == image_content_sha256(tmp_path / "b.bmp")


def test_task_exclusion_export_resume(tmp_path):
    from types import SimpleNamespace
    from cold_ghost.exclusions import export_task_images
    images = [Image.new("RGB", (4, 3), (index * 50, 20, 2)) for index in range(3)]
    class Task:
        config = SimpleNamespace(test_split="test", dataset_path="synthetic", dataset_name="tiny")
        def has_test_docs(self): return True
        def test_docs(self): return [{"image": image} for image in images]
        def doc_to_visual(self, doc): return [doc["image"]]
    first = export_task_images(Task(), "tiny", tmp_path, every=1, progress=lambda _: None)
    assert first["complete"] and first["cursor"] == 3 and len(first["content_sha256"]) == 3
    second = export_task_images(Task(), "tiny", tmp_path, progress=lambda _: None)
    assert first == second
