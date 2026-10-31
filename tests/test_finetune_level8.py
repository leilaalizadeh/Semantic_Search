import os
import subprocess
import sys

from semantic_search.levels.level8.semantic import cache_path


def test_base_model_is_trained_in_float32():
    # gte-small is stored in float16; training it in float16 turned every weight into NaN.
    # Checked in a separate Python process: the other tests have loaded FAISS, and loading the model
    # next to FAISS crashes on macOS (training itself never loads FAISS).
    code = ("from semantic_search.levels.level8.train import load_base_model; "
            "print({str(p.dtype) for p in load_base_model('thenlper/gte-small').parameters()})")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.stdout.strip().splitlines()[-1] == "{'torch.float32'}"


def test_retrained_model_does_not_reuse_old_embeddings(tmp_path):
    # A model trained again keeps its folder name, so the cache must also look at the weights file
    model_dir = tmp_path / "gte-small-llm"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"first training")
    texts = ["Coleman camping tent", "Yoga mat"]
    first = cache_path(texts, str(model_dir), tmp_path)

    weights.write_bytes(b"second training")
    os.utime(weights, ns=(weights.stat().st_atime_ns, weights.stat().st_mtime_ns + 1_000_000_000))
    assert cache_path(texts, str(model_dir), tmp_path) != first


def test_hugging_face_models_keep_the_same_cache_name(tmp_path):
    texts = ["Coleman camping tent"]
    assert cache_path(texts, "thenlper/gte-small", tmp_path) == cache_path(texts, "thenlper/gte-small", tmp_path)
