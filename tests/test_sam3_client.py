import subprocess
import sys
from pathlib import Path

from src.tools.sam3.client import _seed_from
from src.tools.sam3 import sam3_segment_exemplar_prompt, sam3_segment_text_prompt


def test_sam3_segment_text_prompt_deterministic() -> None:
    image = Path("sample.jpg")
    first = sam3_segment_text_prompt(image, "car", "sam3_b", {})
    second = sam3_segment_text_prompt(image, "car", "sam3_b", {})
    assert len(first) == len(second)
    assert first[0].bbox == second[0].bbox


def test_sam3_segment_exemplar_prompt() -> None:
    image = Path("sample.jpg")
    masks = sam3_segment_exemplar_prompt(image, (1, 2, 30, 40), "sam3_b", {})
    assert len(masks) == 1
    assert masks[0].bbox == (1, 2, 30, 40)


def test_seed_is_stable_across_processes() -> None:
    expected = _seed_from(Path("foo/bar.jpg"), "car")
    code = (
        "from pathlib import Path; "
        "from src.tools.sam3.client import _seed_from; "
        "print(_seed_from(Path('foo/bar.jpg'), 'car'))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert int(out) == expected


def test_sam3_segment_text_prompt_api_backend(tmp_path, monkeypatch) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"fake-image")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"masks":[{"polygon":[[1,2],[31,2],[31,42],[1,42]],'
                b'"bbox":[1,2,30,40],"area":1200.0,"confidence":0.88}]}'
            )

    def _fake_urlopen(req, timeout=60):  # noqa: ANN001
        return _FakeResponse()

    monkeypatch.setattr("src.tools.sam3.client.urlrequest.urlopen", _fake_urlopen)
    masks = sam3_segment_text_prompt(
        image,
        "car",
        "sam3_b",
        {
            "backend": "sam3_api",
            "allow_mock_fallback": False,
            "api_url": "https://example.com/sam3",
            "api_key": "test-key",
        },
    )
    assert len(masks) == 1
    assert masks[0].bbox == (1, 2, 30, 40)


def test_sam3_segment_exemplar_prompt_hf_local() -> None:
    import pytest
    import warnings
    from pathlib import Path

    image = Path("sample.jpg")
    
    # 1. Fallback disabled -> raises RuntimeError
    with pytest.raises(RuntimeError, match="hf_local backend does not support exemplar"):
        sam3_segment_exemplar_prompt(
            image, (1, 2, 30, 40), "sam3_b",
            {"backend": "hf_local", "allow_mock_fallback": False}
        )

    # 2. Fallback enabled -> warns and returns mock
    with pytest.warns(UserWarning, match="does not support exemplar prompting"):
        masks = sam3_segment_exemplar_prompt(
            image, (1, 2, 30, 40), "sam3_b",
            {"backend": "hf_local", "allow_mock_fallback": True}
        )
    assert len(masks) == 1
    assert masks[0].bbox == (1, 2, 30, 40)
