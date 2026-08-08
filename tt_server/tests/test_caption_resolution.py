"""Self-check: caption frames reach the VLM at their own size, capped, not squashed.

The HTTP layer used to do `.resize((224, 224))` on every incoming frame -- unconditional,
square, no setting. The camera is 1224x1024, so the VLM saw 1/25 of the area, and captions
could not name what was no longer in the pixels: of 272 captions, 33 named a building and
1 mentioned the police call pole the questions ask about (#59). It hid itself too, since a
resolution sweep from 1224 down to 256 returned ten identical responses.

    python tt_server/tests/test_caption_resolution.py
"""
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tt_server.domain.models import cap_long_side  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def read(relative):
    return open(os.path.join(REPO, relative)).read()


def test_a_frame_over_the_cap_is_downscaled():
    from PIL import Image

    sent = cap_long_side(Image.new("RGB", (1224, 1024)), 768)
    assert max(sent.size) == 768, sent.size


def test_the_aspect_ratio_survives():
    """A square resize is what lost the detail; keeping shape is half the fix."""
    from PIL import Image

    sent = cap_long_side(Image.new("RGB", (1224, 1024)), 768)
    assert abs(sent.size[0] / sent.size[1] - 1224 / 1024) < 0.01, sent.size


def test_a_frame_within_the_cap_is_untouched():
    from PIL import Image

    assert cap_long_side(Image.new("RGB", (320, 240)), 768).size == (320, 240)


def test_zero_disables_downscaling():
    from PIL import Image

    assert cap_long_side(Image.new("RGB", (1224, 1024)), 0).size == (1224, 1024)


def test_the_http_layer_no_longer_squashes_to_224():
    """Checks the executable body, not the docstring, which quotes the old call."""
    source = read("tt_server/presentation/http/app.py")
    decode = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == "_decode_image")
    body = ast.unparse(decode.body[1:])          # [0] is the docstring
    assert "resize" not in body, body
    assert "cap_long_side" in body, body


def test_the_cap_is_configurable():
    """Hardcoding it again is how this became invisible the first time."""
    assert "TT_VLM_IMAGE_LONG_SIDE" in read("tt_server/presentation/http/app.py")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
