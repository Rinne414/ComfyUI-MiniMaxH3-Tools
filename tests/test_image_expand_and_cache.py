"""Expander reply parsing, ratio precedence and the encode cache (no ComfyUI needed)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conditioning_io as cond_io  # noqa: E402
import image_cache as encode_cache  # noqa: E402
import image_expand_parse as expand_parse  # noqa: E402
import image_sizes as sizes  # noqa: E402

GOOD = '{"expanded_caption": "The image is a red fox.", "aspect_ratio": "4:3"}'


@pytest.mark.parametrize(
    "reply",
    [
        GOOD,
        "```json\n" + GOOD + "\n```",
        "Here you go:\n" + GOOD + "\nDone.",
        "{not json} " + GOOD,
    ],
)
def test_reply_with_noise_around_the_json_is_read(reply):
    assert expand_parse.parse_reply(reply) == ("The image is a red fox.", "4:3", "")


def test_edit_key_and_ratio_formats():
    reply = '{"expanded_caption": "Edit text.", "target_aspect_ratio": "1920x1080"}'
    assert expand_parse.parse_reply(reply) == ("Edit text.", "16:9", "")
    assert expand_parse.normalize_ratio("adaptive") == ""
    assert expand_parse.normalize_ratio("0:5") == ""
    assert expand_parse.normalize_ratio(" 8/6 ") == "4:3"


@pytest.mark.parametrize(
    "reply",
    [
        '{"expanded_caption": ""}',
        '{"aspect_ratio": "1:1"}',
        '{"expanded_caption": "cut off',
        "plain text only",
        "[1, 2]",
    ],
)
def test_unreadable_replies_give_nothing_and_a_reason(reply):
    caption, ratio, problem = expand_parse.parse_reply(reply)
    assert caption == "" and ratio == "" and problem


def test_empty_reply_mentions_running_out_of_tokens():
    assert "tokens" in expand_parse.parse_reply("")[2]


def test_ratio_precedence_picture_then_suggestion_then_menu():
    assert sizes.resolve_ratio("1:1", "16:9", (1024, 768)) == ("4:3", "picture")
    assert sizes.resolve_ratio("1:1", "16:9", None) == ("16:9", "suggested")
    assert sizes.resolve_ratio("1:1", "5:4", None) == ("5:4", "suggested")
    assert sizes.resolve_ratio("3:2", "", None) == ("3:2", "menu")
    assert sizes.resolve_ratio("3:2", "adaptive", None) == ("3:2", "menu")
    assert sizes.resolve_ratio("3:2", "7:3", None) == ("21:9", "suggested")


def _key(prompt="a fox", enc="te|1|2", vae="no vae", pics=None):
    return encode_cache.cache_key(prompt, enc, vae, pics if pics is not None else [])


def test_cache_key_depends_on_every_input():
    pic = torch.zeros(1, 4, 4, 3)
    other = pic.clone()
    other[0, 0, 0, 0] = 1.0 / 255
    base = _key(pics=[pic])
    assert base == _key(pics=[pic.clone()])
    assert (
        len(
            {
                base,
                _key(prompt="a cat", pics=[pic]),
                _key(enc="te|1|3", pics=[pic]),
                _key(vae="VAE|5", pics=[pic]),
                _key(pics=[other]),
                _key(pics=[pic, pic]),
            }
        )
        == 6
    )


def test_cache_hit_miss_and_corrupt_file(tmp_path):
    path = os.path.join(tmp_path, "k.safetensors")
    assert encode_cache.read_cache(path) == (None, "cache miss")
    cond = [[torch.ones(1, 2, 3), {"minimax_token_tags": torch.tensor([[1, 0]])}]]
    cond_io.save_conditioning(path, cond)
    got, note = encode_cache.read_cache(path)
    assert note.startswith("cache hit") and torch.equal(got[0][0], cond[0][0])
    with open(path, "wb") as f:
        f.write(b"not a safetensors file")
    got, note = encode_cache.read_cache(path)
    assert got is None and "unreadable" in note
    assert encode_cache.folder_stats(str(tmp_path))[0] == 1


def test_text_cache_missing_valid_and_corrupt(tmp_path):
    path = os.path.join(tmp_path, "r.json")
    assert encode_cache.read_text_cache(path) is None
    encode_cache.write_text_cache(path, GOOD, {"seed": 0})
    assert encode_cache.read_text_cache(path) == GOOD
    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken")
    assert encode_cache.read_text_cache(path) is None
    with open(path, "w", encoding="utf-8") as f:
        f.write("[1, 2]")
    assert encode_cache.read_text_cache(path) is None
