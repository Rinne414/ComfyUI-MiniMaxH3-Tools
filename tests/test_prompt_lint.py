"""Tests for the prompt linter.

The primary bar: MiniMax's own worked examples must pass clean. A linter that
flags the reference prompts from the guide it implements is worse than no
linter, so those fixtures come first and any warning on them is a bug here.

Sources:
  BASE = docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md, section 5 (cases 1-4)
  REF  = docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md, section 7 (complete example)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_lint import lint  # noqa: E402

# --- Official fixtures, transcribed verbatim from the guides ----------------

BASE_CASE_1_T2VA = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from the sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. The doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end."""

BASE_CASE_2_I2VA = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> remains beside the rain-covered train window, preserving her appearance, clothing, seat position, and the carriage layout. The camera trucks right with small amplitude at slow speed as she lifts her gaze from the folded letter toward the passing city lights. Her reflection moves across the glass while the quiet, breathy young woman (S1) says: <d>[English] I get off at the next station.</d> She folds the letter along its existing crease.

overall_soundscape: The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain ticks against the window while paper rustles softly in her hands.

non_diegetic_music: Sustained cello notes at a slow tempo with widely spaced piano tones, gradually decreasing in volume."""

BASE_CASE_3_FL2VA = """How the reference pictures align with the target video - Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a rain-soaked cyclist begins in the position and framing established by Picture 1, holding a closed black umbrella beside a silver bicycle. The camera pulls out with small amplitude at slow speed as she releases the bicycle handle, raises the umbrella above her shoulder, and presses the runner upward until the canopy opens. Water rolls from the expanding fabric while she steps beneath it, rotates the handle into the final angle, and settles into the pose, spacing, and composition established by Picture 2 at the end of the shot.

overall_soundscape: Rain falls steadily on the pavement, followed by the metallic click of the umbrella runner and the soft snap of the canopy opening. Water drips from the bicycle frame as distant traffic passes.

non_diegetic_music: N/A"""

BASE_CASE_4_L2VA = """How the reference pictures align with the target video - <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot begins with an intact drinking glass near the edge of a dark wooden table, while the same hand and sleeve visible in <Picture 1> approach from the right. The camera pushes in with small amplitude at slow speed as the fingertips strike the rim. The glass tips, falls, and hits the floor with a sharp impact; cracks spread through it as fragments slide outward. Toward the end, the moving pieces lose momentum and settle into the exact broken arrangement, hand position, camera angle, lighting, and final composition established by <Picture 1>.

overall_soundscape: Fingertips tap the glass before it scrapes across the tabletop, falls, and breaks with a sharp crash. Small fragments scatter and gradually stop sliding across the floor.

non_diegetic_music: A low electronic pulse at a slow tempo, ending immediately after the glass breaks."""

REF_COMPLETE = """subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall, an orange tufted sofa with patterned pillows, a neon sign, and a wooden coffee table.
<Subject 2> is the fluffy white Samoyed in <Picture 2>, <Picture 3>, and <Picture 4>, with thick white fur, pointed ears, a dark nose, and a curved tail.
<Subject 3> is the young blonde woman in <Video 1>, with long blonde hair and a light-pink button-down shirt with rolled-up sleeves.
<Subject 4> is the young man in <Video 2>, with short wavy brown hair and a dark-grey hoodie with drawstrings.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1), containing a spoken English vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 3> eating a cookie in <Subject 1>. <Subject 4> enters with <Subject 2>, which lunges toward the cookie. The three-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 3> and ends with a canned audience laugh.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the Samoyed's thick white fur, pointed ears, dark nose, and curved tail are retained.
<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the blonde woman's identity, long hair, and light-pink shirt are retained.
<Subject 4> (appears in [Shot 1], [Shot 2]): fully_preserved - the young man's short wavy brown hair and dark-grey hoodie are retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 3> without copying the original signal.

detailed_description:
The target video uses a realistic multi-camera sitcom style with warm indoor lighting.
[Shot 1] A medium shot establishes <Subject 1>, the coffee shop with its exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table. <Subject 3> (S1), the young woman with long blonde hair and a light-pink button-down shirt with rolled-up sleeves, sits on the sofa holding a chocolate-chip cookie. From the left, <Subject 4>, the young man with short wavy brown hair and a dark-grey hoodie with drawstrings, enters holding the leash of <Subject 2>, the thick-furred white Samoyed with pointed ears, a dark nose, and a curved tail. The dog lunges toward the cookie and pulls the leash taut. <Subject 3> (S1) jerks her hand back and, using the clear youthful voice timbre referenced from <Audio 1>, exclaims with light annoyance, <d>[English] Hey! Watch your dog!</d> She closes her lips and guards the cookie while <Subject 4> pulls the dog back.
[Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1, sitting beside <Subject 3> on the sofa and holding <Subject 2> securely in his arms. <Subject 4> (S2) says in a casual young male voice with a playful tone and an easy conversational pace, <d>[English] He just likes cookies more than me.</d> He closes his mouth into an apologetic smile and strokes the dog's thick white fur.
[Shot 3] At 00:05.000, the shot cuts to a close-up of <Subject 3> (S1), the blonde woman in the light-pink shirt from Shot 1. Her annoyance softens as she looks toward the Samoyed. <Subject 3> (S1) replies in the same clear youthful voice referenced from <Audio 1> with an amused cadence, <d>[English] Well, he has good taste at least.</d> She smiles and raises the cookie in a small toast-like gesture. A classic canned audience laugh begins immediately after the line and continues through the final frame.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene.

non_diegetic_music:
N/A"""

OFFICIAL_BASE = [
    ("base case 1 (T2VA)", BASE_CASE_1_T2VA),
    ("base case 2 (I2VA)", BASE_CASE_2_I2VA),
    ("base case 3 (FL2VA)", BASE_CASE_3_FL2VA),
    ("base case 4 (L2VA)", BASE_CASE_4_L2VA),
]


@pytest.mark.parametrize("name,text", OFFICIAL_BASE)
def test_official_base_examples_pass_clean(name, text):
    # fl2va mode: cases 2-4 legitimately carry <Picture N> instruction headers,
    # which are part of the documented keyframe format, not stray ref tags.
    warnings = lint(text, mode="fl2va")
    assert warnings == [], "%s produced warnings: %s" % (name, warnings)


def test_official_ref_example_passes_clean():
    warnings = lint(REF_COMPLETE, mode="ref2va", ref_images=4, ref_videos=2,
                    ref_audios=1)
    assert warnings == [], "ref example produced warnings: %s" % warnings


def test_t2va_case_passes_in_t2va_mode():
    assert lint(BASE_CASE_1_T2VA, mode="t2va") == []


# --- Each rule must fire on exactly its own violation ----------------------

def _only(warnings, needle):
    hits = [w for w in warnings if needle.lower() in w.lower()]
    assert hits, "expected a warning matching %r, got: %s" % (needle, warnings)
    return hits


def test_missing_field_is_flagged():
    text = BASE_CASE_1_T2VA.replace("non_diegetic_music:", "background_music:")
    _only(lint(text, mode="t2va"), "missing required field 'non_diegetic_music:'")


def test_misspelled_field_is_flagged():
    text = BASE_CASE_1_T2VA.replace("overall_soundscape:", "overall_soundscapes:")
    _only(lint(text, mode="t2va"), "missing required field 'overall_soundscape:'")


def test_fields_out_of_order_is_flagged():
    parts = BASE_CASE_1_T2VA.split("\n\n")
    swapped = "\n\n".join([parts[0], parts[2], parts[1]])
    _only(lint(swapped, mode="t2va"), "out of order")


def test_shot_1_with_timestamp_is_flagged():
    text = BASE_CASE_1_T2VA.replace("[Shot 1] Live-action",
                                    "[Shot 1] At 00:00.000, Live-action")
    _only(lint(text, mode="t2va"), "must not have one")


def test_non_increasing_cut_time_is_flagged():
    text = BASE_CASE_1_T2VA.replace("[Shot 2] At 00:05.000",
                                    "[Shot 2] At 00:00.500")
    # Shot 1 has no timestamp, so make a 3-shot case where 3 precedes 2 in time.
    text = text.replace("previous shot.",
                        "previous shot. [Shot 3] At 00:00.100, the shot cuts back.")
    _only(lint(text, mode="t2va"), "must strictly increase")


def test_cut_time_past_duration_is_flagged():
    _only(lint(BASE_CASE_1_T2VA, mode="t2va", duration_seconds=4.0),
          "past the 4.000 s clip duration")


def test_missing_cut_time_is_flagged():
    text = BASE_CASE_1_T2VA.replace("[Shot 2] At 00:05.000, the camera cuts",
                                    "[Shot 2] the camera cuts")
    _only(lint(text, mode="t2va"), "no 'At MM:SS.mmm,' cut time")


def test_shot_numbering_gap_is_flagged():
    text = BASE_CASE_1_T2VA.replace("[Shot 2]", "[Shot 3]")
    _only(lint(text, mode="t2va"), "sequentially from 1")


def test_unknown_dialogue_language_is_flagged():
    text = BASE_CASE_1_T2VA.replace("[English] First batch", "[Klingon] First batch")
    _only(lint(text, mode="t2va"), "outside the 11 with stable support")


def test_missing_language_tag_is_flagged():
    text = BASE_CASE_1_T2VA.replace("<d>[English] First batch of the morning.</d>",
                                    "<d>First batch of the morning.</d>")
    _only(lint(text, mode="t2va"), "no [Language] tag")


def test_unbalanced_dialogue_tag_is_flagged():
    text = BASE_CASE_1_T2VA.replace("of the morning.</d>", "of the morning.")
    _only(lint(text, mode="t2va"), "unbalanced dialogue tags")


def test_unattributed_dialogue_is_flagged():
    text = BASE_CASE_1_T2VA.replace("voice (S1) places", "voice places")
    _only(lint(text, mode="t2va"), "no (Sx) speaker ID")


def test_speaker_id_gap_is_flagged():
    text = BASE_CASE_1_T2VA.replace("(S1) places", "(S3) places")
    _only(lint(text, mode="t2va"), "speaker IDs jump")


def test_diegetic_source_in_music_is_flagged():
    text = BASE_CASE_1_T2VA.replace(
        "non_diegetic_music: A soft acoustic-guitar pattern",
        "non_diegetic_music: A radio plays a soft acoustic-guitar pattern")
    _only(lint(text, mode="t2va"), "characters can hear")


def test_dialogue_restated_in_soundscape_is_flagged():
    text = BASE_CASE_1_T2VA.replace(
        "overall_soundscape: Wooden shutters",
        "overall_soundscape: He says First batch of the morning. Wooden shutters")
    _only(lint(text, mode="t2va"), "restates dialogue")


def test_too_many_soundscape_sentences_is_flagged():
    extra = " A dog barks. A car passes. A bell rings. Someone coughs. Wind rises."
    text = BASE_CASE_1_T2VA.replace("being sliced.", "being sliced." + extra)
    _only(lint(text, mode="t2va"), "1-4")


def test_voiceover_without_exact_phrase_is_flagged():
    text = BASE_CASE_1_T2VA.replace(
        "(S1) places a fresh loaf on the wooden counter and says:",
        "(S1) speaks in a voiceover:")
    _only(lint(text, mode="t2va"), "exact phrase")


def test_voiceover_without_closed_lips_is_flagged():
    text = BASE_CASE_1_T2VA.replace(
        "(S1) places a fresh loaf on the wooden counter and says:",
        "(S1) says in an off-screen voiceover:")
    _only(lint(text, mode="t2va"), "lips remain closed")


def test_ref_tag_in_t2va_mode_is_flagged():
    text = BASE_CASE_1_T2VA.replace("a baker opening", "<Picture 1> a baker opening")
    _only(lint(text, mode="t2va"), "mode is t2va")


def test_picture_beyond_connected_refs_is_flagged():
    _only(lint(REF_COMPLETE, mode="ref2va", ref_images=2, ref_videos=2, ref_audios=1),
          "only 2 picture input(s) are connected")


def test_ref_tag_numbering_gap_is_flagged():
    text = REF_COMPLETE.replace("<Picture 2>", "<Picture 7>")
    _only(lint(text, mode="ref2va", ref_images=9, ref_videos=2, ref_audios=1),
          "numbering has gaps")


def test_undefined_subject_is_flagged():
    text = REF_COMPLETE.replace("holding <Subject 2> securely", "holding <Subject 9> securely")
    _only(lint(text, mode="ref2va", ref_images=4, ref_videos=2, ref_audios=1),
          "never defined in subject_definitions")


def test_missing_summary_task_prefix_is_flagged():
    text = REF_COMPLETE.replace("[reference generation + audio reference] The target",
                                "The target")
    _only(lint(text, mode="ref2va", ref_images=4, ref_videos=2, ref_audios=1),
          "task type")


def test_invalid_task_type_is_flagged():
    text = REF_COMPLETE.replace("[reference generation + audio reference]",
                                "[magic generation + audio reference]")
    _only(lint(text, mode="ref2va", ref_images=4, ref_videos=2, ref_audios=1),
          "is not one of")


def test_speaker_id_in_retention_is_flagged():
    text = REF_COMPLETE.replace(
        "<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved",
        "<Subject 3> (S1) (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved")
    _only(lint(text, mode="ref2va", ref_images=4, ref_videos=2, ref_audios=1),
          "must not appear in this section")


def test_invalid_retention_marker_is_flagged():
    text = REF_COMPLETE.replace(
        "<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved",
        "<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): mostly_kept")
    _only(lint(text, mode="ref2va", ref_images=4, ref_videos=2, ref_audios=1),
          "must use one of")


def test_audio_marker_on_visual_entry_is_flagged():
    # fully_copy is an audio-only marker
    text = REF_COMPLETE.replace(
        "<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved",
        "<Subject 2> (appears in [Shot 1], [Shot 2]): fully_copy")
    _only(lint(text, mode="ref2va", ref_images=4, ref_videos=2, ref_audios=1),
          "visual entries must use")


def test_empty_prompt_is_flagged():
    assert lint("") == ["prompt is empty"]
    assert lint("   \n  ") == ["prompt is empty"]


# --- False-positive probes -------------------------------------------------

def test_na_soundscape_and_music_are_accepted():
    text = BASE_CASE_1_T2VA.split("overall_soundscape:")[0].rstrip()
    text += "\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A"
    assert lint(text, mode="t2va") == []


def test_three_increasing_shots_pass():
    text = BASE_CASE_1_T2VA.replace(
        "previous shot.",
        "previous shot. [Shot 3] At 00:09.250, the shot cuts to the empty street.")
    assert lint(text, mode="t2va") == []


def test_diegetic_radio_in_description_is_not_flagged():
    text = BASE_CASE_1_T2VA.replace(
        "places a fresh loaf",
        "switches on a radio playing a faint march, then places a fresh loaf")
    assert lint(text, mode="t2va") == [], lint(text, mode="t2va")


def test_group_speaker_id_is_accepted():
    text = BASE_CASE_1_T2VA.replace(
        "voice (S1) places a fresh loaf on the wooden counter and says:",
        "voice (S1) and his apprentice (S2) place a fresh loaf down and (S1,S2) say together:")
    assert lint(text, mode="t2va") == [], lint(text, mode="t2va")


def test_all_eleven_languages_accepted():
    for lang in ("Arabic", "Chinese", "English", "French", "German", "Italian",
                 "Japanese", "Korean", "Portuguese", "Russian", "Spanish"):
        text = BASE_CASE_1_T2VA.replace("[English] First batch", "[%s] First batch" % lang)
        assert lint(text, mode="t2va") == [], "%s was rejected" % lang


def test_correct_voiceover_form_passes():
    text = BASE_CASE_1_T2VA.replace(
        "(S1) places a fresh loaf on the wooden counter and says:",
        "(S1) says in an off-screen voiceover:")
    text = text.replace(
        "<d>[English] First batch of the morning.</d>",
        "<d>[English] First batch of the morning.</d> while his lips remain completely closed.")
    assert lint(text, mode="t2va") == [], lint(text, mode="t2va")
