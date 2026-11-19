import itertools

import pytest

from app.providers import Segment
from app.stitch import Chunk, plan, stitch


@pytest.mark.parametrize("duration,max_len", [(100, 30), (3000, 1771), (240, 60), (61, 60)])
def test_plan_covers_the_recording_in_chunks_within_the_limit(duration, max_len):
    chunks = plan(duration, max_len)
    assert chunks[0].start == 0
    assert chunks[-1].start + chunks[-1].length == pytest.approx(duration)
    assert all(c.length <= max_len + 1e-9 for c in chunks)
    for a, b in itertools.pairwise(chunks):
        overlap = a.start + a.length - b.start
        assert overlap == pytest.approx(min(60, max_len / 4))


def test_short_recordings_are_one_chunk():
    assert plan(50, 60) == [Chunk(0, 50)]


# A made-up conversation: 120 distinct words, the speaker changes every 15 words.
WORDS = [f"word{i}" for i in range(120)]
SPEAKER = [(i // 15) % 2 for i in range(120)]


def segments(lo, hi, swap=False, cut_first=False, cut_last=False):
    """Words lo..hi-1 as a chunk would return them: chunk-local speaker numbers
    (optionally swapped), and optionally a word chopped at either edge."""
    words = list(WORDS[lo:hi])
    if cut_first:
        words[0] = words[0][-2:]
    if cut_last:
        words[-1] = words[-1][:3]
    out = []
    for i, w in zip(range(lo, hi), words, strict=True):
        spk = SPEAKER[i] ^ swap
        if out and out[-1].speaker == spk:
            out[-1].text += " " + w
        else:
            out.append(Segment(spk, None, None, w))
    return out


def flatten(segs):
    return [(s.speaker, w) for s in segs for w in s.text.split()]


TRUTH = list(zip(SPEAKER, WORDS, strict=True))


def test_overlap_fixes_swapped_speakers_and_chopped_words():
    chunks = [
        (Chunk(0, 1), segments(0, 70, cut_last=True)),
        (Chunk(1, 1), segments(50, 120, swap=True, cut_first=True)),
    ]
    result = stitch(chunks)
    assert flatten(result.segments) == TRUTH
    assert result.warnings == []


def test_speaker_silent_in_the_overlap_is_matched_by_elimination():
    # The overlap (words 60-74) is all speaker 0, and the next chunk swaps labels.
    chunks = [(Chunk(0, 1), segments(0, 75)), (Chunk(1, 1), segments(60, 120, swap=True))]
    assert flatten(stitch(chunks).segments) == TRUTH


def test_three_chunks():
    chunks = [
        (Chunk(0, 1), segments(0, 50)),
        (Chunk(1, 1), segments(35, 90, swap=True)),
        (Chunk(2, 1), segments(75, 120)),
    ]
    assert flatten(stitch(chunks).segments) == TRUTH


def test_unalignable_join_keeps_everything_and_warns():
    a = [Segment(0, None, None, "hola que tal")]
    b = [Segment(0, None, None, "bien gracias y tu")]
    result = stitch([(Chunk(0, 60), a), (Chunk(45, 60), b)])
    assert [w for _, w in flatten(result.segments)] == "hola que tal bien gracias y tu".split()
    assert "00:00:45" in result.warnings[0]


def test_timestamps_shift_by_the_chunk_start():
    a = [Segment(0, 0.0, 1.0, " ".join(WORDS[:40]))]
    b = [Segment(0, 5.0, 6.0, " ".join(WORDS[30:60]))]
    segs = stitch([(Chunk(0, 60), a), (Chunk(100, 60), b)]).segments
    assert segs[0].start == 0.0
    assert segs[0].end == 105.0
