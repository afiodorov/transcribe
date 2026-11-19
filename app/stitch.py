"""Transcribe recordings longer than a model accepts: split, then stitch.

The recording is cut into equal chunks that overlap by a short stretch. Each
chunk numbers its speakers on its own, so "speaker 1" in one chunk need not
be "speaker 1" in the next. Both chunks transcribe the overlap, so aligning
those words tells us which speaker is which, and gives a cut point inside text
both chunks agree on, so no word is chopped or duplicated.
"""

import difflib
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.providers import Segment, merge_turns

MAX_OVERLAP = 60.0
WINDOW_WORDS = 600  # how far into each side of a boundary to look for the overlap
MIN_MATCHED_WORDS = 8


@dataclass
class Chunk:
    start: float
    length: float


def plan(duration: float, max_len: float) -> list[Chunk]:
    """Fewest equal chunks of at most max_len that cover duration, overlapping
    by a quarter of max_len (capped at MAX_OVERLAP)."""
    if duration <= max_len:
        return [Chunk(0.0, duration)]
    overlap = min(MAX_OVERLAP, max_len / 4)
    n = math.ceil((duration - overlap) / (max_len - overlap))
    step = (duration - overlap) / n
    return [Chunk(i * step, step + overlap) for i in range(n)]


@dataclass
class Word:
    speaker: int
    text: str
    start: float | None  # start of the segment the word came from


def to_words(segments: list[Segment], offset: float) -> list[Word]:
    return [
        Word(s.speaker, w, None if s.start is None else s.start + offset)
        for s in segments
        for w in s.text.split()
    ]


def norm(word: str) -> str:
    return re.sub(r"\W", "", word.lower())


def align(a: list[Word], b: list[Word]) -> list[tuple[int, int]]:
    """Index pairs (in a, in b) of words both sides transcribed the same, looking
    only at the end of a and the start of b."""
    a0 = max(0, len(a) - WINDOW_WORDS)
    # Words that normalise to nothing ("—") get unique keys so they never match.
    ka = [norm(w.text) or f"\0a{i}" for i, w in enumerate(a[a0:])]
    kb = [norm(w.text) or f"\0b{i}" for i, w in enumerate(b[:WINDOW_WORDS])]
    matcher = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
    return [
        (a0 + m.a + k, m.b + k)
        for m in matcher.get_matching_blocks()
        if m.size >= 2  # lone common words ("the", "que") are coincidences
        for k in range(m.size)
    ]


def map_speakers(a: list[Word], b: list[Word], pairs: list[tuple[int, int]]) -> dict[int, int]:
    """Chunk-local speaker in b -> global speaker in a, by majority over the
    aligned words; a speaker silent in the overlap is matched by elimination
    when exactly one speaker is left on each side."""
    votes = Counter((b[j].speaker, a[i].speaker) for i, j in pairs)
    mapping: dict[int, int] = {}
    for (sb, sa), _ in votes.most_common():
        if sb not in mapping and sa not in mapping.values():
            mapping[sb] = sa
    left_b = sorted({w.speaker for w in b} - mapping.keys())
    left_a = sorted({w.speaker for w in a} - set(mapping.values()))
    if len(left_b) == 1 and len(left_a) == 1:
        mapping[left_b[0]] = left_a[0]
    next_id = max((w.speaker for w in a), default=-1) + 1
    for sb in left_b:
        if sb not in mapping:
            mapping[sb] = next_id
            next_id += 1
    return mapping


@dataclass
class Stitched:
    segments: list[Segment]
    warnings: list[str]


def stitch(chunks: list[tuple[Chunk, list[Segment]]]) -> Stitched:
    """Join per-chunk transcripts, given in order with the chunk they came from."""
    words: list[Word] = []
    warnings: list[str] = []
    for chunk, segments in chunks:
        new = to_words(segments, chunk.start)
        if not words:
            words = new
            continue
        pairs = align(words, new)
        if len(pairs) >= MIN_MATCHED_WORDS:
            mapping = map_speakers(words, new, pairs)
            cut_a, cut_b = pairs[len(pairs) // 2]  # middle of the agreed stretch
            words = words[:cut_a] + relabel(new[cut_b:], mapping)
        else:
            # Nothing to align on (silence, or the chunks disagree): keep both
            # sides whole and say so rather than guess silently.
            mapping = map_speakers(words, new, [])
            words = words + relabel(new, mapping)
            warnings.append(
                f"Could not line up the recording around {clock(chunk.start)}: speaker "
                "names after that point may be swapped and a few words may repeat."
            )
    segments = merge_turns([Segment(w.speaker, w.start, w.start, w.text) for w in words])
    return Stitched(segments, warnings)


def relabel(words: list[Word], mapping: dict[int, int]) -> list[Word]:
    return [Word(mapping[w.speaker], w.text, w.start) for w in words]


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
