import re

from PIL import Image

# Dense, retrieval-oriented caption prompt — an alternative to the paper's verbose
# "Think step by step" query, selected via `preprocess_captions.py --caption_style dense`.
#
# WHY: on the seq0 run made with the paper prompt, 271/272 captions opened with the
# same preamble ("Certainly! Let's break down the image description step by step:")
# and averaged ~196 words, mostly speculation. That boilerplate dominates the mxbai
# sentence vector, which is why embedding-cosine memory selection (30.1 m) lost to
# discrete content-word overlap (17.6 m) — see _select_memory in
# agents/remembr_agent.py. Comma phrases keep the vector on objects and relations.
#
# The rules go beyond bare object tags because the CODa human_qa questions need
# attributes ("what color jacket"), people and their actions ("any bicyclists"),
# indoor/outdoor state ("how long have you been outside") and sign text.
#
# HARD LENGTH BUDGET: the Qwen prefill pads to the nearest power of two and only the
# 128/256-token shapes work on this board — a request encoding past 256 pads to 512
# and the server 500s instantly (measured: 244 OK, 297 fails; the paper prompt
# encodes to 138, which is why it never hit this). Encoded length = prompt tokens +
# ~21 template tokens + ~66 vision tokens per 224x224 image, leaving the prompt
# ~165 tokens at --num-video-frames 1. Re-check with the Qwen tokenizer if you edit.
#
# FORMAT NOTES — every clause below is load-bearing, from measured 3B failures:
#   * no examples at all -> the list collapses into single-word soup
#     ("indoor, ground, gray, concrete, ramp") with no phrase structure;
#   * schematic placeholder examples (<color> <object>) -> the same collapse;
#   * greedy decoding (temperature 0) -> repetition loops ("near sign, near sign,
#     ..." to the token cap). The captioner now samples at 0.2 and the server stops
#     at the first repeated phrase.
#   * "do not guess a name for it" is what keeps the model off eval-critical nouns.
#     Without it, a frame holding a red painted curb was captioned "red fire hydrant"
#     -- an object the questions ask about -- while the paper prompt on the same frame
#     correctly said "red curb". Terseness removes the room to hedge, so permission to
#     describe shape instead of naming has to be explicit.
#   * ANY concrete phrase anywhere in the prompt gets copied onto unrelated frames,
#     and the more imperative its position the worse: an illustration written inline
#     as "write X, not Y" was copied onto 5 frames out of 5. Concrete text therefore
#     appears ONLY in the example block, and DENSE_EXAMPLE_PHRASES lets
#     clean_dense_caption() strip verbatim copies (still ~1 frame in 4).
#   * enumerating the prepositions ("next to, on, near, under, in front of, behind")
#     made the model recite them as subjectless phrases ("under tree, near building,
#     in front of street sign"), so the relation is now asked for in prose.
#   * "including ... anything at the edges of the frame" is what finally captured the
#     campus sculpture, which every earlier version dropped entirely: terseness makes
#     the model report only what fills the frame, so a large object cropped into a
#     corner loses to the pavement in front of it.
#   * "one phrase per object, never its parts" stops "window, window frame, window
#     sill, window ledge, window ledge with plant" once completeness is requested.
#
# ⚠️ WHAT THE PROMPT CANNOT FIX: the server resizes every frame to 224x224 (forced --
# see the budget note above), which is a 5.5x downscale of CODa's 1224x1024 frames.
# Small and distant objects are a few pixels, so some invention is irreducible here:
# phantom people and a "red fire hose" survived every wording tried, and the paper
# prompt hallucinates on the same frames too (it just hedges with "appears to be").
# The real lever is resolution, and that is blocked by the prefill limit, not by text.
DENSE_CAPTION_PROMPT = (
    "Frames from a robot's camera on a university campus. List every distinct thing "
    "you see, as comma-separated phrases, including large structures and anything at "
    "the edges of the frame. One phrase per object, never its parts. Give each its "
    "color, material, and what makes it distinctive: its shape, its pattern, or what "
    "hangs or sits on it - not a bare category. When you cannot tell what a thing is, "
    "give its color and shape and stop; do not guess a name for it. Name a person only "
    "if you clearly see one; never guess a count. Quote sign text only if legible. "
    "Begin with indoor or outdoor and the ground. No sentences, no filler.\n"
    "Format example only, never reuse its content:\n"
    "outdoor gravel path, orange cone on curb, low wall behind parked bus"
)

# Exactly the phrases used in the prompt's examples, including the ones from earlier
# revisions so captions made with those can still be cleaned. The model copies these
# onto unrelated frames, so clean_dense_caption() drops verbatim copies: losing a rare
# genuine "orange cone on curb" is much cheaper than injecting phantom objects into
# every fourth memory.
DENSE_EXAMPLE_PHRASES = frozenset({
    "outdoor gravel path", "orange cone on curb", "low wall behind parked bus",
    "indoor tiled lobby", "wooden reception desk near glass wall", "two people sitting",
})

# Meta-commentary the model emits when it follows a rule but has nothing to report:
# "sign text not readable", "quote sign visible", "no distinctive features". These
# describe the prompt rather than the scene and would pollute the embedding.
_DENSE_META_RE = re.compile(
    r"^(quote\b"
    r"|no\s+(text|people|sign|distinctive)"
    r"|sign\s+(text\s+)?(not\s+)?(readable|legible|visible|with\s+text)"
    r"|(nothing|none)\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Balanced captions: prose without the boilerplate (--caption_style balanced)
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. The selector that actually answers position questions
# (_select_memory) scores a memory by how many content words its caption shares
# with the question, so its whole signal is caption vocabulary. Measured over the
# 12 CODa position questions:
#
#     captions   words   content words   question words covered   selector
#     paper       196        79.5             44 of 66            16.0 m 9/12
#     dense-v2     22        15.8             24 of 66            40.3 m 4/12
#
# On dense captions the selector performs at or below a blind pick, because 16
# content words cannot separate five candidates. Terseness was the mistake: it
# bought a cleaner embedding space (mean pairwise cosine 0.906 -> 0.713) and gave
# it all back at selection. What the paper prompt has that dense lacks is not
# length as such but VOCABULARY, and the two are not the same thing -- asking for
# longer comma phrases only produced repeated filler ("beside a concrete path"
# three times) and left content words flat at 17.8.
#
# Sentences are what generate vocabulary: articles, verbs and elaboration drag in
# the category words the questions use ("place to sit", "recycling bin"), which a
# tag list never emits. So this prompt allows prose and bans only the two things
# that made the paper captions bad for retrieval -- the fixed preamble and the
# speculation ("which is typical of...", "likely used for..."). Measured on six
# frames: 102 words, 42.8 content words, i.e. between the two, which is the point.
#
# The model will NOT stop opening with "The image shows" no matter how explicitly
# the prompt forbids it (5 of 6 frames still did), so that is stripped afterwards
# by clean_balanced_caption() instead of fought in the prompt.
BALANCED_CAPTION_PROMPT = (
    "Frames from a robot's camera on a university campus. Describe everything visible in "
    "plain sentences: the objects and what kind of thing each one is, their colours, "
    "materials, sizes and positions, any people and what they are doing, and any sign or "
    "building text you can read. Begin straight away with the description: no preamble, "
    "no \"Certainly\", no headings, no numbered lists. Write only what is visible - never "
    "what something is probably for, what it is typical of, or what it suggests. If you "
    "cannot tell what a thing is, describe its colour and shape instead of naming it."
)

# The lead-in the model cannot be told to drop. Only the announcing clause is removed,
# not the sentence: "The image shows a large concrete area in front of a building." ->
# "A large concrete area in front of a building." -- the constant part goes, the content
# stays. Removing it matters because a preamble shared by every caption is exactly what
# pushed the paper captions to 0.906 mean pairwise cosine.
_BALANCED_LEADIN_RE = re.compile(
    r"^\s*(?:in\s+)?(?:this|the)\s+(?:image|photo|picture|scene|frame|camera\s+view|camera)"
    r"\s*,?\s*(?:shows?|captures?|depicts?|displays?|features?|is\s+of|we\s+see)?\s*",
    re.IGNORECASE,
)


def clean_balanced_caption(text: str) -> str:
    """Strip the fixed announcing clause from a balanced caption, keeping its content."""
    out = _BALANCED_LEADIN_RE.sub("", text or "", count=1).lstrip()
    return (out[:1].upper() + out[1:]) if out else (text or "").strip()


def clean_dense_caption(text: str) -> str:
    """Drop copied example phrases and prompt meta-commentary from a dense caption,
    preserving order and de-duplicating."""
    seen = set()
    kept = []
    for phrase in (p.strip() for p in (text or "").split(",")):
        low = phrase.lower().strip(" .")
        if not low or low in DENSE_EXAMPLE_PHRASES or low in seen:
            continue
        if _DENSE_META_RE.match(low):
            continue
        seen.add(low)
        kept.append(phrase.strip(" ."))
    return ", ".join(kept)


class Captioner:
    def caption(self, images: list[Image.Image]):
        raise NotImplementedError