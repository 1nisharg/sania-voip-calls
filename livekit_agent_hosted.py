"""
livekit_agent_hosted.py
=======================

Hosted LiveKit/WebRTC version of the Sania voice agent.

Architecture:
    Browser
       |
       v
    app.py
       |
       +--> creates LiveKit room
       +--> creates explicit agent dispatch
       +--> returns browser token
       |
       v
    LiveKit Cloud
       |
       v
    Sania worker
       |
       +--> Deepgram Nova-3 STT
       +--> Groq GPT-OSS-120B LLM
       +--> Fish Audio TTS
       +--> Silero VAD
       +--> deterministic conversation router
       +--> local semantic KB

IMPORTANT:
    FastEmbed is intentionally NOT loaded at module import time.

    LiveKit uses separate worker/job processes. Loading the FastEmbed model
    during worker initialization can prevent a process from becoming warm,
    which causes:
        "no warmed process available for job"

    The KB embedding model is therefore loaded lazily on the first genuine
    question that reaches the KB branch.

Run locally:
    python livekit_agent_hosted.py download-files
    python livekit_agent_hosted.py dev

Hosted:
    python livekit_agent_hosted.py start
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import AsyncIterable

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    RoomInputOptions,
    StopResponse,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import ModelSettings
from livekit.plugins import deepgram, fishaudio, openai, silero


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv()

logger = logging.getLogger("livekit_agent")
logging.basicConfig(level=logging.INFO)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_REASONING_EFFORT = os.getenv(
    "GROQ_REASONING_EFFORT",
    "low",
)

FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")

FISH_AUDIO_MODEL = os.getenv(
    "FISH_AUDIO_MODEL",
    "s2.1-pro-free",
)

FISH_AUDIO_VOICE_ID = os.getenv(
    "FISH_AUDIO_VOICE_ID",
    "",
)

FISH_AUDIO_SPEED = float(
    os.getenv(
        "FISH_AUDIO_SPEED",
        "1.15",
    )
)

MAX_SPOKEN_SENTENCES_PER_TURN = int(
    os.getenv(
        "MAX_SPOKEN_SENTENCES_PER_TURN",
        "2",
    )
)


# ============================================================================
# FAREWELL DETECTION
# ============================================================================

_FAREWELL_PATTERNS = re.compile(
    r"\b("
    r"have a wonderful day"
    r"|goodbye"
    r"|take care"
    r"|talk soon"
    r"|wishing your business continued success"
    r")\b",
    re.IGNORECASE,
)


# ============================================================================
# DEEPGRAM KEYTERMS
# ============================================================================

_STATIC_KEYTERMS = [
    "Viator",
    "Aarna",
    "Sania",
    "Mondee",
    "GetYourGuide",
    "Klook",
    "TripAdvisor",
    "Expedia",
    "Booking.com",
    "Airbnb",
]


# ============================================================================
# TEXT PREPARATION FOR SPEECH
# ============================================================================

def _normalize_number_words(n: int) -> str:
    """Convert an integer to natural English speech words."""

    ones = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]

    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    scales = [
        "",
        "thousand",
        "million",
        "billion",
        "trillion",
    ]

    if n == 0:
        return "zero"

    if n < 0:
        return "minus " + _normalize_number_words(-n)

    def under_1000(x: int) -> str:
        parts = []

        if x >= 100:
            parts.append(
                ones[x // 100] + " hundred"
            )
            x %= 100

        if x >= 20:
            word = tens[x // 10]

            if x % 10:
                word += "-" + ones[x % 10]

            parts.append(word)

        elif x:
            parts.append(ones[x])

        return " ".join(parts)

    parts = []
    scale_idx = 0

    while n:
        group = n % 1000

        if group:
            group_text = under_1000(group)

            if scale_idx:
                group_text += " " + scales[scale_idx]

            parts.append(group_text)

        n //= 1000
        scale_idx += 1

    return " ".join(reversed(parts))


def _number_to_spoken(match: "re.Match") -> str:
    raw = match.group(0)

    try:
        return _normalize_number_words(
            int(
                raw.replace(",", "")
                .replace(" ", "")
            )
        )
    except ValueError:
        return raw


_CURRENCY_RE = re.compile(
    r"(?P<cur>AED|USD|EUR|GBP|\$|€|£)"
    r"\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*%"
)

_MULTIPLIER_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*[xX]\b"
)

_LARGE_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})+\b"
    r"|\b\d{1,3}(?: \d{3})+\b"
    r"|\b\d{4,}\b"
)

_DECIMAL_RE = re.compile(
    r"\b\d+\.\d+\b"
)


def _decimal_to_spoken(value: str) -> str:
    left, right = value.split(".", 1)

    return (
        _normalize_number_words(int(left))
        + " point "
        + " ".join(
            _normalize_number_words(int(ch))
            for ch in right
        )
    )


def _normalize_for_speech(text: str) -> str:
    """
    Normalize business text for natural spoken pronunciation.

    Examples:
        65000 -> sixty-five thousand
        25% -> twenty-five percent
        AED 50,000 -> fifty thousand UAE dirhams
        2.5M -> two point five million
    """

    if not text:
        return ""

    out = text

    def currency_repl(m: "re.Match") -> str:
        cur = m.group("cur").upper()
        num = m.group("num")

        if "." in num:
            spoken_num = _decimal_to_spoken(
                num.replace(",", "")
            )
        else:
            spoken_num = _normalize_number_words(
                int(num.replace(",", ""))
            )

        currency_name = {
            "AED": "UAE dirhams",
            "$": "dollars",
            "USD": "US dollars",
            "€": "euros",
            "EUR": "euros",
            "£": "pounds",
            "GBP": "British pounds",
        }.get(cur, cur)

        return f"{spoken_num} {currency_name}"

    out = _CURRENCY_RE.sub(
        currency_repl,
        out,
    )

    def percent_repl(m: "re.Match") -> str:
        num = m.group("num")

        spoken = (
            _decimal_to_spoken(num)
            if "." in num
            else _normalize_number_words(int(num))
        )

        return spoken + " percent"

    out = _PERCENT_RE.sub(
        percent_repl,
        out,
    )

    def multiplier_repl(m: "re.Match") -> str:
        num = m.group("num")

        spoken = (
            _decimal_to_spoken(num)
            if "." in num
            else _normalize_number_words(int(num))
        )

        return spoken + " times"

    out = _MULTIPLIER_RE.sub(
        multiplier_repl,
        out,
    )

    for pattern, magnitude in (
        (
            re.compile(
                r"\b(\d+(?:\.\d+)?)\s*[Kk]\b"
            ),
            "thousand",
        ),
        (
            re.compile(
                r"\b(\d+(?:\.\d+)?)\s*[Mm]\b"
            ),
            "million",
        ),
        (
            re.compile(
                r"\b(\d+(?:\.\d+)?)\s*[Bb]\b"
            ),
            "billion",
        ),
    ):

        def magnitude_repl(
            m: "re.Match",
            magnitude=magnitude,
        ) -> str:

            num = m.group(1)

            spoken = (
                _decimal_to_spoken(num)
                if "." in num
                else _normalize_number_words(int(num))
            )

            return f"{spoken} {magnitude}"

        out = pattern.sub(
            magnitude_repl,
            out,
        )

    out = _DECIMAL_RE.sub(
        lambda m: _decimal_to_spoken(
            m.group(0)
        ),
        out,
    )

    out = _LARGE_NUMBER_RE.sub(
        _number_to_spoken,
        out,
    )

    out = re.sub(
        r"\bUAE\b",
        "U.A.E.",
        out,
        flags=re.IGNORECASE,
    )

    out = re.sub(
        r"[ \t]{2,}",
        " ",
        out,
    ).strip()

    return out


# ============================================================================
# PRONUNCIATION FIXES
# ============================================================================

_PRONUNCIATION_SUBSTITUTIONS = {
    "Sania": "Saanya",
    "Aarna": "Arnah",
}

_PRONUNCIATION_RE = re.compile(
    r"\b("
    + "|".join(
        re.escape(k)
        for k in _PRONUNCIATION_SUBSTITUTIONS
    )
    + r")\b"
)


def _apply_pronunciation_fixes(
    text: str,
) -> str:

    return _PRONUNCIATION_RE.sub(
        lambda m: _PRONUNCIATION_SUBSTITUTIONS[
            m.group(1)
        ],
        text,
    )


def _prepare_tts_text(text: str) -> str:
    normalized = _normalize_for_speech(text)

    normalized = _apply_pronunciation_fixes(
        normalized
    )

    allowed = {
        "pause",
        "short pause",
        "emphasis",
        "exhale",
        "chuckle",
        "delight",
        "sigh",
    }

    def tag_filter(m: "re.Match") -> str:
        tag = m.group(1).strip().lower()

        if tag in allowed:
            return m.group(0)

        return ""

    normalized = re.sub(
        r"\[([^\]]+)\]",
        tag_filter,
        normalized,
    )

    return re.sub(
        r"\s{2,}",
        " ",
        normalized,
    ).strip()


# ============================================================================
# DETERMINISTIC CONVERSATION ROUTER
# ============================================================================

_DNC_PATTERNS = re.compile(
    r"\b("
    r"not interested"
    r"|stop"
    r"|no thanks"
    r"|don't call"
    r"|do not call"
    r"|remove me"
    r"|unsubscribe"
    r"|leave me alone"
    r"|never call"
    r"|go away"
    r"|hang up"
    r")\b",
    re.IGNORECASE,
)

_ESCALATION_PATTERNS = re.compile(
    r"\b("
    r"commission"
    r"|revenue share"
    r"|percentage"
    r"|contract"
    r"|legal"
    r"|lawyer"
    r"|privacy"
    r"|gdpr"
    r"|data protection"
    r"|complaint"
    r"|frustrated"
    r"|angry"
    r"|speak to a human"
    r"|speak to someone"
    r"|real person"
    r"|your manager"
    r"|head office"
    r"|enterprise"
    r"|multiple locations"
    r"|chain"
    r"|franchise"
    r")\b",
    re.IGNORECASE,
)

_BUSY_PATTERNS = re.compile(
    r"\b("
    r"call back"
    r"|call (?:me )?later"
    r"|i'?m busy"
    r"|at work"
    r"|in a meeting"
    r"|can'?t talk (?:now|right now)"
    r"|bad time"
    r"|not a good time"
    r"|driving"
    r")\b",
    re.IGNORECASE,
)

_WHO_IS_THIS_PATTERNS = re.compile(
    r"\b("
    r"who is this"
    r"|who am i speaking (?:with|to)"
    r"|who'?s calling"
    r"|who are you"
    r"|who is calling"
    r")\b",
    re.IGNORECASE,
)

_WHY_CALLING_PATTERNS = re.compile(
    r"\b("
    r"why are you calling"
    r"|why are you telling me this"
    r"|why r u calling"
    r"|what'?s this (?:call|about)"
    r"|what is this call (?:for|about)"
    r"|purpose of (?:this|your) call"
    r"|why this call"
    r")\b",
    re.IGNORECASE,
)

_WHAT_IS_THE_NAME_PATTERNS = re.compile(
    r"\b("
    r"what is the company name"
    r"|what'?s the company name"
    r"|what is (?:it|this|the platform|the marketplace) called"
    r"|what'?s (?:it|this|the platform|the marketplace) called"
    r"|what did you say the name was"
    r"|what is the name"
    r"|what'?s the name"
    r"|say the name again"
    r"|repeat the name"
    r")\b",
    re.IGNORECASE,
)

_SOFT_DECLINE_PATTERNS = re.compile(
    r"\b("
    r"not right now"
    r"|not currently"
    r"|not just now"
    r"|not at this time"
    r"|maybe later"
    r"|not for now"
    r"|no,? not really"
    r")\b",
    re.IGNORECASE,
)

_CONTINUATION_MARKERS = re.compile(
    r"\b("
    r"but"
    r"|however"
    r"|although"
    r"|also"
    r"|actually"
    r"|one more thing"
    r")\b",
    re.IGNORECASE,
)

_NOTHING_FURTHER_PATTERNS = re.compile(
    r"^\s*("
    r"no"
    r"|nope"
    r"|nothing"
    r"|that'?s (?:all|it)"
    r"|i'?m good"
    r"|no,? (?:that'?s|i'?m) (?:all|good|fine)"
    r"|no questions"
    r"|nothing else"
    r"|that'?s everything"
    r")\b",
    re.IGNORECASE,
)

_HOLD_REQUEST_PATTERNS = re.compile(
    r"\b("
    r"hold on"
    r"|hold for"
    r"|please hold"
    r"|one moment"
    r"|one sec"
    r"|just a (?:sec|second|moment)"
    r"|give me a (?:sec|second|moment)"
    r"|wait a (?:sec|second|moment)"
    r")\b",
    re.IGNORECASE,
)

_STILL_HERE_PATTERNS = re.compile(
    r"^\s*("
    r"yes"
    r"|yeah"
    r"|yep"
    r"|still here"
    r"|i'?m here"
    r"|here"
    r"|yes,? (?:i'?m|still) here"
    r")\b\.?\s*$",
    re.IGNORECASE,
)

_HOLD_DURATION_WORDS = {
    "a": 60,
    "one": 60,
    "1": 60,
    "two": 120,
    "2": 120,
    "a couple": 120,
    "a couple of": 120,
    "few": 180,
    "a few": 180,
    "three": 180,
    "3": 180,
}

_HOLD_DURATION_PATTERN = re.compile(
    r"\bhold(?:\s+on)?\s+for\s+"
    r"(a couple of|a couple|a few|a|one|two|three|\d+)"
    r"\s*"
    r"(second|sec|minute|min)s?\b",
    re.IGNORECASE,
)

_IDENTITY_CONFIRM_PATTERNS = re.compile(
    r"^\s*("
    r"yes"
    r"|yeah"
    r"|yep"
    r"|yup"
    r"|correct"
    r"|right"
    r"|speaking"
    r"|this is (?:him|her|me)"
    r"|that's me"
    r"|that is me"
    r"|you'?re speaking to (?:him|her|me)"
    r"|yes[, ]+(?:this is|speaking)"
    r")\b",
    re.IGNORECASE,
)

_NEGATIVE_IDENTITY_PATTERNS = re.compile(
    r"\b("
    r"no"
    r"|wrong person"
    r"|wrong number"
    r"|not me"
    r"|not him"
    r"|not her"
    r")\b",
    re.IGNORECASE,
)

_AARNA_EXPLICIT_PATTERNS = re.compile(
    r"\b("
    r"what is aarna"
    r"|what's aarna"
    r"|tell me about aarna"
    r"|tell me more about aarna"
    r"|more about aarna"
    r"|what does aarna do"
    r"|how does aarna work"
    r"|how will aarna help"
    r"|explain aarna"
    r"|what exactly is aarna"
    r"|what is this platform"
    r"|what does this platform do"
    r")\b",
    re.IGNORECASE,
)

_MONDEE_EXPLICIT_PATTERNS = re.compile(
    r"\b("
    r"what is mondee"
    r"|what's mondee"
    r"|tell me about mondee"
    r"|tell me more about mondee"
    r"|what does mondee do"
    r"|who is mondee"
    r"|what exactly is mondee"
    r")\b",
    re.IGNORECASE,
)

_BOTH_AARNA_MONDEE_PATTERNS = re.compile(
    r"\b("
    r"aarna.{0,60}mondee"
    r"|mondee.{0,60}aarna"
    r"|both aarna and mondee"
    r")\b",
    re.IGNORECASE,
)

_GENERIC_EXPLANATION_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:can|could|would)\s+you\s+"
    r"(?:please\s+)?(?:tell|explain)\b"
    r"|"
    r"\b(?:tell|explain)\s+(?:me\s+)?(?:more|about|what)\b"
    r"|"
    r"\bwhat\s+(?:is|was)\s+(?:that|this|it)\b"
    r"|"
    r"\bwhat\s+does\s+(?:that|this|it)\s+do\b"
    r"|"
    r"\bhow\s+does\s+(?:that|this|it)\s+work\b"
    r"|"
    r"\bhow\s+(?:will|would)\s+(?:that|this|it)\s+help\b"
    r"|"
    r"\bwhat\s+exactly\s+is\s+(?:that|this|it)\b"
    r"|"
    r"\b(?:so|okay|ok|well|please),?\s+"
    r"(?:like,?\s*)?"
    r"(?:can you|could you|would you|tell me|explain)\b"
    r")",
    re.IGNORECASE,
)

_EXISTING_PLATFORM_PATTERNS = re.compile(
    r"\b("
    r"already on"
    r"|getyourguide"
    r"|viator"
    r"|tripadvisor"
    r"|trip advisor"
    r"|klook"
    r"|expedia"
    r"|booking\.com"
    r"|airbnb"
    r"|already listed"
    r"|already use"
    r"|already using"
    r"|we're on"
    r"|we are on"
    r"|we use"
    r")\b",
    re.IGNORECASE,
)

_BENEFIT_QUESTION_PATTERNS = re.compile(
    r"\b("
    r"what are the benefits"
    r"|what'?s the benefit"
    r"|what benefits"
    r"|why should (?:i|we)"
    r"|what'?s in it for (?:me|us)"
    r"|how does (?:it|this|aarna) help"
    r"|how would (?:it|this|aarna) help"
    r"|what do (?:i|we) get"
    r"|why (?:aarna|use aarna|join)"
    r"|what'?s the advantage"
    r"|why partner with"
    r")\b",
    re.IGNORECASE,
)

_SHORT_CONTINUATION_PATTERNS = re.compile(
    r"^\s*("
    r"yes"
    r"|yeah"
    r"|yep"
    r"|yup"
    r"|no"
    r"|nope"
    r"|okay"
    r"|ok"
    r"|sure"
    r"|maybe"
    r"|thanks"
    r"|thank you"
    r"|great"
    r"|good"
    r"|fine"
    r"|alright"
    r"|right"
    r"|correct"
    r"|got it"
    r"|understood"
    r"|hello"
    r"|hi"
    r"|what"
    r"|huh"
    r"|sorry"
    r"|please"
    r"|go ahead"
    r"|continue"
    r")\b\.?\s*$",
    re.IGNORECASE,
)

_WH_QUESTION_STARTERS = (
    "what",
    "who",
    "why",
    "how",
    "when",
    "where",
    "is",
    "are",
    "do",
    "does",
    "can",
    "could",
    "would",
    "will",
    "tell",
    "explain",
)

_CALL_FLOW_EXCLUSION_PATTERNS = re.compile(
    r"\b("
    r"next step"
    r"|next steps"
    r"|what next"
    r"|what'?s next"
    r"|move forward"
    r"|moving forward"
    r"|how do we proceed"
    r"|what do we do now"
    r"|what happens next"
    r"|how does this work from here"
    r")\b",
    re.IGNORECASE,
)


def _extract_hold_duration_s(
    text: str,
) -> "int | None":

    match = _HOLD_DURATION_PATTERN.search(text)

    if not match:
        return None

    quantity_word = match.group(1).lower()
    unit = match.group(2).lower()

    quantity = _HOLD_DURATION_WORDS.get(
        quantity_word
    )

    if quantity is None:
        try:
            quantity = int(quantity_word)
        except ValueError:
            return None

        return quantity * (
            1 if unit.startswith("sec")
            else 60
        )

    if unit.startswith("sec"):
        seconds_map = {
            "a": 1,
            "one": 1,
            "1": 1,
            "two": 2,
            "2": 2,
            "a couple": 2,
            "a couple of": 2,
            "few": 3,
            "a few": 3,
            "three": 3,
            "3": 3,
        }

        return seconds_map.get(
            quantity_word,
            quantity,
        )

    return quantity


def _normalize_script_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        (text or "").strip().lower(),
    )


def _looks_like_a_question(
    text: str,
) -> bool:

    normalized = _normalize_script_text(
        text
    )

    if not normalized:
        return False

    if _SHORT_CONTINUATION_PATTERNS.match(
        normalized
    ):
        return False

    if _CALL_FLOW_EXCLUSION_PATTERNS.search(
        normalized
    ):
        return False

    if "?" in text:
        return True

    first_word = (
        normalized.split()[0]
        if normalized.split()
        else ""
    )

    return first_word in _WH_QUESTION_STARTERS


def _is_bare_soft_decline(
    text: str,
) -> bool:

    if not _SOFT_DECLINE_PATTERNS.search(
        text
    ):
        return False

    if _looks_like_a_question(text):
        return False

    if _CONTINUATION_MARKERS.search(text):
        return False

    return True


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

KB_SIMILARITY_THRESHOLD = float(
    os.getenv(
        "KB_SIMILARITY_THRESHOLD",
        "0.55",
    )
)

KB_EMBEDDING_MODEL_NAME = os.getenv(
    "KB_EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-en-v1.5",
)

KNOWLEDGE_BASE: list[
    tuple[str, str, str]
] = [
    (
        "kb_partner_count",
        "how many travel partners does Mondee have",
        "Mondee distributes travel content across "
        "sixty-five thousand travel partners globally.",
    ),
    (
        "kb_mondee_size",
        "how big is Mondee, how many offices does Mondee have",
        "Mondee is the third-largest travel consolidator "
        "in the U.S., with nineteen offices worldwide.",
    ),
    (
        "kb_aarna_reach",
        "how does Aarna help me get more customers or bookings",
        "Aarna gets you discovered by advisors and travelers "
        "who wouldn't otherwise have found you, through "
        "Mondee's global network.",
    ),
    (
        "kb_aarna_vs_others",
        "how is Aarna different from other platforms I'm already listed on",
        "Aarna doesn't replace those, it adds Mondee's advisor "
        "and corporate travel network specifically, which "
        "those platforms don't cover.",
    ),
]


# IMPORTANT:
# These remain None when the worker process starts.
#
# DO NOT eagerly call _load_kb_embedding_model() here.
#
# This is the key startup fix for the hosted Render deployment.
_kb_embedding_model = None
_kb_question_embeddings = None


def _load_kb_embedding_model() -> None:
    """
    Lazily load FastEmbed.

    Failure is deliberately non-fatal. If FastEmbed cannot be loaded,
    the normal LLM path remains available.
    """

    global _kb_embedding_model
    global _kb_question_embeddings

    if _kb_embedding_model is not None:
        return

    try:
        from fastembed import TextEmbedding

        t_start = time.monotonic()

        _kb_embedding_model = TextEmbedding(
            model_name=KB_EMBEDDING_MODEL_NAME
        )

        questions = [
            question
            for _, question, _ in KNOWLEDGE_BASE
        ]

        _kb_question_embeddings = list(
            _kb_embedding_model.embed(
                questions
            )
        )

        logger.info(
            "KB embedding model loaded and %d KB entries "
            "embedded in %.0fms",
            len(KNOWLEDGE_BASE),
            (
                time.monotonic()
                - t_start
            )
            * 1000,
        )

    except Exception as exc:
        logger.warning(
            "KB embedding model failed to load (%s) — "
            "KB lookup disabled for this process; "
            "question-like utterances will use the "
            "normal LLM flow instead.",
            exc,
        )

        _kb_embedding_model = None
        _kb_question_embeddings = None


def _cosine_similarity(
    a,
    b,
) -> float:

    import numpy as np

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def _kb_available() -> bool:
    return (
        _kb_embedding_model is not None
        and bool(KNOWLEDGE_BASE)
        and _kb_question_embeddings is not None
    )


def _kb_lookup(
    text: str,
) -> "tuple[str, str, float] | None":

    # LAZY LOAD.
    #
    # This is intentionally inside the request path rather than
    # module initialization so FastEmbed cannot block LiveKit's
    # worker process from becoming warm.
    if not _kb_available():
        _load_kb_embedding_model()

    if not _kb_available():
        return None

    try:
        query_vec = list(
            _kb_embedding_model.embed(
                [text]
            )
        )[0]

        scored = [
            (
                KNOWLEDGE_BASE[i][0],
                KNOWLEDGE_BASE[i][2],
                _cosine_similarity(
                    query_vec,
                    _kb_question_embeddings[i],
                ),
            )
            for i in range(
                len(KNOWLEDGE_BASE)
            )
        ]

        best_key, best_answer, best_score = max(
            scored,
            key=lambda x: x[2],
        )

        if best_score >= KB_SIMILARITY_THRESHOLD:
            return (
                best_key,
                best_answer,
                best_score,
            )

    except Exception as exc:
        logger.warning(
            "KB lookup failed: %s",
            exc,
        )

    return None


# ============================================================================
# APPROVED SCRIPT
# ============================================================================

AARNA_INTRO_TEMPLATE = (
    "We've recently launched Aarna, our experiences "
    "marketplace here in the UAE, and {partner} came up "
    "as exactly the kind of partner we'd love to work with. "
    "Is there anything specific you want to know?"
)

AARNA_EXPLANATION = (
    "aarna is Mondee's experiences marketplace, and the idea "
    "is to explore whether your experiences can reach travellers "
    "through Mondee's wider distribution network."
)

MONDEE_EXPLANATION = (
    "Mondee is the third-largest travel consolidator in the "
    "United States and has nineteen offices worldwide. Through "
    "our proprietary Mondee Marketplace technology, we distribute "
    "air tickets and hotel accommodation across sixty-five "
    "thousand travel partners globally."
)

AARNA_AND_MONDEE_EXPLANATION = (
    MONDEE_EXPLANATION
    + " "
    + AARNA_EXPLANATION
)

AARNA_BENEFIT = (
    "Aarna doesn't replace those, it adds Mondee's advisor "
    "and corporate travel network specifically, which those "
    "platforms don't cover."
)

AARNA_GENERAL_BENEFIT = (
    "Aarna gives you direct access to Mondee's global advisor "
    "and corporate travel network, so new travelers and advisors "
    "who wouldn't otherwise find you can discover and book your "
    "experiences directly."
)

MEETING_SCHEDULE_FOLLOWUP = (
    "Would you like to schedule a brief call with our partnership "
    "team to go over this in more detail?"
)

WHO_IS_THIS_REPLY = (
    "Of course — I am Sania, Mondee's AI voice assistant."
)

WHAT_IS_THE_NAME_REPLY = (
    "It's called Aarna — A, A, R, N, A."
)

WHY_CALLING_TEMPLATE = (
    "I'm calling because we believe {partner} could be relevant "
    "for Aarna, our experiences marketplace — my purpose is to "
    "share some information and see if an introductory "
    "conversation would be of interest."
)

ESCALATION_DEFLECTION_REPLY = (
    "That is a good question. Our partnerships team can explain "
    "that accurately during the introductory call."
)

HARD_DECLINE_REPLY = (
    "Understood. We will not contact you again. Goodbye."
)

SOFT_DECLINE_CHECK_REPLY = (
    "Understood. Is there anything you would like to know "
    "before we close?"
)

SOFT_DECLINE_CLOSING_REPLY = (
    "Understood. Thank you for your time. Have a wonderful day."
)

BUSY_RECIPIENT_REPLY = (
    "No problem — would you prefer we call you back at a better "
    "time, or is now still okay for a quick word?"
)

AARNA_OFF_TOPIC_REDIRECT_TEMPLATE = (
    "I'm here to help with Aarna and our partnership with "
    "{partner} — let's stick to that so I can make the best "
    "use of your time."
)

_QUESTION_ANSWER_KEYS = {
    "aarna_explanation",
    "aarna_and_mondee",
    "mondee_explanation",
    "aarna_benefit",
    "aarna_general_benefit",
    "why_calling",
} | {
    kb_key
    for kb_key, _, _
    in KNOWLEDGE_BASE
}


# ============================================================================
# HOLD SETTINGS
# ============================================================================

HOLD_FIRST_CHECK_S = int(
    os.getenv(
        "HOLD_FIRST_CHECK_S",
        "5",
    )
)

HOLD_SECOND_CHECK_S = int(
    os.getenv(
        "HOLD_SECOND_CHECK_S",
        "10",
    )
)

HOLD_FINAL_WAIT_S = int(
    os.getenv(
        "HOLD_FINAL_WAIT_S",
        "10",
    )
)


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

def _build_prompt(
    partner_name: str,
    contact_name: str,
    category: str,
    company_synopsis: str,
    digitisation: str,
) -> str:

    partner = (
        partner_name
        or "your business"
    )

    who = (
        f"{contact_name} from {partner}"
        if contact_name
        else partner
    )

    if digitisation == "digitised":
        tone = (
            "Professional partnership tone — this supplier "
            "understands B2B language."
        )

    elif digitisation == "hyperlocal":
        tone = (
            "Highly conversational, human tone — avoid all "
            "tech/corporate language."
        )

    else:
        tone = (
            "Friendly, simple business-expansion tone."
        )

    context_lines = [
        f"partner_name={partner}"
    ]

    if category:
        context_lines.append(
            f"category={category}"
        )

    if company_synopsis:
        context_lines.append(
            f"company_synopsis={company_synopsis}"
        )

    context_block = "\n".join(
        context_lines
    )

    return f"""
You are Sania, an AI voice assistant calling on behalf of Mondee about a potential Aarna partnership with {who}. You are not a human, a general-information assistant, a negotiator or a legal adviser. Never deny or hide that you are an AI assistant.

LIVE CALL CONTEXT
{context_block}

RULES:
- The conversation always takes priority over script progression. Always answer the supplier's actual question or statement before moving to the next step. Never repeat a point already delivered. Never ask for information already given.
- Respond to the supplier's latest words. Never invent missing facts.
- Max 2 spoken sentences per normal turn and one question per turn.
- If the supplier's speech seems incomplete, trails off mid-thought, or contains a continuation word like "but", "however", "although", "also", "actually", "one more thing", "is there" or "what about", do not treat it as a completed statement — say only "Of course — please go ahead." and wait.
- A reply starting with "no" is not automatically a refusal if it continues with "but", a question, or more information; respond to the complete meaning, not just the first word.
- If you were just interrupted or your previous reply was cut off, do not assume the caller heard the complete response. Answer what they just said first.
- If unclear/garbled, ask "Sorry, could you say that again?" rather than guessing.
- Never invent or imply client names, bookings, revenue, supplier performance, pricing, commissions, fees, revenue share, contract terms, onboarding terms, launch dates, guarantees, or unsupported capabilities; defer those to the partnership lead.
- A plain date/time answer gets a direct confirmation, never the unknown-fact deferral.
- Do not repeat a question just because of a short pause; only repeat it after a genuine no-response, and then only once, in shorter wording.
- Use a farewell only when actually ending the call.
- Polite English, no slang, no pressure. {tone}

FLOW:
1. After identity confirmation, move immediately into the main body. NEVER ask whether they are the owner, decision maker, or whether it is a good time — that was already asked in the opening greeting.
2. The Aarna introduction, known Aarna/Mondee/existing-platform questions, decline handling, escalation, hold requests, and off-topic questions are handled outside this prompt with fixed approved wording.
3. Your job is to handle genuinely novel questions or statements naturally, then offer and schedule the human partnerships call — capture and confirm a concrete date/time, then close warmly: "Perfect, I'll send a calendar invite to your email. Have a wonderful day!"

VOICE TAGS:
- Apply every time the trigger genuinely occurs, max ONE tag per turn.
- Turn opens with "Great" or "Perfect" / confirming something positive: [delight] after that word with a space.
- About to ask the key scheduling question, or supplier just raised real pushback: [short pause] to open.
- One word carries real weight: [emphasis] on that word only.
- Never use theatrical tags such as [laughing], [excited], [whisper], [screaming], [moaning], or [singing].
""".strip()


# ============================================================================
# SANIA AGENT
# ============================================================================

class Sania(Agent):

    _KNOWN_PLATFORMS = (
        "Viator",
        "GetYourGuide",
        "TripAdvisor",
        "Klook",
        "Expedia",
        "Booking.com",
        "Airbnb",
    )

    def __init__(
        self,
        instructions: str,
        ctx: JobContext,
        partner_name: str,
    ):
        super().__init__(
            instructions=instructions
        )

        self._ctx = ctx

        self._partner_name = (
            partner_name
            or "partners like you"
        )

        self.call_state = "IDENTITY"

        self.conversation_facts = {
            "existing_platforms": []
        }

        self.script_segments_spoken = set()

        self._answered_question_count = 0

        self._soft_decline_check_pending = False

        self._on_hold = False

        self._hold_awaiting_checkin_response = False

        self._hold_check_task = None

        self.dnc_requested = False

        self.escalation_requested = False

        self._call_ended = False

        # Deterministic script replies must never be cut by
        # the normal LLM sentence cap.
        self._suppress_sentence_cap = False

    # ---------------------------------------------------------------------
    # DETERMINISTIC SCRIPT ROUTER
    # ---------------------------------------------------------------------

    def _script_reply_for_utterance(
        self,
        text: str,
    ) -> "tuple[str, str] | None":

        state = self.call_state

        normalized = _normalize_script_text(
            text
        )

        partner = self._partner_name

        if _WHO_IS_THIS_PATTERNS.search(
            normalized
        ):
            return (
                "who_is_this",
                WHO_IS_THIS_REPLY,
            )

        if _WHAT_IS_THE_NAME_PATTERNS.search(
            normalized
        ):
            return (
                "what_is_the_name",
                WHAT_IS_THE_NAME_REPLY,
            )

        if _WHY_CALLING_PATTERNS.search(
            normalized
        ):
            return (
                "why_calling",
                WHY_CALLING_TEMPLATE.format(
                    partner=partner
                ),
            )

        if state == "IDENTITY":

            if _NEGATIVE_IDENTITY_PATTERNS.search(
                normalized
            ):
                return None

            if (
                _IDENTITY_CONFIRM_PATTERNS.search(
                    normalized
                )
                or normalized in {
                    "yes",
                    "yeah",
                    "yep",
                    "speaking",
                }
            ):
                return (
                    "aarna_intro",
                    AARNA_INTRO_TEMPLATE.format(
                        partner=partner
                    ),
                )

        if _BOTH_AARNA_MONDEE_PATTERNS.search(
            normalized
        ):
            return (
                "aarna_and_mondee",
                AARNA_AND_MONDEE_EXPLANATION,
            )

        if _MONDEE_EXPLICIT_PATTERNS.search(
            normalized
        ):
            return (
                "mondee_explanation",
                MONDEE_EXPLANATION,
            )

        if _AARNA_EXPLICIT_PATTERNS.search(
            normalized
        ):
            return (
                "aarna_explanation",
                AARNA_EXPLANATION,
            )

        if (
            state == "AFTER_AARNA_INTRO"
            and _GENERIC_EXPLANATION_PATTERNS.search(
                normalized
            )
        ):
            return (
                "aarna_explanation",
                AARNA_EXPLANATION,
            )

        if (
            _EXISTING_PLATFORM_PATTERNS.search(
                normalized
            )
            or _BENEFIT_QUESTION_PATTERNS.search(
                normalized
            )
        ):

            if self.conversation_facts.get(
                "existing_platforms"
            ):
                return (
                    "aarna_benefit",
                    AARNA_BENEFIT,
                )

            return (
                "aarna_general_benefit",
                AARNA_GENERAL_BENEFIT,
            )

        return None

    # ---------------------------------------------------------------------
    # FIXED SCRIPT SPEECH
    # ---------------------------------------------------------------------

    async def _say_script(
        self,
        text: str,
        allow_interruptions: bool = True,
    ) -> bool:

        self._suppress_sentence_cap = True

        try:

            handle = self.session.say(
                _prepare_tts_text(text),
                allow_interruptions=allow_interruptions,
                add_to_chat_ctx=True,
            )

            await handle.wait_for_playout()

            return not handle.interrupted

        finally:
            self._suppress_sentence_cap = False

    async def _speak_script(
        self,
        key: str,
        text: str,
    ) -> None:

        self.script_segments_spoken.add(
            key
        )

        if key == "aarna_intro":
            self.call_state = (
                "AFTER_AARNA_INTRO"
            )

        elif key in {
            "aarna_explanation",
            "aarna_and_mondee",
            "mondee_explanation",
            "aarna_benefit",
            "aarna_general_benefit",
        }:
            self.call_state = (
                "POST_EXPLANATION"
            )

        logger.info(
            "SCRIPT FAST PATH key=%s — LLM bypassed",
            key,
        )

        completed = await self._say_script(
            text
        )

        if completed:
            await self._maybe_speak_meeting_followup(
                key
            )

    async def _maybe_speak_meeting_followup(
        self,
        answer_key: str,
    ) -> None:

        if answer_key not in _QUESTION_ANSWER_KEYS:
            return

        self._answered_question_count += 1

        if self._answered_question_count < 2:
            return

        logger.info(
            "answered question #%d this call, "
            "attaching meeting-schedule follow-up",
            self._answered_question_count,
        )

        await self._say_script(
            MEETING_SCHEDULE_FOLLOWUP
        )

    # ---------------------------------------------------------------------
    # END CALL
    # ---------------------------------------------------------------------

    async def _end_call(
        self,
        farewell_text: str,
    ) -> None:

        self._call_ended = True

        await self._say_script(
            farewell_text,
            allow_interruptions=False,
        )

        await asyncio.sleep(1.5)

        self._ctx.shutdown(
            reason="Call ended by Sania"
        )

    # ---------------------------------------------------------------------
    # HOLD HANDLING
    # ---------------------------------------------------------------------

    async def _hold_check_loop(
        self,
        duration_s: "int | None",
    ) -> None:

        first_wait_s = (
            duration_s
            if duration_s is not None
            else HOLD_FIRST_CHECK_S
        )

        try:

            await asyncio.sleep(
                first_wait_s
            )

            if not self._on_hold:
                return

            logger.info(
                "hold check #1 firing after %.0fs",
                first_wait_s,
            )

            self._hold_awaiting_checkin_response = True

            await self._say_script(
                "Are we still connected?"
            )

            await asyncio.sleep(
                HOLD_SECOND_CHECK_S
            )

            if not self._on_hold:
                return

            logger.info(
                "hold check #2 firing, "
                "no response to check #1"
            )

            await self._say_script(
                "Do you need more time, "
                "or would a callback be easier?"
            )

            await asyncio.sleep(
                HOLD_FINAL_WAIT_S
            )

            if not self._on_hold:
                return

            logger.info(
                "no response after hold checks, "
                "ending call"
            )

            self._on_hold = False

            await self._end_call(
                "It seems this may not be a convenient "
                "time. We can try again later. Goodbye."
            )

        except asyncio.CancelledError:
            pass

    # ---------------------------------------------------------------------
    # USER TURN ROUTER
    # ---------------------------------------------------------------------

    async def on_user_turn_completed(
        self,
        turn_ctx: "ChatContext",
        new_message: "ChatMessage",
    ) -> None:

        text = (
            new_message.text_content
            or ""
        ).strip()

        if not text:
            return

        normalized = _normalize_script_text(
            text
        )

        facts = self.conversation_facts

        # Track known platforms mentioned by the supplier.
        for platform in self._KNOWN_PLATFORMS:

            if re.search(
                rf"\b{re.escape(platform)}\b",
                text,
                re.IGNORECASE,
            ):

                if platform not in facts.setdefault(
                    "existing_platforms",
                    [],
                ):
                    facts[
                        "existing_platforms"
                    ].append(platform)

        # ---------------------------------------------------------------
        # DNC
        # ---------------------------------------------------------------

        if _DNC_PATTERNS.search(text):

            self.dnc_requested = True

            logger.info(
                "DNC signal detected — ending immediately"
            )

            await self._end_call(
                HARD_DECLINE_REPLY
            )

            raise StopResponse()

        # ---------------------------------------------------------------
        # HOLD
        # ---------------------------------------------------------------

        if _HOLD_REQUEST_PATTERNS.search(text):

            duration_s = _extract_hold_duration_s(
                text
            )

            duration_match = (
                _HOLD_DURATION_PATTERN.search(
                    text
                )
            )

            if (
                duration_s is not None
                and duration_match
            ):
                hold_ack_reply = (
                    "Of course, I can hold for "
                    + duration_match.group(0)
                    .split("for", 1)[-1]
                    .strip()
                    + "."
                )
            else:
                hold_ack_reply = (
                    "Of course, I can hold."
                )

            logger.info(
                "hold request detected "
                "(duration=%s)",
                duration_s,
            )

            await self._say_script(
                hold_ack_reply
            )

            self._on_hold = True

            self._hold_check_task = (
                asyncio.create_task(
                    self._hold_check_loop(
                        duration_s
                    )
                )
            )

            raise StopResponse()

        # ---------------------------------------------------------------
        # RETURN FROM HOLD
        # ---------------------------------------------------------------

        if self._on_hold:

            self._on_hold = False

            was_awaiting_checkin = (
                self._hold_awaiting_checkin_response
            )

            self._hold_awaiting_checkin_response = False

            if (
                self._hold_check_task is not None
                and not self._hold_check_task.done()
            ):
                self._hold_check_task.cancel()

            if (
                was_awaiting_checkin
                and _STILL_HERE_PATTERNS.match(
                    normalized
                )
            ):

                logger.info(
                    "caller confirmed still connected"
                )

                await self._say_script(
                    "Would you like me to continue?"
                )

                raise StopResponse()

            logger.info(
                "caller returned from hold with "
                "real content, resuming normally"
            )

        # ---------------------------------------------------------------
        # SOFT DECLINE CLOSING CHECK
        # ---------------------------------------------------------------

        if self._soft_decline_check_pending:

            self._soft_decline_check_pending = False

            if _NOTHING_FURTHER_PATTERNS.match(
                normalized
            ):

                logger.info(
                    "soft-decline closing confirmed"
                )

                await self._end_call(
                    SOFT_DECLINE_CLOSING_REPLY
                )

                raise StopResponse()

        # ---------------------------------------------------------------
        # BUSY
        # ---------------------------------------------------------------

        if _BUSY_PATTERNS.search(text):

            logger.info(
                "busy recipient signal detected"
            )

            await self._say_script(
                BUSY_RECIPIENT_REPLY
            )

            raise StopResponse()

        # ---------------------------------------------------------------
        # SOFT DECLINE
        # ---------------------------------------------------------------

        if _is_bare_soft_decline(text):

            logger.info(
                "bare soft decline, "
                "asking closing-check question"
            )

            await self._say_script(
                SOFT_DECLINE_CHECK_REPLY
            )

            self._soft_decline_check_pending = True

            raise StopResponse()

        # ---------------------------------------------------------------
        # DETERMINISTIC SCRIPT FAST PATH
        # ---------------------------------------------------------------

        fast_script = (
            self._script_reply_for_utterance(
                text
            )
        )

        if fast_script:

            script_key, script_text = (
                fast_script
            )

            await self._speak_script(
                script_key,
                script_text,
            )

            raise StopResponse()

        # ---------------------------------------------------------------
        # ESCALATION
        # ---------------------------------------------------------------

        if _ESCALATION_PATTERNS.search(text):

            self.escalation_requested = True

            logger.info(
                "escalation trigger detected"
            )

            await self._say_script(
                ESCALATION_DEFLECTION_REPLY
            )

            raise StopResponse()

        # ---------------------------------------------------------------
        # KNOWLEDGE BASE
        #
        # IMPORTANT:
        # Do NOT gate this on _kb_available().
        #
        # _kb_lookup() is responsible for lazy-loading FastEmbed.
        # ---------------------------------------------------------------

        if _looks_like_a_question(text):

            kb_match = _kb_lookup(text)

            if kb_match:

                (
                    kb_key,
                    kb_answer,
                    kb_score,
                ) = kb_match

                logger.info(
                    "KB LOOKUP hit key=%s "
                    "score=%.3f "
                    "(threshold=%.2f): %r",
                    kb_key,
                    kb_score,
                    KB_SIMILARITY_THRESHOLD,
                    text[:60],
                )

                await self._speak_script(
                    kb_key,
                    kb_answer,
                )

                raise StopResponse()

            logger.info(
                "question-like utterance, "
                "no confident KB match, "
                "using fixed redirect: %r",
                text[:60],
            )

            redirect_text = (
                AARNA_OFF_TOPIC_REDIRECT_TEMPLATE
                .format(
                    partner=self._partner_name
                )
            )

            await self._say_script(
                redirect_text
            )

            raise StopResponse()

        # ---------------------------------------------------------------
        # NORMAL LLM FLOW
        # ---------------------------------------------------------------

        return

    # ---------------------------------------------------------------------
    # TTS NODE
    # ---------------------------------------------------------------------

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:

        _sentence_end_re = re.compile(
            r"(?<=[.!?])\s+"
        )

        suppress_cap = (
            self._suppress_sentence_cap
        )

        async def _prepared():
            buffer = ""
            sentence_count = 0

            async for chunk in text:

                buffer += chunk

                parts = _sentence_end_re.split(
                    buffer
                )

                if len(parts) > 1:

                    for sentence in parts[:-1]:

                        sentence = sentence.strip()

                        if not sentence:
                            continue

                        sentence_count += 1

                        yield _prepare_tts_text(
                            sentence
                        )

                        if (
                            not suppress_cap
                            and sentence_count
                            >= MAX_SPOKEN_SENTENCES_PER_TURN
                        ):

                            logger.info(
                                "reached "
                                "MAX_SPOKEN_SENTENCES_PER_TURN=%d",
                                MAX_SPOKEN_SENTENCES_PER_TURN,
                            )

                            return

                    buffer = parts[-1]

            remainder = buffer.strip()

            if remainder:
                yield _prepare_tts_text(
                    remainder
                )

        return Agent.default.tts_node(
            self,
            _prepared(),
            model_settings,
        )


# ============================================================================
# LIVEKIT ENTRYPOINT
# ============================================================================

async def entrypoint(
    ctx: JobContext,
) -> None:

    # ---------------------------------------------------------------------
    # JOB METADATA
    # ---------------------------------------------------------------------

    metadata = {}

    if ctx.job.metadata:

        try:
            metadata = json.loads(
                ctx.job.metadata
            )

        except json.JSONDecodeError:

            logger.warning(
                "Job metadata wasn't valid JSON: %r",
                ctx.job.metadata,
            )

    partner_name = metadata.get(
        "partner_name",
        "",
    )

    contact_name = metadata.get(
        "contact_name",
        "",
    )

    category = metadata.get(
        "category",
        "",
    )

    company_synopsis = metadata.get(
        "company_synopsis",
        "",
    )

    digitisation = metadata.get(
        "digitisation",
        "semi",
    )

    logger.info(
        "=================================================="
    )

    logger.info(
        "JOB STARTED"
    )

    logger.info(
        "room=%s",
        ctx.room.name,
    )

    logger.info(
        "partner=%r",
        partner_name,
    )

    logger.info(
        "contact=%r",
        contact_name,
    )

    logger.info(
        "category=%r",
        category,
    )

    logger.info(
        "=================================================="
    )

    # ---------------------------------------------------------------------
    # CONNECT TO ROOM
    # ---------------------------------------------------------------------

    await ctx.connect()

    logger.info(
        "LiveKit room connected: %s",
        ctx.room.name,
    )

    # ---------------------------------------------------------------------
    # DEEPGRAM KEYTERMS
    # ---------------------------------------------------------------------

    keyterms = list(
        _STATIC_KEYTERMS
    )

    for name in (
        partner_name,
        contact_name,
        category,
    ):

        if name:
            keyterms.append(name)

    # ---------------------------------------------------------------------
    # AGENT SESSION
    # ---------------------------------------------------------------------

    logger.info(
        "Creating AgentSession..."
    )

    session = AgentSession(

        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            api_key=DEEPGRAM_API_KEY,
            keyterms=keyterms,
        ),

        llm=openai.LLM(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url=(
                "https://api.groq.com/openai/v1"
            ),
            temperature=0.4,
            extra_body={
                "reasoning_effort":
                    GROQ_REASONING_EFFORT
            },
        ),

        tts=fishaudio.TTS(
            api_key=FISH_AUDIO_API_KEY,
            model=FISH_AUDIO_MODEL,
            voice_id=(
                FISH_AUDIO_VOICE_ID
                or None
            ),
            speed=FISH_AUDIO_SPEED,
            sample_rate=24000,
        ),

        vad=silero.VAD.load(),
    )

    # ---------------------------------------------------------------------
    # SANIA AGENT
    # ---------------------------------------------------------------------

    agent = Sania(
        instructions=_build_prompt(
            partner_name,
            contact_name,
            category,
            company_synopsis,
            digitisation,
        ),
        ctx=ctx,
        partner_name=partner_name,
    )

    # ---------------------------------------------------------------------
    # FAREWELL DETECTION FOR LLM-GENERATED CLOSINGS
    # ---------------------------------------------------------------------

    @session.on(
        "conversation_item_added"
    )
    def _on_conversation_item_added(
        event,
    ) -> None:

        item = getattr(
            event,
            "item",
            None,
        )

        if item is None:
            return

        if getattr(
            item,
            "role",
            None,
        ) != "assistant":
            return

        if agent._call_ended:
            return

        reply_text = (
            getattr(
                item,
                "text_content",
                "",
            )
            or ""
        )

        if _FAREWELL_PATTERNS.search(
            reply_text
        ):

            logger.info(
                "LLM-generated farewell detected — "
                "ending call"
            )

            agent._call_ended = True

            asyncio.create_task(
                _end_after_farewell(ctx)
            )

    async def _end_after_farewell(
        job_ctx: JobContext,
    ) -> None:

        # Give Fish Audio time to finish.
        await asyncio.sleep(2)

        job_ctx.shutdown(
            reason="Call ended after farewell"
        )

    # ---------------------------------------------------------------------
    # START AGENT SESSION
    # ---------------------------------------------------------------------

    logger.info(
        "Starting AgentSession..."
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
    )

    logger.info(
        "AgentSession started successfully."
    )

    # ---------------------------------------------------------------------
    # WAIT FOR BROWSER PARTICIPANT
    # ---------------------------------------------------------------------

    logger.info(
        "Waiting for a participant to join room=%s ...",
        ctx.room.name,
    )

    participant = (
        await ctx.wait_for_participant()
    )

    logger.info(
        "Participant joined: identity=%s",
        participant.identity,
    )

    logger.info(
        "Starting Sania conversation..."
    )

    # ---------------------------------------------------------------------
    # OPENING GREETING
    # ---------------------------------------------------------------------

    agent._suppress_sentence_cap = True

    try:

        if (
            contact_name
            and partner_name
        ):

            who_clause = (
                f"you're speaking with "
                f"{contact_name} from "
                f"{partner_name}"
            )

        elif partner_name:

            who_clause = (
                "you're speaking with "
                "the right contact from "
                f"{partner_name}"
            )

        else:

            who_clause = (
                "it's a good time to talk"
            )

        await session.generate_reply(
            instructions=(
                "Greet the person now: say hello, "
                "identify yourself as Sania, an AI "
                "assistant from Aarna, a travel "
                "management platform, and confirm "
                f"{who_clause}. "

                "Never say a bracketed placeholder "
                "or a variable name out loud — if "
                "you don't have a name, just phrase "
                "around it naturally. "

                "Keep it to two sentences."
            )
        )

    finally:

        agent._suppress_sentence_cap = False

    logger.info(
        "Opening greeting generated."
    )


# ============================================================================
# WORKER STARTUP
# ============================================================================

if __name__ == "__main__":

    cli.run_app(
        WorkerOptions(

            entrypoint_fnc=entrypoint,

            # IMPORTANT:
            # app.py dispatches each browser session explicitly
            # using this same agent name.
            agent_name=os.getenv(
                "LIVEKIT_LAPTOP_AGENT_NAME",
                "aarna-sania-laptop-test",
            ),

            # Keep one warm worker process on Render.
            num_idle_processes=1,

            # Give LiveKit enough time for process initialization.
            # FastEmbed is no longer part of this startup path.
            initialize_process_timeout=120,
        )
    )
