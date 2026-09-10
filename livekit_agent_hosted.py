"""
voice_agent/livekit_agent_laptop.py
------------------------------------
EXPERIMENT — laptop-to-laptop test variant of livekit_agent.py.

This is a COPY of livekit_agent.py with ONE change: the entrypoint no
longer dials out over a Twilio SIP trunk. Instead, it waits for a human
to join the SAME LiveKit room directly over WebRTC (e.g. from a browser
tab on your laptop, or a teammate's), and starts the conversation once
they do. Every other line — STT/LLM/TTS setup, the Sania agent class,
the text-preparation pipeline (number normalization, Sania/Aarna
pronunciation fixes, voice delivery tags), the system prompt — is
UNCHANGED from livekit_agent.py. See that file's own docstring for the
full stack details and the unverified-parameter caveats (Fish Audio
constructor kwargs, GROQ_REASONING_EFFORT pass-through) — both apply
here identically, since this is the same session-setup code.

WHAT THIS ACTUALLY PROVES: whether the conversational pipeline (STT,
LLM, TTS, barge-in, text prep) works at all over LiveKit's WebRTC
transport, with ZERO Twilio involvement — no phone number, no PSTN, no
per-minute telephony cost. Deepgram/Groq/Fish Audio usage still has its
own separate cost; this experiment only removes Twilio specifically,
exactly as asked.

NOT changing engine.py or the production Twilio pipeline in any way —
this is a standalone side experiment, per explicit instruction.

HOW TO RUN THIS EXPERIMENT (two terminals + one browser tab):

  Terminal 1 — start the worker, leave it running:
    python voice_agent/livekit_agent_laptop.py dev

  Terminal 2 — create a room and dispatch this agent into it, and
  generate a join token for yourself:
    python voice_agent/livekit_laptop_test.py "Test Partner"

  That second command prints a room name and an access token. Open
  livekit_laptop_client.html (see that file) in a browser, paste in the
  printed token, click Connect. Allow microphone access. Sania will
  greet you once you've joined the room.

STACK DETAILS (unchanged from livekit_agent.py):
  STT: Deepgram Nova-3          (same as engine.py's _build_deepgram_url,
                                  same keyterm boosting via the plugin's
                                  native `keyterms` param)
  LLM: Groq openai/gpt-oss-120b (via the OpenAI-compatible plugin,
                                  base_url pointed at Groq — same model,
                                  same API key, same .env var)
  TTS: Fish Audio                (FIX — this file previously ran Deepgram
                                  Aura-2, which engine.py has since moved
                                  PAST: real calls measured 4.5-4.8s
                                  synthesis time, far outside Deepgram's
                                  own documented worst case, and it was
                                  reverted in favour of Fish Audio, which
                                  engine.py has since tuned extensively —
                                  same voice_id, same speed, same text
                                  preparation pipeline. Running Aura-2 here
                                  meant this file was NOT actually "the
                                  same stack" despite the docstring saying
                                  so — this update makes that true.)
  VAD: Silero (local, free — LiveKit's turn-detection uses this alongside
       Deepgram's own endpointing). Deliberately NOT porting engine.py's
       hand-tuned energy-based RMS barge-in system (BARGE_IN_RMS_THRESHOLD,
       hangover frames, etc.) — that system exists specifically because
       Twilio's raw Media Streams give no native turn-detection at all.
       LiveKit's Silero VAD + AgentSession already provides real,
       production-grade turn detection out of the box; reimplementing the
       Twilio-specific workaround here would be strictly worse, not
       "matching the stack."

WHY THIS IS SIMPLER THAN engine.py's PIPELINE: LiveKit's AgentSession
handles audio transport, resampling, barge-in/interruption, and turn
detection itself. There is no mulaw chunking, no manual VAD/RMS
threshold tuning, no Twilio 'clear' message, no hand-written recorder —
all of that (which is a large fraction of engine.py) is infrastructure
LiveKit already provides. This file is the conversation logic only.

TEXT PREPARATION — FIX, ported verbatim from engine.py's validated
pipeline via the tts_node() override hook (confirmed real, documented
LiveKit API: "Modify LLM output before sending it to TTS to customize
pronunciation" — literally this use case):
  - Number normalization: "65,000" -> "sixty-five thousand". Confirmed
    real bug on the Twilio side: raw digits get read back character by
    character ("six five zero zero zero"), not as a number.
  - Pronunciation fixes: "Sania" -> "Saanya", "Aarna" -> "Arnah" —
    confirmed real bug: both got mangled by TTS as unfamiliar proper
    nouns ("Sania" heard as "Senia", "Aarna" heard as "Ana", losing the R
    entirely). Same substitution dict as engine.py — if you correct one,
    correct both files.
  - Voice delivery tags: same restrained, trigger-based tag system as
    engine.py — [delight] on "Great"/"Perfect", [short pause] before key
    questions, still excluding theatrical tags. A confirmed real call on
    the Twilio side used these tags ZERO times until the instruction was
    made concrete rather than optional — same wording carried over here.

UNVERIFIED — CHECK BEFORE FIRST RUN: the exact constructor kwargs for
`fishaudio.TTS(...)` below (api_key/model/voice_id/prosody-speed param
names) are my best-informed match to Fish Audio's own REST API
terminology (confirmed via engine.py's tested REST integration), NOT
independently verified against the installed `livekit-plugins-fishaudio`
package's actual signature — I don't have that package available to
import and introspect from here. Before your first real test call, run
`python -c "from livekit.plugins import fishaudio; help(fishaudio.TTS)"`
and fix any parameter name mismatch — it will fail loudly and immediately
at startup if wrong, not silently, so this is safe to try. Same caveat
for GROQ_REASONING_EFFORT below — passed via extra_kwargs on a best-
effort basis; verify it actually reaches Groq rather than being silently
dropped by checking Groq's own request logs for the reasoning_effort field.

FIRST-TIME SETUP:
  pip install "livekit-agents[deepgram,openai,fishaudio,silero]~=1.0" livekit-api python-dotenv
  python voice_agent/livekit_agent_laptop.py download-files   # fetches the local Silero VAD model
  python voice_agent/livekit_agent_laptop.py dev               # starts this worker, leave running
  python voice_agent/livekit_laptop_test.py "Test Partner"     # in a second terminal
  (then open livekit_laptop_client.html in a browser — see that file)
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
    Agent, AgentSession, ChatContext, ChatMessage, JobContext,
    RoomInputOptions, StopResponse, WorkerOptions, cli,
)
from livekit.agents.voice import ModelSettings
from livekit.plugins import deepgram, fishaudio, openai, silero

load_dotenv()

logger = logging.getLogger("livekit_agent")
logging.basicConfig(level=logging.INFO)

# ── Same env vars engine.py already uses — no new credentials needed ──────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# FIX — same reasoning_effort fix as engine.py: gpt-oss-120b is a
# reasoning model that can burn many seconds on invisible chain-of-thought
# between visible sentences. Confirmed across multiple real calls on the
# Twilio side (gaps of 9-19+ seconds mid-reply). "low" is Groq/OpenAI's
# own documented setting for "fast responses for general dialogue" — this
# exact use case. See engine.py's GROQ_REASONING_EFFORT comment for the
# full evidence trail.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# FIX — TTS swapped from Deepgram Aura-2 to Fish Audio, matching what
# engine.py actually runs now (see module docstring for why Aura-2 was
# dropped). Same env vars, same defaults as engine.py — one .env file
# configures both agents identically.
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")
FISH_AUDIO_MODEL = os.getenv("FISH_AUDIO_MODEL", "s2.1-pro-free")
FISH_AUDIO_VOICE_ID = os.getenv("FISH_AUDIO_VOICE_ID", "")
FISH_AUDIO_SPEED = float(os.getenv("FISH_AUDIO_SPEED", "1.15"))

# FIX — mechanical enforcement of the prompt's own "max 2 sentences per
# turn, no exceptions" rule — matching engine.py's
# MAX_SPOKEN_SENTENCES_PER_TURN pattern exactly. Confirmed real gap this
# closes: a live test showed a 3-sentence reply (an unscripted "Great,
# thank you." prepended ahead of the already-2-sentence scripted pitch)
# take long enough to speak that a genuine mid-reply interruption
# attempt went unanswered for 23+ seconds — nothing in this file was
# stopping the reply once it ran long, since the "max 2 sentences" rule
# was prompt text only, exactly the same class of gap engine.py had
# before its own mechanical cap was added.
MAX_SPOKEN_SENTENCES_PER_TURN = int(os.getenv("MAX_SPOKEN_SENTENCES_PER_TURN", "2"))

# FIX — exemption for the cap above. Without this, the SAME cap would
# cut off this file's own hard-decline farewell text mid-sentence:
# "Understood — thanks for letting me know. We won't reach out further.
# Wishing your business continued success." is THREE sentences by
# punctuation — a blind 2-sentence cap would lose the final one. This is
# the exact same class of confirmed bug already found and fixed once in
# engine.py's Twilio path; not reintroducing it here. Phrases pulled
# directly from this file's own FAREWELL and RESPONSE HANDLING prompt
# sections below, not invented.
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

# FIX — per explicit request ("universal" credentials — anyone connects
# anytime with no separate dispatch step): this worker no longer uses
# explicit dispatch. AGENT_NAME (and the agent_name= kwarg on
# WorkerOptions below) is intentionally gone — confirmed directly
# against LiveKit's own docs: "By default, an agent is automatically
# dispatched to each new room... Automatic dispatch is the best option
# if you want to assign the same agent to all new participants" — that
# is exactly this use case. As long as this worker process is running,
# ANY room someone connects to (via livekit_laptop_client.html + the
# token from livekit_laptop_test.py) gets Sania joining automatically —
# no need to run a separate dispatch script per session anymore.
#
# Real trade-off, not hidden: livekit_laptop_test.py no longer creates a
# job dispatch, so it can no longer attach per-call metadata
# (partner_name, category, etc.) — there's nothing left to attach it to.
# Sania now always uses the same generic pitch context on this file. If
# you need a personalized per-partner test again, that requires
# switching back to explicit dispatch, not something this file can do
# both ways at once.

_STATIC_KEYTERMS = [
    "Viator", "Aarna", "Sania", "Mondee", "GetYourGuide",
    "Klook", "TripAdvisor", "Expedia", "Booking.com", "Airbnb",
]

# ---------------------------------------------------------------------------
# Text preparation for TTS — ported verbatim from engine.py's validated
# pipeline. If you fix a bug here, fix it in engine.py too (and vice
# versa) — these are meant to be kept identical, not maintained twice
# independently. See module docstring for the confirmed real bugs each
# piece fixes.
# ---------------------------------------------------------------------------

def _normalize_number_words(n: int) -> str:
    """Convert an integer to natural English speech words."""
    ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety",
    ]
    scales = ["", "thousand", "million", "billion", "trillion"]

    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _normalize_number_words(-n)

    def under_1000(x: int) -> str:
        parts = []
        if x >= 100:
            parts.append(ones[x // 100] + " hundred")
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
        # Strip both comma AND space thousand-separators — see
        # _LARGE_NUMBER_RE for why space-grouping needs handling too.
        return _normalize_number_words(int(raw.replace(",", "").replace(" ", "")))
    except ValueError:
        return raw


_CURRENCY_RE = re.compile(
    r"(?P<cur>AED|USD|EUR|GBP|\$|€|£)\s*(?P<num>\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*%")
_MULTIPLIER_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*[xX]\b")
_LARGE_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})+\b"       # comma-grouped: 65,000
    r"|\b\d{1,3}(?: \d{3})+\b"      # FIX — space-grouped: "65 000" — a
                                     # real call showed gpt-oss-120b
                                     # occasionally writing numbers with a
                                     # space instead of a comma as the
                                     # thousands separator (European-style
                                     # grouping); the old regex only
                                     # matched comma-grouped or 4+ solid
                                     # digits, so "65 000" slipped through
                                     # completely unnormalized and got
                                     # read back as raw digits.
    r"|\b\d{4,}\b"                   # solid digits: 65000
)
_DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")


def _decimal_to_spoken(value: str) -> str:
    left, right = value.split(".", 1)
    return (
        _normalize_number_words(int(left))
        + " point "
        + " ".join(_normalize_number_words(int(ch)) for ch in right)
    )


def _normalize_for_speech(text: str) -> str:
    """
    Normalize business text for natural spoken pronunciation.
    Examples: 65000 -> sixty-five thousand | 25% -> twenty-five percent |
    AED 50,000 -> fifty thousand UAE dirhams | 2.5M -> two point five million
    """
    if not text:
        return ""

    out = text

    def currency_repl(m: "re.Match") -> str:
        cur = m.group("cur").upper()
        num = m.group("num")
        if "." in num:
            spoken_num = _decimal_to_spoken(num.replace(",", ""))
        else:
            spoken_num = _normalize_number_words(int(num.replace(",", "")))
        currency_name = {
            "AED": "UAE dirhams", "$": "dollars", "USD": "US dollars",
            "€": "euros", "EUR": "euros", "£": "pounds", "GBP": "British pounds",
        }.get(cur, cur)
        return f"{spoken_num} {currency_name}"

    out = _CURRENCY_RE.sub(currency_repl, out)

    def percent_repl(m: "re.Match") -> str:
        num = m.group("num")
        return (
            (_decimal_to_spoken(num) if "." in num else _normalize_number_words(int(num)))
            + " percent"
        )

    out = _PERCENT_RE.sub(percent_repl, out)

    def multiplier_repl(m: "re.Match") -> str:
        num = m.group("num")
        return (
            (_decimal_to_spoken(num) if "." in num else _normalize_number_words(int(num)))
            + " times"
        )

    out = _MULTIPLIER_RE.sub(multiplier_repl, out)

    for pattern, magnitude in (
        (re.compile(r"\b(\d+(?:\.\d+)?)\s*[Kk]\b"), "thousand"),
        (re.compile(r"\b(\d+(?:\.\d+)?)\s*[Mm]\b"), "million"),
        (re.compile(r"\b(\d+(?:\.\d+)?)\s*[Bb]\b"), "billion"),
    ):
        def magnitude_repl(m: "re.Match", magnitude=magnitude) -> str:
            num = m.group(1)
            spoken = _decimal_to_spoken(num) if "." in num else _normalize_number_words(int(num))
            return f"{spoken} {magnitude}"
        out = pattern.sub(magnitude_repl, out)

    out = _DECIMAL_RE.sub(lambda m: _decimal_to_spoken(m.group(0)), out)
    out = _LARGE_NUMBER_RE.sub(_number_to_spoken, out)

    # UAE is spoken as letters, while ordinary English remains untouched.
    out = re.sub(r"\bUAE\b", "U.A.E.", out, flags=re.IGNORECASE)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Proper-noun pronunciation fixes for TTS
#
# FIX — confirmed real bug from a real call transcript: "Sania" was heard
# spoken as "Senia" and "Aarna" was heard as "Ana" — both losing a sound
# in a CONSISTENT way, not randomly. This is a well-known TTS failure
# mode: unusual, foreign-origin proper nouns the model has little
# training data for get pulled toward the nearest familiar-sounding
# pattern instead of pronounced as written. Both "Sania" and "Aarna" are
# exactly that kind of rare, non-dictionary proper noun.
#
# Fix: respell these specific words phonetically ONLY in the text that
# reaches TTS — the actual business-logic text (what the LLM reasons
# over, what gets logged, what goes to the DB) is completely untouched,
# since this substitution happens here, at the TTS-preparation stage,
# not upstream in the prompt or history.
#
# IMPORTANT — these respellings are my best-effort guess at the intended
# pronunciation, not a confirmed-correct one: "Sania" respelled toward
# /SAHN-ya/ (the common pronunciation of this name, e.g. as in "Sania
# Mirza") and "Aarna" respelled toward /AR-nah/, ensuring the R is
# clearly present (the R was the exact sound going missing in the real
# transcript — "Ana" has no R at all). If either isn't the actual
# intended pronunciation, update _PRONUNCIATION_SUBSTITUTIONS directly —
# it's a single, easy-to-edit dict, not something that needs a prompt
# change or a deeper fix.
# ---------------------------------------------------------------------------
_PRONUNCIATION_SUBSTITUTIONS = {
    "Sania": "Saanya",
    "Aarna": "Arnah",
}
_PRONUNCIATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _PRONUNCIATION_SUBSTITUTIONS) + r")\b"
)


def _apply_pronunciation_fixes(text: str) -> str:
    return _PRONUNCIATION_RE.sub(lambda m: _PRONUNCIATION_SUBSTITUTIONS[m.group(1)], text)


def _prepare_tts_text(text: str) -> str:
    """Normalize speech while preserving only the tags we actually allow."""
    normalized = _normalize_for_speech(text)
    normalized = _apply_pronunciation_fixes(normalized)
    # FIX — per explicit instruction to incorporate real "voice gestures"
    # now that call quality (pronunciation, pace, latency) is stable.
    # Widened from the original deliberately-narrow set — [chuckle],
    # [delight], and [sigh] added as genuinely warm, professional-
    # appropriate reactions (a business call CAN sound human without
    # sounding theatrical). Still explicitly excludes the dramatic/
    # inappropriate tags Fish Audio supports ([screaming], [shouting],
    # [moaning], [singing], [audience laughter], etc.) — see the prompt
    # guidance below for the same restraint applied at the model level.
    allowed = {"pause", "short pause", "emphasis", "exhale", "chuckle", "delight", "sigh"}

    def tag_filter(m: "re.Match") -> str:
        tag = m.group(1).strip().lower()
        return m.group(0) if tag in allowed else ""

    normalized = re.sub(r"\[([^\]]+)\]", tag_filter, normalized)
    return re.sub(r"\s{2,}", " ", normalized).strip()

# ---------------------------------------------------------------------------
# FIX — deterministic script router, ported from engine.py's Twilio path,
# per explicit request: "I want the call's quality and context and flow
# to be as good as engine.py as it has everything which sania should
# speak on call." Confirmed real cause this addresses: without this,
# EVERYTHING routed through one general-purpose prompt, and the LLM
# decided from scratch on every turn — producing inconsistent wording
# for what should be fixed script lines (observed: the same scripted
# pitch coming out as 1, 2, or 3 sentences across different test runs),
# skipped explanations when a competitor platform was mentioned, and a
# wrong deferral for a genuinely off-topic question ("who is the
# president of UAE?"). engine.py's Twilio path avoids all of this by
# answering known intents from fixed, pre-written text BEFORE the LLM
# ever sees them — this section ports that same mechanism here, using
# LiveKit's on_user_turn_completed hook (see the Sania class below) as
# the interception point, verified against LiveKit's own documentation
# for that exact purpose ("cancel the agent's reply").
#
# Ported as close to verbatim as the two frameworks allow — same
# constants, same wording, same regex patterns as engine.py. Divergences
# from engine.py, and why:
#   - No booking-link / WhatsApp / SMS content — same exclusion already
#     applied to this file's prompt.
#   - No audio pre-caching — engine.py prewarms TTS audio per call for
#     near-zero latency on fast-path replies; doing the same here would
#     mean calling the Fish Audio plugin directly outside the normal
#     session pipeline, a second unverified integration on top of the
#     one already flagged in this file's docstring. Skipped in favour of
#     correctness first — session.say() still synthesizes live through
#     the already-configured pipeline, just without the extra latency
#     optimization. Worth revisiting once the base port is confirmed
#     working.
#   - Hold-request timing (5s / 10s / 10s) matches this project's
#     CURRENT engine.py values (the user's own earlier deliberate
#     deviation from the original 10s/10s spec), not the original spec.
# ---------------------------------------------------------------------------

_DNC_PATTERNS = re.compile(
    r"\b(not interested|stop|no thanks|don't call|do not call|remove me|"
    r"unsubscribe|leave me alone|never call|go away|hang up)\b",
    re.IGNORECASE,
)

_ESCALATION_PATTERNS = re.compile(
    r"\b(commission|revenue share|percentage|contract|legal|lawyer|"
    r"privacy|gdpr|data protection|complaint|frustrated|angry|"
    r"speak to a human|speak to someone|real person|your manager|"
    r"head office|enterprise|multiple locations|chain|franchise)\b",
    re.IGNORECASE,
)

_BUSY_PATTERNS = re.compile(
    r"\b(call back|call (?:me )?later|i'?m busy|at work|in a meeting|"
    r"can'?t talk (?:now|right now)|bad time|not a good time|driving)\b",
    re.IGNORECASE,
)

_WHO_IS_THIS_PATTERNS = re.compile(
    r"\b(who is this|who am i speaking (?:with|to)|who'?s calling|who are you|who is calling)\b",
    re.IGNORECASE,
)
_WHY_CALLING_PATTERNS = re.compile(
    r"\b(why are you calling|why are you telling me this|why r u calling|"
    r"what'?s this (?:call|about)|what is this call (?:for|about)|"
    r"purpose of (?:this|your) call|why this call)\b",
    re.IGNORECASE,
)
_WHAT_IS_THE_NAME_PATTERNS = re.compile(
    r"\b(what is the company name|what'?s the company name|"
    r"what is (?:it|this|the platform|the marketplace) called|"
    r"what'?s (?:it|this|the platform|the marketplace) called|"
    r"what did you say the name was|what is the name|what'?s the name|"
    r"say the name again|repeat the name)\b",
    re.IGNORECASE,
)
_SOFT_DECLINE_PATTERNS = re.compile(
    r"\b(not right now|not currently|not just now|not at this time|maybe later|"
    r"not for now|no,? not really)\b",
    re.IGNORECASE,
)
_CONTINUATION_MARKERS = re.compile(
    r"\b(but|however|although|also|actually|one more thing)\b", re.IGNORECASE,
)
_NOTHING_FURTHER_PATTERNS = re.compile(
    r"^\s*(no|nope|nothing|that'?s (?:all|it)|i'?m good|no,? (?:that'?s|i'?m) (?:all|good|fine)|"
    r"no questions|nothing else|that'?s everything)\b",
    re.IGNORECASE,
)
_HOLD_REQUEST_PATTERNS = re.compile(
    r"\b(hold on|hold for|please hold|one moment|one sec|just a (?:sec|second|moment)|"
    r"give me a (?:sec|second|moment)|wait a (?:sec|second|moment))\b",
    re.IGNORECASE,
)
_STILL_HERE_PATTERNS = re.compile(
    r"^\s*(yes|yeah|yep|still here|i'?m here|here|yes,? (?:i'?m|still) here)\b\.?\s*$",
    re.IGNORECASE,
)
_HOLD_DURATION_WORDS = {
    "a": 60, "one": 60, "1": 60, "two": 120, "2": 120, "a couple": 120,
    "a couple of": 120, "few": 180, "a few": 180, "three": 180, "3": 180,
}
_HOLD_DURATION_PATTERN = re.compile(
    r"\bhold(?:\s+on)?\s+for\s+(a couple of|a couple|a few|a|one|two|three|\d+)\s*"
    r"(second|sec|minute|min)s?\b",
    re.IGNORECASE,
)


def _extract_hold_duration_s(text: str) -> "int | None":
    """Parses a spoken hold duration into whole seconds, or None if unspecified."""
    match = _HOLD_DURATION_PATTERN.search(text)
    if not match:
        return None
    quantity_word = match.group(1).lower()
    unit = match.group(2).lower()
    quantity = _HOLD_DURATION_WORDS.get(quantity_word)
    if quantity is None:
        try:
            quantity = int(quantity_word)
        except ValueError:
            return None
        quantity = quantity * (1 if unit.startswith("sec") else 60)
        return quantity
    if unit.startswith("sec"):
        seconds_map = {"a": 1, "one": 1, "1": 1, "two": 2, "2": 2,
                       "a couple": 2, "a couple of": 2, "few": 3, "a few": 3,
                       "three": 3, "3": 3}
        return seconds_map.get(quantity_word, quantity)
    return quantity


_IDENTITY_CONFIRM_PATTERNS = re.compile(
    r"^\s*(yes|yeah|yep|yup|correct|right|speaking|this is (?:him|her|me)|"
    r"that's me|that is me|you'?re speaking to (?:him|her|me)|"
    r"yes[, ]+(?:this is|speaking))\b", re.I
)
_NEGATIVE_IDENTITY_PATTERNS = re.compile(
    r"\b(no|wrong person|wrong number|not me|not him|not her)\b", re.I
)
_AARNA_EXPLICIT_PATTERNS = re.compile(
    r"\b(what is aarna|what's aarna|tell me about aarna|tell me more about aarna|"
    r"more about aarna|what does aarna do|how does aarna work|how will aarna help|"
    r"explain aarna|what exactly is aarna|what is this platform|what does this platform do)\b", re.I
)
_MONDEE_EXPLICIT_PATTERNS = re.compile(
    r"\b(what is mondee|what's mondee|tell me about mondee|tell me more about mondee|"
    r"what does mondee do|who is mondee|what exactly is mondee)\b", re.I
)
_BOTH_AARNA_MONDEE_PATTERNS = re.compile(
    r"\b(aarna.{0,60}mondee|mondee.{0,60}aarna|both aarna and mondee)\b", re.I
)
_GENERIC_EXPLANATION_PATTERNS = re.compile(
    r"(?:\b(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|explain)\b|"
    r"\b(?:tell|explain)\s+(?:me\s+)?(?:more|about|what)\b|"
    r"\bwhat\s+(?:is|was)\s+(?:that|this|it)\b|"
    r"\bwhat\s+does\s+(?:that|this|it)\s+do\b|"
    r"\bhow\s+does\s+(?:that|this|it)\s+work\b|"
    r"\bhow\s+(?:will|would)\s+(?:that|this|it)\s+help\b|"
    r"\bwhat\s+exactly\s+is\s+(?:that|this|it)\b|"
    r"\b(?:so|okay|ok|well|please),?\s+(?:like,?\s*)?(?:can you|could you|would you|tell me|explain)\b)",
    re.I
)
_EXISTING_PLATFORM_PATTERNS = re.compile(
    r"\b(already on|getyourguide|viator|tripadvisor|trip advisor|klook|expedia|"
    r"booking\.com|airbnb|already listed|already use|already using|we're on|we are on|we use)\b", re.I
)
_BENEFIT_QUESTION_PATTERNS = re.compile(
    r"\b(what are the benefits|what'?s the benefit|what benefits|"
    r"why should (?:i|we)|what'?s in it for (?:me|us)|how does (?:it|this|aarna) help|"
    r"how would (?:it|this|aarna) help|what do (?:i|we) get|why (?:aarna|use aarna|join)|"
    r"what'?s the advantage|why partner with)\b", re.I
)


def _normalize_script_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


_SHORT_CONTINUATION_PATTERNS = re.compile(
    r"^\s*(yes|yeah|yep|yup|no|nope|okay|ok|sure|maybe|thanks|thank you|"
    r"great|good|fine|alright|right|correct|got it|understood|hello|hi|"
    r"what|huh|sorry|please|go ahead|continue)\b\.?\s*$", re.I
)

_WH_QUESTION_STARTERS = (
    "what", "who", "why", "how", "when", "where", "is", "are", "do",
    "does", "can", "could", "would", "will", "tell", "explain",
)
_CALL_FLOW_EXCLUSION_PATTERNS = re.compile(
    r"\b(next step|next steps|what next|what'?s next|move forward|"
    r"moving forward|how do we proceed|what do we do now|what happens next|"
    r"how does this work from here)\b", re.I
)


def _looks_like_a_question(text: str) -> bool:
    """Conservative gate for whether an utterance should be considered for KB lookup at all."""
    normalized = _normalize_script_text(text)
    if not normalized:
        return False
    if _SHORT_CONTINUATION_PATTERNS.match(normalized):
        return False
    if _CALL_FLOW_EXCLUSION_PATTERNS.search(normalized):
        return False
    if "?" in text:
        return True
    first_word = normalized.split()[0] if normalized.split() else ""
    return first_word in _WH_QUESTION_STARTERS


def _is_bare_soft_decline(text: str) -> bool:
    """True only for a soft-decline phrase with nothing else attached."""
    if not _SOFT_DECLINE_PATTERNS.search(text):
        return False
    if _looks_like_a_question(text):
        return False
    if _CONTINUATION_MARKERS.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Local semantic knowledge base — same model, same entries, same threshold
# as engine.py's KB. Loaded once at module import time, same as there.
# ---------------------------------------------------------------------------
KB_SIMILARITY_THRESHOLD = float(os.getenv("KB_SIMILARITY_THRESHOLD", "0.55"))
KB_EMBEDDING_MODEL_NAME = os.getenv("KB_EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

KNOWLEDGE_BASE: list[tuple[str, str, str]] = [
    ("kb_partner_count",
     "how many travel partners does Mondee have",
     "Mondee distributes travel content across sixty-five thousand travel partners globally."),
    ("kb_mondee_size",
     "how big is Mondee, how many offices does Mondee have",
     "Mondee is the third-largest travel consolidator in the U.S., with nineteen offices worldwide."),
    ("kb_aarna_reach",
     "how does Aarna help me get more customers or bookings",
     "Aarna gets you discovered by advisors and travelers who wouldn't otherwise have found you, "
     "through Mondee's global network."),
    ("kb_aarna_vs_others",
     "how is Aarna different from other platforms I'm already listed on",
     "Aarna doesn't replace those, it adds Mondee's advisor and corporate travel network "
     "specifically, which those platforms don't cover."),
]

_kb_embedding_model = None
_kb_question_embeddings = None


def _load_kb_embedding_model() -> None:
    """Same graceful-degradation pattern as engine.py: failure disables KB, never crashes."""
    global _kb_embedding_model, _kb_question_embeddings
    try:
        from fastembed import TextEmbedding
        t_start = time.monotonic()
        _kb_embedding_model = TextEmbedding(model_name=KB_EMBEDDING_MODEL_NAME)
        questions = [q for _, q, _ in KNOWLEDGE_BASE]
        _kb_question_embeddings = list(_kb_embedding_model.embed(questions))
        logger.info(
            "KB embedding model loaded and %d KB entries embedded in %.0fms",
            len(KNOWLEDGE_BASE), (time.monotonic() - t_start) * 1000,
        )
    except Exception as exc:
        logger.warning(
            "KB embedding model failed to load (%s) — KB lookup disabled for "
            "this process; question-like utterances will use the normal LLM "
            "flow instead.", exc,
        )
        _kb_embedding_model = None
        _kb_question_embeddings = None


_load_kb_embedding_model()


def _cosine_similarity(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _kb_available() -> bool:
    return _kb_embedding_model is not None and bool(KNOWLEDGE_BASE)


def _kb_lookup(text: str) -> "tuple[str, str, float] | None":
    if not _kb_available():
        return None
    query_vec = list(_kb_embedding_model.embed([text]))[0]
    scored = [
        (KNOWLEDGE_BASE[i][0], KNOWLEDGE_BASE[i][2], _cosine_similarity(query_vec, _kb_question_embeddings[i]))
        for i in range(len(KNOWLEDGE_BASE))
    ]
    best_key, best_answer, best_score = max(scored, key=lambda x: x[2])
    if best_score >= KB_SIMILARITY_THRESHOLD:
        return best_key, best_answer, best_score
    return None


# ---------------------------------------------------------------------------
# Script constants — same wording as engine.py, "Abhee"->"Aarna" already
# correct here, booking-link content already excluded.
# ---------------------------------------------------------------------------
AARNA_INTRO_TEMPLATE = (
    "We've recently launched Aarna, our experiences marketplace here in the UAE, "
    "and {partner} came up as exactly the kind of partner we'd love to work with. "
    "Is there anything specific you want to know?"
)
AARNA_EXPLANATION = (
    "aarna is Mondee's experiences marketplace, and the idea is to explore whether "
    "your experiences can reach travellers through Mondee's wider distribution network."
)
MONDEE_EXPLANATION = (
    "Mondee is the third-largest travel consolidator in the United States and has "
    "nineteen offices worldwide. Through our proprietary Mondee Marketplace technology, "
    "we distribute air tickets and hotel accommodation across sixty-five thousand travel "
    "partners globally."
)
# FIX — the "both Aarna and Mondee" intent used to just reuse AARNA_EXPLANATION,
# which previously bundled the Mondee background in as its own opening
# sentences. Now that AARNA_EXPLANATION is Aarna-only (per the updated
# wording above), reusing it alone for a question about BOTH would
# silently drop the Mondee content that used to be there — a real
# content regression, not just a wording tweak. This combines both
# updated answers explicitly instead, preserving the original intent.
AARNA_AND_MONDEE_EXPLANATION = MONDEE_EXPLANATION + " " + AARNA_EXPLANATION
AARNA_BENEFIT = (
    "Aarna doesn't replace those, it adds Mondee's advisor and corporate travel network specifically, "
    "which those platforms don't cover."
)
AARNA_GENERAL_BENEFIT = (
    "Aarna gives you direct access to Mondee's global advisor and corporate travel network, "
    "so new travelers and advisors who wouldn't otherwise find you can discover and book your "
    "experiences directly."
)
MEETING_SCHEDULE_FOLLOWUP = (
    "Would you like to schedule a brief call with our partnership team to go over this in more detail?"
)
WHO_IS_THIS_REPLY = "Of course — I am Sania, Mondee's AI voice assistant."
WHAT_IS_THE_NAME_REPLY = "It's called Aarna — A, A, R, N, A."
WHY_CALLING_TEMPLATE = (
    "I'm calling because we believe {partner} could be relevant for Aarna, our experiences "
    "marketplace — my purpose is to share some information and see if an introductory "
    "conversation would be of interest."
)
ESCALATION_DEFLECTION_REPLY = (
    "That is a good question. Our partnerships team can explain that accurately during "
    "the introductory call."
)
HARD_DECLINE_REPLY = "Understood. We will not contact you again. Goodbye."
SOFT_DECLINE_CHECK_REPLY = "Understood. Is there anything you would like to know before we close?"
SOFT_DECLINE_CLOSING_REPLY = "Understood. Thank you for your time. Have a wonderful day."
BUSY_RECIPIENT_REPLY = (
    "No problem — would you prefer we call you back at a better time, "
    "or is now still okay for a quick word?"
)
AARNA_OFF_TOPIC_REDIRECT_TEMPLATE = (
    "I'm here to help with Aarna and our partnership with {partner} — let's stick to that so "
    "I can make the best use of your time."
)

_QUESTION_ANSWER_KEYS = {
    "aarna_explanation", "aarna_and_mondee", "mondee_explanation",
    "aarna_benefit", "aarna_general_benefit", "why_calling",
} | {kb_key for kb_key, _q, _a in KNOWLEDGE_BASE}

# FIX — hold-check cadence matches this project's CURRENT engine.py
# values (5s first check, per the user's own earlier deliberate change
# from the original 10s spec) — not the original build-spec default.
HOLD_FIRST_CHECK_S = int(os.getenv("HOLD_FIRST_CHECK_S", "5"))
HOLD_SECOND_CHECK_S = int(os.getenv("HOLD_SECOND_CHECK_S", "10"))
HOLD_FINAL_WAIT_S = int(os.getenv("HOLD_FINAL_WAIT_S", "10"))


def _build_prompt(
    partner_name: str,
    contact_name: str,
    category: str,
    company_synopsis: str,
    digitisation: str,
) -> str:
    """
    FIX — brought into parity with engine.py's _system_prompt_compact,
    per explicit request ("I want the call's quality and context and
    flow to be as good as engine.py"). Known intents (who is this, why
    calling, what's the name, Aarna/Mondee explanations, decline
    handling, escalation, hold requests, off-topic questions) are now
    handled OUTSIDE this prompt entirely, by the deterministic router in
    Sania.on_user_turn_completed below — this prompt's job is FLOW and
    general conversational judgment for whatever's left over, matching
    engine.py's own division of labor exactly (see that file's
    "APPROVED SCRIPT RULE" section). The old bespoke pitch_question
    mechanic is dropped — engine.py's actual live flow doesn't force a
    scripted existing-platforms question either; that intent is caught
    naturally by the router if the caller brings it up themselves (see
    _EXISTING_PLATFORM_PATTERNS / _BENEFIT_QUESTION_PATTERNS above).
    """
    partner = partner_name or "your business"
    who = f"{contact_name} from {partner}" if contact_name else partner

    if digitisation == "digitised":
        tone = "Professional partnership tone — this supplier understands B2B language."
    elif digitisation == "hyperlocal":
        tone = "Highly conversational, human tone — avoid all tech/corporate language."
    else:
        tone = "Friendly, simple business-expansion tone."

    context_lines = [f"partner_name={partner}"]
    if category:
        context_lines.append(f"category={category}")
    if company_synopsis:
        context_lines.append(f"company_synopsis={company_synopsis}")
    context_block = "\n".join(context_lines)

    return f"""You are Sania, an AI voice assistant calling on behalf of Mondee about a potential Aarna partnership with {who}. You are not a human, a general-information assistant, a negotiator or a legal adviser. Never deny or hide that you are an AI assistant.

LIVE CALL CONTEXT
{context_block}

RULES:
- The conversation always takes priority over script progression. Always answer the supplier's actual question or statement before moving to the next step. Never repeat a point already delivered. Never ask for information already given.
- Respond to the supplier's latest words. Never invent missing facts.
- Max 2 spoken sentences per normal turn and one question per turn.
- If the supplier's speech seems incomplete, trails off mid-thought, or contains a continuation word like "but", "however", "although", "also", "actually", "one more thing", "is there" or "what about", do not treat it as a completed statement — say only "Of course — please go ahead." and wait. A reply starting with "no" is not automatically a refusal if it continues with "but", a question, or more information; respond to the complete meaning, not just the first word.
- If you were just interrupted or your previous reply was cut off, do not assume the caller heard the complete response. Answer what they just said first; never refer to cancelled or unspoken content as though it was delivered.
- If unclear/garbled, ask "Sorry, could you say that again?" rather than guessing.
- Never invent or imply client names, bookings, revenue, supplier performance, pricing, commissions, fees, revenue share, contract terms, onboarding terms, launch dates, guarantees, or unsupported capabilities; defer those to the partnership lead.
- A plain date/time answer gets a direct confirmation, never the unknown-fact deferral.
- Do not repeat a question just because of a short pause; only repeat it after a genuine no-response, and then only once, in shorter wording.
- Use a farewell only when actually ending the call.
- Polite English, no slang, no pressure. {tone}

FLOW:
1. After identity confirmation, move immediately into the main body. NEVER ask whether they are the owner, decision maker, or whether it is a good time — that was already asked in the opening greeting.
2. The Aarna introduction, and known Aarna/Mondee/existing-platform questions, decline handling, escalation, hold requests, and off-topic questions are ALL handled outside you, automatically, with fixed approved wording — you will simply not see most of those turns. If one somehow still reaches you, keep your answer brief and consistent with what a partnership assistant would actually say, don't improvise new facts.
3. Your job: handle genuinely novel questions or statements naturally, then offer and schedule the human partnerships call — capture and confirm a concrete date/time, then close warmly: "Perfect, I'll send a calendar invite to your email. Have a wonderful day!"

VOICE TAGS — apply every time the trigger genuinely occurs, max ONE tag per turn: turn opens with "Great"/"Perfect"/confirming something positive → [delight] right after that word WITH A SPACE before the tag — "Great [delight]," never "Great[delight],". About to ask the key scheduling question, or supplier just raised real pushback → [short pause] to open. One word carries real weight (a number, "free", a deadline) → [emphasis] on that word only. Never [laughing], [excited], [whisper], [screaming], [moaning], [singing] or similar — too theatrical for a professional call."""



class Sania(Agent):
    """
    FIX — full deterministic-router port from engine.py, per explicit
    request ("I want the call's quality and context and flow to be as
    good as engine.py"). See the module-level comment block above
    (right before _DNC_PATTERNS) for the full rationale and the honest
    list of what's ported vs. deliberately different.
    """

    _KNOWN_PLATFORMS = ("Viator", "GetYourGuide", "TripAdvisor", "Klook",
                        "Expedia", "Booking.com", "Airbnb")

    def __init__(self, instructions: str, ctx: JobContext, partner_name: str):
        super().__init__(instructions=instructions)
        self._ctx = ctx
        self._partner_name = partner_name or "partners like you"
        # Per-call state — mirrors engine.py's meta dict fields used by
        # the deterministic router and _script_reply_for_utterance.
        self.call_state = "IDENTITY"
        self.conversation_facts: dict = {"existing_platforms": []}
        self.script_segments_spoken: set[str] = set()
        self._answered_question_count = 0
        self._soft_decline_check_pending = False
        self._on_hold = False
        self._hold_awaiting_checkin_response = False
        self._hold_check_task: "asyncio.Task | None" = None
        self.dnc_requested = False
        self.escalation_requested = False
        self._call_ended = False
        # FIX — True only while a deterministic/fixed script line is
        # being spoken via _say_script/_end_call. tts_node's sentence
        # cap checks this and never cuts off script text — matches
        # engine.py's ACTUAL behavior: fixed constants there are
        # dispatched via a completely separate code path
        # (_record_and_speak_script) that the cap never touches at all,
        # not via any "farewell exemption" inside the cap logic itself.
        # Confirmed real risk this avoids: AARNA_EXPLANATION is 3
        # sentences — cutting it at MAX_SPOKEN_SENTENCES_PER_TURN=2
        # would silently drop "Aarna is our newer experiences
        # marketplace..." every single time it's spoken.
        self._suppress_sentence_cap = False

    def _script_reply_for_utterance(self, text: str) -> "tuple[str, str] | None":
        """Ported from engine.py's _script_reply_for_utterance; reads state from self instead of a meta dict."""
        state = self.call_state
        normalized = _normalize_script_text(text)
        partner = self._partner_name

        if _WHO_IS_THIS_PATTERNS.search(normalized):
            return "who_is_this", WHO_IS_THIS_REPLY
        if _WHAT_IS_THE_NAME_PATTERNS.search(normalized):
            return "what_is_the_name", WHAT_IS_THE_NAME_REPLY
        if _WHY_CALLING_PATTERNS.search(normalized):
            return "why_calling", WHY_CALLING_TEMPLATE.format(partner=partner)

        if state == "IDENTITY":
            if _NEGATIVE_IDENTITY_PATTERNS.search(normalized):
                return None
            if _IDENTITY_CONFIRM_PATTERNS.search(normalized) or normalized in {"yes", "yeah", "yep", "speaking"}:
                return "aarna_intro", AARNA_INTRO_TEMPLATE.format(partner=partner)

        if _BOTH_AARNA_MONDEE_PATTERNS.search(normalized):
            return "aarna_and_mondee", AARNA_AND_MONDEE_EXPLANATION
        if _MONDEE_EXPLICIT_PATTERNS.search(normalized):
            return "mondee_explanation", MONDEE_EXPLANATION
        if _AARNA_EXPLICIT_PATTERNS.search(normalized):
            return "aarna_explanation", AARNA_EXPLANATION.format(partner=partner)

        if state == "AFTER_AARNA_INTRO" and _GENERIC_EXPLANATION_PATTERNS.search(normalized):
            return "aarna_explanation", AARNA_EXPLANATION.format(partner=partner)

        if _EXISTING_PLATFORM_PATTERNS.search(normalized) or _BENEFIT_QUESTION_PATTERNS.search(normalized):
            if self.conversation_facts.get("existing_platforms"):
                return "aarna_benefit", AARNA_BENEFIT
            return "aarna_general_benefit", AARNA_GENERAL_BENEFIT
        return None

    async def _say_script(self, text: str, allow_interruptions: bool = True) -> bool:
        """
        Speak pre-approved, fixed script text (never LLM-generated).
        Applies text preparation directly here so pronunciation fixes
        and number normalization are GUARANTEED to apply, rather than
        relying on an unverified assumption about whether session.say()
        with a plain string routes through tts_node the same way an LLM
        stream does. Also suppresses tts_node's sentence cap for the
        duration — see _suppress_sentence_cap's docstring above.

        FIX — confirmed real bug from a live test call: a reply spoken
        via this method got genuinely interrupted mid-sentence (correct
        — the caller was talking), but the caller of this method
        (_speak_script) had no way to know that, and unconditionally
        went on to speak the meeting-schedule follow-up anyway — right
        before Sania had even acknowledged the interruption. engine.py's
        Twilio path explicitly guards this with `if completed:`; this
        now does the same, using LiveKit's own documented pattern for
        detecting it: `handle = session.say(...)`, then
        `await handle.wait_for_playout()`, then read `handle.interrupted`
        (confirmed against LiveKit's own field-guide example — not
        guessed). Returns True only if the reply played out in full.
        """
        self._suppress_sentence_cap = True
        try:
            handle = self.session.say(
                _prepare_tts_text(text), allow_interruptions=allow_interruptions, add_to_chat_ctx=True,
            )
            await handle.wait_for_playout()
            return not handle.interrupted
        finally:
            self._suppress_sentence_cap = False

    async def _speak_script(self, key: str, text: str) -> None:
        """Speak a deterministic line, tracking state transitions + the meeting-followup counter."""
        self.script_segments_spoken.add(key)
        if key == "aarna_intro":
            self.call_state = "AFTER_AARNA_INTRO"
        elif key in {"aarna_explanation", "aarna_and_mondee", "mondee_explanation", "aarna_benefit", "aarna_general_benefit"}:
            self.call_state = "POST_EXPLANATION"
        logger.info("SCRIPT FAST PATH key=%s — LLM bypassed", key)
        completed = await self._say_script(text)
        # FIX — only attach the follow-up if the answer was actually
        # heard in full; an interrupted reply should not have anything
        # tacked onto it — the caller's interruption gets handled by the
        # LLM's normal next turn instead, same as engine.py.
        if completed:
            await self._maybe_speak_meeting_followup(key)

    async def _maybe_speak_meeting_followup(self, answer_key: str) -> None:
        """
        Ported from engine.py's _maybe_speak_meeting_followup: the
        caller's FIRST answered question just gets answered, no
        pressure. From the second onward, a brief follow-up asking to
        schedule a call is attached right after.
        """
        if answer_key not in _QUESTION_ANSWER_KEYS:
            return
        self._answered_question_count += 1
        if self._answered_question_count < 2:
            return
        logger.info(
            "answered question #%d this call, attaching meeting-schedule follow-up",
            self._answered_question_count,
        )
        await self._say_script(MEETING_SCHEDULE_FOLLOWUP)

    async def _end_call(self, farewell_text: str) -> None:
        """Speak a farewell, then end the job — mirrors the SIP file's proven ctx.shutdown() pattern."""
        self._call_ended = True
        await self._say_script(farewell_text, allow_interruptions=False)
        await asyncio.sleep(1.5)
        self._ctx.shutdown(reason="Call ended by Sania")

    async def _hold_check_loop(self, duration_s: "int | None") -> None:
        """Ported from engine.py's _hold_check_loop — timed check-in sequence while on hold."""
        first_wait_s = duration_s if duration_s is not None else HOLD_FIRST_CHECK_S
        try:
            await asyncio.sleep(first_wait_s)
            if not self._on_hold:
                return
            logger.info("hold check #1 firing after %.0fs", first_wait_s)
            self._hold_awaiting_checkin_response = True
            await self._say_script("Are we still connected?")

            await asyncio.sleep(HOLD_SECOND_CHECK_S)
            if not self._on_hold:
                return
            logger.info("hold check #2 firing, no response to check #1")
            await self._say_script("Do you need more time, or would a callback be easier?")

            await asyncio.sleep(HOLD_FINAL_WAIT_S)
            if not self._on_hold:
                return
            logger.info("no response after hold checks, ending call")
            self._on_hold = False
            await self._end_call("It seems this may not be a convenient time. We can try again later. Goodbye.")
        except asyncio.CancelledError:
            pass

    async def on_user_turn_completed(
        self, turn_ctx: "ChatContext", new_message: "ChatMessage",
    ) -> None:
        """
        FIX — the deterministic-router entry point, confirmed against
        LiveKit's own documentation ("cancel the agent's reply" via
        raise StopResponse(), speak fixed text via session.say()
        instead). Waterfall order ported to match engine.py's
        _handle_utterance exactly: DNC -> hold request -> return-from-
        hold -> soft-decline closing-check pending -> busy -> bare soft
        decline -> script fast path -> escalation -> KB lookup /
        off-topic redirect -> fall through to the normal LLM turn
        (return without raising, exactly as engine.py falls through to
        Groq for anything none of the above catches).
        """
        text = (new_message.text_content or "").strip()
        if not text:
            return

        normalized = _normalize_script_text(text)

        facts = self.conversation_facts
        for platform in self._KNOWN_PLATFORMS:
            if re.search(rf"\b{re.escape(platform)}\b", text, re.IGNORECASE):
                if platform not in facts.setdefault("existing_platforms", []):
                    facts["existing_platforms"].append(platform)

        # -- DNC — hard stop -----------------------------------------------
        if _DNC_PATTERNS.search(text):
            self.dnc_requested = True
            logger.info("DNC signal detected — ending immediately per approved script")
            await self._end_call(HARD_DECLINE_REPLY)
            raise StopResponse()

        # -- Hold request -----------------------------------------------------
        if _HOLD_REQUEST_PATTERNS.search(text):
            duration_s = _extract_hold_duration_s(text)
            duration_match = _HOLD_DURATION_PATTERN.search(text)
            if duration_s is not None and duration_match:
                hold_ack_reply = f"Of course, I can hold for {duration_match.group(0).split('for', 1)[-1].strip()}."
            else:
                hold_ack_reply = "Of course, I can hold."
            logger.info("hold request detected (duration=%s), starting check-in sequence", duration_s)
            await self._say_script(hold_ack_reply)
            self._on_hold = True
            self._hold_check_task = asyncio.create_task(self._hold_check_loop(duration_s))
            raise StopResponse()

        # -- Return from hold ---------------------------------------------------
        if self._on_hold:
            self._on_hold = False
            was_awaiting_checkin = self._hold_awaiting_checkin_response
            self._hold_awaiting_checkin_response = False
            if self._hold_check_task is not None and not self._hold_check_task.done():
                self._hold_check_task.cancel()
            if was_awaiting_checkin and _STILL_HERE_PATTERNS.match(normalized):
                logger.info("caller confirmed still connected, asking to continue")
                await self._say_script("Would you like me to continue?")
                raise StopResponse()
            logger.info("caller returned from hold with real content, resuming normally")
            # falls through — text gets processed by the normal routing below

        # -- Soft-decline closing-check pending ------------------------------
        if self._soft_decline_check_pending:
            self._soft_decline_check_pending = False
            if _NOTHING_FURTHER_PATTERNS.match(normalized):
                logger.info("soft-decline closing confirmed, ending warmly")
                await self._end_call(SOFT_DECLINE_CLOSING_REPLY)
                raise StopResponse()
            # else: they had something else to say — fall through below

        # -- Busy recipient ------------------------------------------------------
        if _BUSY_PATTERNS.search(text):
            logger.info("busy recipient signal detected")
            await self._say_script(BUSY_RECIPIENT_REPLY)
            raise StopResponse()

        # -- Bare soft decline ---------------------------------------------------
        if _is_bare_soft_decline(text):
            logger.info("bare soft decline, asking closing-check question")
            await self._say_script(SOFT_DECLINE_CHECK_REPLY)
            self._soft_decline_check_pending = True
            raise StopResponse()

        # -- Deterministic script fast path ---------------------------------------
        fast_script = self._script_reply_for_utterance(text)
        if fast_script:
            script_key, script_text = fast_script
            await self._speak_script(script_key, script_text)
            raise StopResponse()

        # -- Escalation ------------------------------------------------------------
        if _ESCALATION_PATTERNS.search(text):
            self.escalation_requested = True
            logger.info("escalation trigger detected — deflecting, call continues")
            await self._say_script(ESCALATION_DEFLECTION_REPLY)
            raise StopResponse()

        # -- KB lookup, question-gated, off-topic redirect as the miss case --------
        if _looks_like_a_question(text) and _kb_available():
            kb_match = _kb_lookup(text)
            if kb_match:
                kb_key, kb_answer, kb_score = kb_match
                logger.info(
                    "KB LOOKUP hit key=%s score=%.3f (threshold=%.2f): %r",
                    kb_key, kb_score, KB_SIMILARITY_THRESHOLD, text[:60],
                )
                await self._speak_script(kb_key, kb_answer)
                raise StopResponse()

            logger.info("question-like utterance, no confident KB match, using fixed redirect: %r", text[:60])
            redirect_text = AARNA_OFF_TOPIC_REDIRECT_TEMPLATE.format(partner=self._partner_name)
            await self._say_script(redirect_text)
            raise StopResponse()

        # Nothing matched — fall through to the normal LLM turn, unchanged.
        return

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """
        Text-preparation hook (number normalization, Sania/Aarna
        pronunciation fixes, voice-tag filtering) — same mechanism as
        before, confirmed real, documented LiveKit hook for this exact
        purpose.

        FIX — now also enforces MAX_SPOKEN_SENTENCES_PER_TURN
        mechanically on genuinely LLM-generated replies, matching
        engine.py's own cap. Deliberately does NOT apply while
        self._suppress_sentence_cap is True (set by _say_script/
        _end_call around every deterministic dispatch) — see that
        flag's docstring in __init__ for why a blind cap here would be
        wrong, not just unnecessary.
        """
        _sentence_end_re = re.compile(r"(?<=[.!?])\s+")
        suppress_cap = self._suppress_sentence_cap  # snapshot at generation start

        async def _prepared() -> AsyncIterable[str]:
            buffer = ""
            sentence_count = 0
            async for chunk in text:
                buffer += chunk
                parts = _sentence_end_re.split(buffer)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        sentence_count += 1
                        yield _prepare_tts_text(sentence)
                        if not suppress_cap and sentence_count >= MAX_SPOKEN_SENTENCES_PER_TURN:
                            logger.info(
                                "reached MAX_SPOKEN_SENTENCES_PER_TURN=%d, closing out the rest of this turn",
                                MAX_SPOKEN_SENTENCES_PER_TURN,
                            )
                            return
                    buffer = parts[-1]
            remainder = buffer.strip()
            if remainder:
                yield _prepare_tts_text(remainder)

        return Agent.default.tts_node(self, _prepared(), model_settings)


async def entrypoint(ctx: JobContext) -> None:
    metadata = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            logger.warning("Job metadata wasn't valid JSON: %r", ctx.job.metadata)

    partner_name = metadata.get("partner_name", "")
    contact_name = metadata.get("contact_name", "")
    category = metadata.get("category", "")
    company_synopsis = metadata.get("company_synopsis", "")
    digitisation = metadata.get("digitisation", "semi")

    logger.info("Job started room=%s partner=%r", ctx.room.name, partner_name)

    await ctx.connect()

    keyterms = list(_STATIC_KEYTERMS)
    for name in (partner_name, contact_name, category):
        if name:
            keyterms.append(name)

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
            base_url="https://api.groq.com/openai/v1",
            temperature=0.4,
            extra_body={"reasoning_effort": GROQ_REASONING_EFFORT},
        ),
        tts=fishaudio.TTS(
            api_key=FISH_AUDIO_API_KEY,
            model=FISH_AUDIO_MODEL,
            voice_id=FISH_AUDIO_VOICE_ID or None,
            speed=FISH_AUDIO_SPEED,
            sample_rate=24000,
        ),
        vad=silero.VAD.load(),
    )

    agent = Sania(
        instructions=_build_prompt(partner_name, contact_name, category, company_synopsis, digitisation),
        ctx=ctx,
        partner_name=partner_name,
    )

    # FIX — per explicit request for engine.py parity: engine.py detects
    # a farewell in the LLM's OWN generated closing line (the scheduling
    # flow's "Perfect, I'll send a calendar invite... Have a wonderful
    # day!" is LLM-generated here, not a fixed constant) and ends the
    # call. Confirmed real, documented LiveKit mechanism for observing
    # completed assistant turns: the conversation_item_added event.
    @session.on("conversation_item_added")
    def _on_conversation_item_added(event) -> None:
        item = getattr(event, "item", None)
        if item is None or getattr(item, "role", None) != "assistant":
            return
        if agent._call_ended:
            return
        reply_text = getattr(item, "text_content", "") or ""
        if _FAREWELL_PATTERNS.search(reply_text):
            logger.info("LLM-generated farewell detected — ending call")
            agent._call_ended = True
            asyncio.create_task(_end_after_farewell(ctx))

    async def _end_after_farewell(ctx: JobContext) -> None:
        await asyncio.sleep(2)  # let TTS finish playing before closing
        ctx.shutdown(reason="Call ended after farewell")

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
    )

    # FIX — this replaces the SIP branch's wait_until_answered=True. There
    # is no phone call to answer here; the equivalent "someone real is
    # actually present now" signal for a WebRTC room is a participant
    # actually joining. Without this, Sania could start speaking her
    # greeting into an empty room before you've even opened the browser
    # tab and clicked Connect — confirmed real risk given the two-terminal
    # workflow this experiment requires (worker started in terminal 1
    # before you've necessarily joined via the browser yet).
    logger.info("Waiting for a participant to join room=%s ...", ctx.room.name)
    participant = await ctx.wait_for_participant()
    logger.info("Participant joined: identity=%s — starting conversation", participant.identity)

    # Opening line — the agent speaks first, matching engine.py's flow
    # (Sania introduces herself before waiting on the caller). This one
    # LLM-generated turn is intentionally exempt from the router (there
    # is no user utterance yet to route) and from the sentence cap
    # concern (it's a short, one-off greeting instruction, not a
    # multi-turn conversational reply).
    agent._suppress_sentence_cap = True
    try:
        # FIX — confirmed real bug from a live test call: when partner_name
        # and contact_name are both empty (this laptop test doesn't always
        # supply them), the LLM invented a literal "[partner_name]"
        # placeholder and spoke it aloud instead of just phrasing around
        # the missing info. Branching the instruction explicitly here,
        # rather than leaving the LLM to improvise a graceful fallback,
        # matches engine.py's own _build_opening_line — which handles the
        # exact same missing-name case with explicit branches, not a
        # single generic instruction hoping the model degrades gracefully.
        if contact_name and partner_name:
            who_clause = f"you're speaking with {contact_name} from {partner_name}"
        elif partner_name:
            who_clause = f"you're speaking with the right contact from {partner_name}"
        else:
            who_clause = "it's a good time to talk"
        await session.generate_reply(
            instructions=(
                "Greet the person now: say hello, identify yourself as Sania, an AI "
                f"assistant from Aarna, a travel management platform, and confirm {who_clause}. "
                "Never say a bracketed placeholder or a variable name out loud — if you don't "
                "have a name, just phrase around it naturally. Keep it to two sentences."
            )
        )
    finally:
        agent._suppress_sentence_cap = False


if __name__ == "__main__":
    # No agent_name here — see the comment near the top of this file for
    # why: this is deliberate, confirmed-correct automatic dispatch, not
    # an oversight.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=os.getenv("LIVEKIT_LAPTOP_AGENT_NAME", "aarna-sania-laptop-test")))