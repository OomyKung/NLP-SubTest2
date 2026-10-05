"""
NetGuide AI - Cisco CCNA Knowledge Assistant
============================================

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit.

How it works:
    1. Load every .txt file in ./data
    2. Clean the text and split it into overlapping chunks
    3. Embed each chunk with sentence-transformers (all-MiniLM-L6-v2)
    4. Store the vectors in a FAISS index (built once on startup, then cached)
    5. For every question: embed it, retrieve the top-k most similar chunks,
       and ask Qwen 3.8 27B (via Groq) to answer ONLY from those chunks.

Run locally:
    streamlit run app.py
"""

import csv
import html
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import APIConnectionError, AuthenticationError, Groq, NotFoundError, RateLimitError
from sentence_transformers import SentenceTransformer


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

APP_NAME = "NetGuide AI"
APP_TAGLINE = "Cisco CCNA Knowledge Assistant"

# Folder that holds the knowledge-base documents (relative to this file,
# so it works no matter which directory Streamlit is started from).
DATA_DIR = Path(__file__).parent / "data"

# Chunking settings (in characters)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Retrieval settings
TOP_K = 5
# Chunks whose cosine similarity is below this value are treated as irrelevant.
# Kept low on purpose so real questions are never blocked.
MIN_SIMILARITY = 0.20
# If the best match for a question is below this score, it is treated as a
# possible follow-up ("How do I configure it?") and the previous question is
# added to the search. See retrieve().
FOLLOWUP_THRESHOLD = 0.45
# Words that usually point back to the previous question ("Which ports does
# IT use?"). A question containing one is also treated as a follow-up.
FOLLOWUP_WORDS = {"it", "its", "they", "them", "their", "this", "that", "these", "those", "one"}
FOLLOWUP_WORDS_THAI = ("มัน", "นี้", "นั้น", "ดังกล่าว")

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# The brief asked for llama-3.3-70b-versatile, but Groq has retired it.
# Qwen 3.8 27B was tested on the same RAG prompts: it cites file names,
# answers in Thai and English, and replies with the "not found" message correctly.
LLM_MODEL = "qwen/qwen3.8-27b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# How many previous chat messages are sent to the LLM as conversation memory
MAX_HISTORY_MESSAGES = 4

# Exact reply required by the system prompt when the answer is not in the docs
NOT_FOUND_MESSAGE = "ไม่พบข้อมูลในเอกสารที่มี"
# The LLM always returns the Thai message above (it is in the system prompt).
# The app then shows it in the language of the question; see localize_not_found().
NOT_FOUND_MESSAGE_EN = "No information was found in the available documents."

# Chat timestamps are shown in Thailand time. Thailand has no daylight saving,
# so a fixed UTC+7 offset is always correct.
DISPLAY_TIMEZONE = timezone(timedelta(hours=7), name="UTC+7")

USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🌐"

SUGGESTED_QUESTIONS = [
    "What is VLAN?",
    "Explain OSPF Areas.",
    "How does DHCP Relay work?",
    "Difference between Standard and Extended ACL?",
]

# Questions whose answers are NOT in the documents. The LLM knows these from its
# training, so replying "ไม่พบข้อมูลในเอกสารที่มี" proves it only uses the retrieved context.
OUT_OF_SCOPE_QUESTIONS = [
    "How do you configure BGP route reflectors?",
    "Who is the current CEO of Cisco Systems?",
    "ราคาสวิตช์ Cisco Catalyst 9300 เท่าไหร่",
    "How do I create a Kubernetes network policy?",
]

# Evaluation questions (used by the Evaluation view)
TEST_QUESTIONS_FILE = Path(__file__).parent / "test_questions.csv"

VIEW_CHAT = "💬 Chat"
VIEW_EVAL = "📊 Evaluation"

USE_CASES = [
    ("🎓", "Exam preparation", "Review CCNA concepts such as VLANs, STP and OSPF areas."),
    ("🛠️", "Configuration help", "Look up IOS commands for DHCP, NAT, ACLs and EtherChannel."),
    ("🔍", "Troubleshooting", "Follow a structured method to find and fix network issues."),
]

# The system prompt. {context} and {question} are filled in for every question.
SYSTEM_PROMPT_TEMPLATE = """You are NetGuide AI.

Answer ONLY using the provided context.

Rules:

1. If the answer exists in the context:
   * Provide a clear answer.
   * Cite document sources.

2. If the answer is not found:
   * Reply:
     "ไม่พบข้อมูลในเอกสารที่มี"

3. Do not make up information.

4. Do not use outside knowledge.

5. Answer in the same language as the user's question.

Context:
{context}

Question:
{question}
"""


# =============================================================================
# 2. STYLING (custom CSS)
# =============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+Thai:wght@400;600&display=swap');

:root {
    --ng-bg: #0B1220;
    --ng-surface: #111A2E;
    --ng-surface-2: #16223B;
    --ng-border: rgba(148, 163, 184, 0.16);
    --ng-border-strong: rgba(4, 159, 217, 0.55);
    --ng-text: #E6EDF7;
    --ng-muted: #94A3B8;
    --ng-primary: #049FD9;      /* Cisco blue */
    --ng-primary-2: #38BDF8;
    --ng-accent: #6EE7B7;
    --ng-warn: #FBBF24;
    --ng-danger: #F87171;
    --ng-shadow: 0 10px 30px -12px rgba(0, 0, 0, 0.6);
    --ng-mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
}

/* ---------- Base ---------- */
html, body, .stApp, .stMarkdown, p, li, label, button, input, textarea {
    font-family: 'Inter', 'Noto Sans Thai', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp {
    background:
        radial-gradient(1100px 560px at 8% -12%, rgba(4, 159, 217, 0.16), transparent 60%),
        radial-gradient(900px 500px at 110% 8%, rgba(110, 231, 183, 0.07), transparent 60%),
        var(--ng-bg);
    color: var(--ng-text);
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 920px;
    padding-top: 2.2rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1729 0%, #0B1220 100%);
    border-right: 1px solid var(--ng-border);
}
.ng-brand { display: flex; align-items: center; gap: .75rem; margin: .25rem 0 .9rem; }
.ng-logo {
    width: 46px; height: 46px; flex: none; border-radius: 14px;
    display: grid; place-items: center; font-size: 1.5rem;
    background: linear-gradient(135deg, var(--ng-primary), #0E7490);
    box-shadow: 0 8px 24px -8px rgba(4, 159, 217, .75);
}
.ng-brand-name { font-weight: 800; font-size: 1.15rem; letter-spacing: -.01em; color: var(--ng-text); }
.ng-brand-sub { color: var(--ng-muted); font-size: .78rem; }
.ng-side-desc { color: var(--ng-muted); font-size: .85rem; line-height: 1.55; }
.ng-section-title {
    text-transform: uppercase; letter-spacing: .08em; font-size: .7rem; font-weight: 700;
    color: var(--ng-muted); margin: 1.4rem 0 .6rem;
}
.ng-status {
    display: inline-flex; align-items: center; gap: .45rem; margin-top: .9rem;
    font-size: .75rem; padding: .28rem .65rem; border-radius: 999px;
    border: 1px solid var(--ng-border); color: var(--ng-text);
}
.ng-dot { width: 8px; height: 8px; border-radius: 50%; }
.ng-status.ok .ng-dot { background: var(--ng-accent); box-shadow: 0 0 0 3px rgba(110, 231, 183, .2); }
.ng-status.off .ng-dot { background: var(--ng-warn); box-shadow: 0 0 0 3px rgba(251, 191, 36, .2); }
.ng-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; }
.ng-stat {
    background: var(--ng-surface); border: 1px solid var(--ng-border); border-radius: 12px;
    padding: .65rem .75rem; transition: border-color .2s ease, transform .2s ease;
}
.ng-stat:hover { border-color: var(--ng-border-strong); transform: translateY(-1px); }
.ng-stat.wide { grid-column: span 2; }
.ng-stat-value { font-weight: 700; font-size: 1.3rem; color: var(--ng-text); line-height: 1.2; }
.ng-stat-value.small { font-size: .8rem; font-family: var(--ng-mono); font-weight: 500; word-break: break-all; }
.ng-stat-label { font-size: .7rem; color: var(--ng-muted); margin-top: .15rem; }
.ng-doc-list {
    max-height: 300px; overflow-y: auto; padding-right: .25rem;
    scrollbar-width: thin; scrollbar-color: rgba(148, 163, 184, .3) transparent;
}
.ng-doc {
    display: flex; align-items: center; justify-content: space-between; gap: .5rem;
    padding: .45rem .6rem; border-radius: 10px; border: 1px solid transparent;
    font-size: .84rem; color: var(--ng-text); transition: background .2s ease, border-color .2s ease;
}
.ng-doc:hover { background: var(--ng-surface); border-color: var(--ng-border); }
.ng-doc-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ng-doc-meta { color: var(--ng-muted); font-size: .72rem; white-space: nowrap; }
.ng-side-footer { color: var(--ng-muted); font-size: .72rem; margin-top: 1.5rem; line-height: 1.5; }

/* ---------- Hero ---------- */
.ng-hero { text-align: center; padding: .25rem 0 1.4rem; animation: ngFadeUp .5s ease both; }
.ng-badge {
    display: inline-flex; gap: .4rem; align-items: center; font-size: .74rem; font-weight: 600;
    color: var(--ng-primary-2); background: rgba(4, 159, 217, .1);
    border: 1px solid rgba(4, 159, 217, .3); padding: .3rem .8rem; border-radius: 999px;
}
.ng-title {
    font-size: clamp(2.1rem, 5.5vw, 3.1rem); font-weight: 800; letter-spacing: -.03em;
    line-height: 1.1; margin: .8rem 0 .25rem;
    background: linear-gradient(90deg, #FFFFFF 0%, #7DD3FC 55%, var(--ng-primary) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.ng-subtitle { font-size: clamp(1rem, 2.4vw, 1.2rem); font-weight: 600; color: var(--ng-text); }
.ng-hero-text {
    color: var(--ng-muted); max-width: 620px; margin: .6rem auto 0;
    font-size: .95rem; line-height: 1.6;
}
.ng-pills { display: flex; flex-wrap: wrap; gap: .4rem; justify-content: center; margin-top: 1rem; }
.ng-pill {
    font-size: .72rem; color: var(--ng-muted); border: 1px solid var(--ng-border);
    padding: .25rem .65rem; border-radius: 999px; background: rgba(17, 26, 46, .6);
}
.ng-hero.compact { padding-bottom: .6rem; }
.ng-hero.compact .ng-title { font-size: clamp(1.6rem, 4vw, 2rem); margin-top: .5rem; }

/* ---------- Empty state / welcome card ---------- */
.ng-welcome {
    background: linear-gradient(180deg, rgba(22, 34, 59, .92), rgba(17, 26, 46, .92));
    border: 1px solid var(--ng-border); border-radius: 22px; padding: 1.75rem;
    box-shadow: var(--ng-shadow); animation: ngFadeUp .55s ease both;
}
.ng-ai-icon {
    width: 72px; height: 72px; margin: 0 auto 1rem; border-radius: 50%;
    display: grid; place-items: center; font-size: 2rem;
    background: radial-gradient(circle at 30% 30%, #38BDF8, #0369A1);
    animation: ngPulse 2.8s ease-in-out infinite;
}
.ng-welcome-title { text-align: center; font-weight: 700; font-size: 1.3rem; color: var(--ng-text); }
.ng-welcome-text {
    text-align: center; color: var(--ng-muted); max-width: 560px;
    margin: .45rem auto 1.3rem; line-height: 1.6; font-size: .92rem;
}
.ng-usecases { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }
.ng-usecase {
    background: rgba(11, 18, 32, .6); border: 1px solid var(--ng-border); border-radius: 14px;
    padding: .9rem; transition: transform .2s ease, border-color .2s ease;
}
.ng-usecase:hover { transform: translateY(-2px); border-color: var(--ng-border-strong); }
.ng-usecase-icon { font-size: 1.25rem; }
.ng-usecase-title { font-weight: 600; font-size: .88rem; margin-top: .35rem; color: var(--ng-text); }
.ng-usecase-text { color: var(--ng-muted); font-size: .78rem; margin-top: .2rem; line-height: 1.45; }
.ng-section-label {
    margin: 1.4rem 0 .2rem; font-size: .75rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; color: var(--ng-muted);
}

/* ---------- Buttons (suggested questions, clear chat) ---------- */
.stButton > button {
    width: 100%; justify-content: flex-start; text-align: left;
    background: var(--ng-surface); color: var(--ng-text);
    border: 1px solid var(--ng-border); border-radius: 14px; padding: .8rem 1rem;
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease, background .18s ease;
}
.stButton > button:hover {
    transform: translateY(-2px); border-color: var(--ng-primary); color: var(--ng-text);
    background: var(--ng-surface-2); box-shadow: 0 8px 22px -12px rgba(4, 159, 217, .8);
}
.stButton > button:focus:not(:active) { border-color: var(--ng-primary); color: var(--ng-text); }
.stButton > button:disabled { opacity: .5; transform: none; box-shadow: none; }

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: transparent; padding: .35rem 0; gap: .75rem;
    animation: ngFadeUp .35s ease both;
}
/* User messages: avatar on the right, bubble aligned right */
[data-testid="stChatMessage"]:has(.ng-user-marker) { flex-direction: row-reverse; }
.ng-user-bubble {
    margin-left: auto; width: fit-content; max-width: min(80%, 620px);
    background: linear-gradient(135deg, var(--ng-primary), #0369A1); color: #FFFFFF;
    padding: .7rem 1rem; border-radius: 18px 6px 18px 18px; line-height: 1.55;
    overflow-wrap: anywhere; box-shadow: 0 10px 24px -14px rgba(4, 159, 217, .9);
}
.ng-time { font-size: .68rem; opacity: .75; margin-top: .3rem; text-align: right; }
/* Assistant messages: left-aligned card */
[data-testid="stChatMessage"]:has(.ng-assistant-marker) [data-testid="stChatMessageContent"] {
    background: var(--ng-surface); border: 1px solid var(--ng-border);
    border-radius: 6px 18px 18px 18px; padding: .9rem 1.1rem; box-shadow: var(--ng-shadow);
}
.ng-meta { display: flex; align-items: center; gap: .45rem; font-size: .75rem; color: var(--ng-muted); margin-bottom: .35rem; }
.ng-meta b { color: var(--ng-primary-2); font-weight: 600; }

/* Typing indicator + skeleton loader */
.ng-typing { display: inline-flex; align-items: center; gap: .65rem; color: var(--ng-muted); font-size: .85rem; }
.ng-dots { display: inline-flex; gap: 4px; }
.ng-dots span {
    width: 7px; height: 7px; border-radius: 50%; background: var(--ng-primary-2);
    animation: ngBounce 1.2s infinite ease-in-out;
}
.ng-dots span:nth-child(2) { animation-delay: .15s; }
.ng-dots span:nth-child(3) { animation-delay: .3s; }
.ng-skeleton {
    height: 10px; border-radius: 6px; margin-top: .6rem;
    background: linear-gradient(90deg, rgba(148,163,184,.07) 25%, rgba(148,163,184,.18) 50%, rgba(148,163,184,.07) 75%);
    background-size: 200% 100%; animation: ngShimmer 1.4s infinite linear;
}

/* Per-answer statistics chips */
.ng-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .8rem 0 .5rem; }
.ng-chip {
    display: inline-flex; align-items: center; gap: .3rem; font-size: .72rem;
    padding: .25rem .6rem; border-radius: 999px; color: #BAE6FD;
    background: rgba(4, 159, 217, .08); border: 1px solid rgba(4, 159, 217, .25);
}
.ng-chip.doc { color: #A7F3D0; background: rgba(110, 231, 183, .07); border-color: rgba(110, 231, 183, .25); }
.ng-chip.warn { color: #FDE68A; background: rgba(251, 191, 36, .08); border-color: rgba(251, 191, 36, .3); }

/* ---------- Expanders ---------- */
[data-testid="stExpander"] > details {
    border: 1px solid var(--ng-border); border-radius: 12px;
    background: rgba(11, 18, 32, .45); transition: border-color .2s ease;
}
[data-testid="stExpander"] > details:hover { border-color: var(--ng-border-strong); }

/* Source cards (native <details> so each card expands on its own) */
details.ng-source {
    background: var(--ng-bg); border: 1px solid var(--ng-border); border-radius: 12px;
    margin-bottom: .5rem; overflow: hidden; transition: border-color .2s ease, transform .2s ease;
}
details.ng-source:hover { border-color: var(--ng-border-strong); transform: translateY(-1px); }
details.ng-source > summary {
    list-style: none; cursor: pointer; padding: .7rem .85rem;
    display: flex; align-items: center; gap: .65rem;
}
details.ng-source > summary::-webkit-details-marker { display: none; }
.ng-rank {
    flex: none; display: inline-grid; place-items: center; min-width: 28px; height: 26px; padding: 0 .3rem;
    border-radius: 8px; font-size: .72rem; font-weight: 700;
    background: rgba(4, 159, 217, .15); color: var(--ng-primary-2);
}
.ng-source-head { flex: 1; min-width: 0; }
.ng-source-name { display: block; font-weight: 600; font-size: .85rem; color: var(--ng-text); }
.ng-source-preview {
    display: block; color: var(--ng-muted); font-size: .78rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ng-score { flex: none; font-family: var(--ng-mono); font-size: .75rem; color: var(--ng-accent); }
.ng-source-body {
    border-top: 1px dashed var(--ng-border); padding: .75rem .85rem .85rem;
    font-family: var(--ng-mono); font-size: .76rem; line-height: 1.65; color: #CBD5E1;
    overflow-wrap: anywhere;
}

/* Retrieval panel */
.ng-ret-head { color: var(--ng-muted); font-size: .78rem; margin-bottom: .4rem; }
.ng-ret-row {
    display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 2fr) auto;
    gap: .75rem; align-items: center; padding: .45rem 0; font-size: .82rem;
    border-bottom: 1px solid rgba(148, 163, 184, .08);
}
.ng-ret-doc { display: flex; align-items: center; gap: .5rem; min-width: 0; color: var(--ng-text); }
.ng-ret-doc-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ng-ret-chunk { color: var(--ng-muted); font-size: .7rem; white-space: nowrap; }
.ng-bar { height: 8px; background: rgba(148, 163, 184, .12); border-radius: 999px; overflow: hidden; }
.ng-bar > span {
    display: block; height: 100%; border-radius: 999px; transform-origin: left;
    background: linear-gradient(90deg, var(--ng-primary), var(--ng-accent));
    animation: ngGrow .8s ease both;
}
.ng-bar > span.mid { background: linear-gradient(90deg, #0EA5E9, var(--ng-warn)); }
.ng-bar > span.low { background: #64748B; }
.ng-ret-score { font-family: var(--ng-mono); font-size: .78rem; color: var(--ng-text); }

/* ---------- Alert cards (warnings / errors) ---------- */
.ng-alert {
    display: flex; gap: .9rem; align-items: flex-start; border-radius: 16px;
    padding: 1rem 1.15rem; border: 1px solid; margin: .25rem 0 1rem;
    animation: ngFadeUp .4s ease both;
}
.ng-alert-warn { background: linear-gradient(135deg, rgba(251,191,36,.12), rgba(251,191,36,.03)); border-color: rgba(251,191,36,.35); }
.ng-alert-error { background: linear-gradient(135deg, rgba(248,113,113,.12), rgba(248,113,113,.03)); border-color: rgba(248,113,113,.35); }
.ng-alert-icon { font-size: 1.4rem; line-height: 1.2; }
.ng-alert-title { font-weight: 700; margin-bottom: .25rem; color: var(--ng-text); }
.ng-alert-body { color: #CBD5E1; font-size: .88rem; line-height: 1.6; }
.ng-alert-body a { color: var(--ng-primary-2); }
.ng-alert code, .ng-code {
    font-family: var(--ng-mono); font-size: .78rem; background: rgba(11, 18, 32, .7);
    border: 1px solid var(--ng-border); border-radius: 6px; padding: .05rem .35rem; color: #E2E8F0;
}
.ng-code { display: block; margin-top: .6rem; padding: .55rem .75rem; }
.ng-steps { margin-top: .5rem; display: grid; gap: .3rem; }
.ng-step-num {
    display: inline-grid; place-items: center; width: 20px; height: 20px; margin-right: .5rem;
    border-radius: 50%; font-size: .7rem; font-weight: 700; background: rgba(251,191,36,.2); color: #FDE68A;
}

/* ---------- RAG guardrail (out-of-scope) buttons ---------- */
.ng-guardrail-note {
    color: var(--ng-muted); font-size: .84rem; line-height: 1.6; margin: .15rem 0 .7rem;
    padding: .7rem .9rem; border-radius: 12px;
    background: rgba(251, 191, 36, .06); border: 1px dashed rgba(251, 191, 36, .35);
}
.ng-guardrail-note b { color: #FDE68A; }
.st-key-ng-guardrail-main .stButton > button,
.st-key-ng-guardrail-side .stButton > button {
    border-color: rgba(251, 191, 36, .3); background: rgba(251, 191, 36, .05);
}
.st-key-ng-guardrail-main .stButton > button:hover,
.st-key-ng-guardrail-side .stButton > button:hover {
    border-color: var(--ng-warn); box-shadow: 0 8px 22px -12px rgba(251, 191, 36, .7);
}
.st-key-ng-guardrail-side .stButton > button { padding: .5rem .7rem; font-size: .8rem; }
.st-key-ng-guardrail-side { gap: .4rem; }

/* ---------- View switch + evaluation page ---------- */
.st-key-ng-topbar [data-testid="stButtonGroup"] { justify-content: center; }
.st-key-ng-topbar .stButton > button { width: auto; padding: .45rem .95rem; border-radius: 999px; font-size: .85rem; }
.st-key-home_side .stButton > button, .st-key-home_side button {
    justify-content: center; font-weight: 600; color: #FFFFFF; border: none;
    background: linear-gradient(135deg, var(--ng-primary), #0369A1);
    box-shadow: 0 10px 24px -14px rgba(4, 159, 217, .9);
}
.st-key-home_side button:hover { color: #FFFFFF; filter: brightness(1.1); }
section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button { padding: .5rem .7rem; font-size: .8rem; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary { font-size: .88rem; }
.ng-eval-intro {
    background: linear-gradient(180deg, rgba(22, 34, 59, .92), rgba(17, 26, 46, .92));
    border: 1px solid var(--ng-border); border-radius: 18px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem; animation: ngFadeUp .45s ease both;
}
.ng-eval-intro .ng-welcome-text { margin-bottom: 0; }
.ng-eval-intro code { font-family: var(--ng-mono); font-size: .8rem; color: var(--ng-primary-2); }
.ng-eval-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .6rem; margin-bottom: 1rem; }
.ng-vs { font-size: .75rem; color: var(--ng-muted); font-weight: 500; }

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 3px rgba(4, 159, 217, .18); }

/* ---------- Animations ---------- */
@keyframes ngFadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes ngPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(4, 159, 217, .45); }
    50% { box-shadow: 0 0 0 14px rgba(4, 159, 217, 0); }
}
@keyframes ngBounce {
    0%, 80%, 100% { transform: translateY(0); opacity: .35; }
    40% { transform: translateY(-5px); opacity: 1; }
}
@keyframes ngGrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes ngShimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* ---------- Responsive (tablet & mobile) ---------- */
@media (max-width: 900px) {
    .ng-usecases { grid-template-columns: 1fr 1fr; }
    .ng-eval-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 640px) {
    .block-container, [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; padding-top: 1.5rem; }
    .ng-usecases { grid-template-columns: 1fr; }
    .ng-welcome { padding: 1.2rem; border-radius: 18px; }
    .ng-user-bubble { max-width: 88%; }
    .ng-ret-row { grid-template-columns: minmax(0, 1fr) auto; }
    .ng-ret-row .ng-bar { grid-column: 1 / -1; grid-row: 2; }
    .ng-hero-text { font-size: .88rem; }
}
"""


# =============================================================================
# 3. SMALL HELPERS
# =============================================================================

def esc(text) -> str:
    """Escape text so it can be placed safely inside HTML."""
    return html.escape(str(text))


def esc_multiline(text: str) -> str:
    """Escape text and keep its line breaks."""
    return esc(text).replace("\n", "<br>")


def shorten(text: str, limit: int) -> str:
    """Collapse whitespace and cut text to `limit` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def plural(count: int, word: str) -> str:
    """'1 document', '3 documents'."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def current_time() -> str:
    """
    Timestamp shown under each chat message, in Thailand time (UTC+7).
    Streamlit Cloud servers run on UTC, so the offset is set explicitly.
    """
    return datetime.now(DISPLAY_TIMEZONE).strftime("%H:%M")


def render_html(markup: str, target=None) -> None:
    """
    Render raw HTML with st.markdown.

    Every line is stripped and blank lines are removed. Otherwise Markdown
    would treat indented lines as code blocks and blank lines would end
    the HTML block early.
    """
    clean = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    (target or st).markdown(clean, unsafe_allow_html=True)


def render_alert(kind: str, icon: str, title: str, body_html: str, target=None) -> None:
    """Show a styled warning/error card. `body_html` must already be escaped."""
    render_html(
        f"""
        <div class="ng-alert ng-alert-{kind}">
            <div class="ng-alert-icon">{icon}</div>
            <div>
                <div class="ng-alert-title">{esc(title)}</div>
                <div class="ng-alert-body">{body_html}</div>
            </div>
        </div>
        """,
        target,
    )


# =============================================================================
# 4. DOCUMENT PROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """Normalise line endings, remove control characters and extra whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control characters
    text = re.sub(r"[^\S\n]+", " ", text)                          # many spaces/tabs -> one space
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)                         # max one blank line
    return text.strip()


def load_documents(data_dir: Path) -> list[dict]:
    """Read every .txt file in `data_dir` and return a list of cleaned documents."""
    if not data_dir.exists():
        raise FileNotFoundError(f"The data folder was not found: {data_dir}")

    documents = []
    for path in sorted(data_dir.glob("*.txt")):
        # utf-8-sig also removes a BOM if the file has one
        text = clean_text(path.read_text(encoding="utf-8-sig"))
        if text:
            documents.append({
                "name": path.name,
                "topic": path.stem.replace("_", " "),
                "text": text,
            })

    if not documents:
        raise ValueError(f"No .txt documents were found in {data_dir}")
    return documents


def _find_break(window: str, min_pos: int) -> int:
    """
    Find a natural place to end a chunk: a paragraph break first,
    then a line break, then the end of a sentence, then a space.
    """
    for separator in ("\n\n", "\n", ". ", " "):
        pos = window.rfind(separator)
        if pos >= min_pos:
            return pos + len(separator)
    return len(window)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks of about `chunk_size` characters.
    Consecutive chunks share about `overlap` characters so that
    information on a chunk boundary is not lost.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Do not cut in the middle of a sentence if we can avoid it
            end = start + _find_break(text[start:end], min_pos=int(chunk_size * 0.6))

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break

        # Step back by `overlap` characters, then move to the next word start
        next_start = end - overlap
        match = re.compile(r"\s").search(text, next_start, end)
        start = match.end() if match else next_start

    return chunks


# =============================================================================
# 5. EMBEDDINGS + VECTOR INDEX (cached so they are built only once)
# =============================================================================

@st.cache_resource(show_spinner="Loading the embedding model…")
def load_embedding_model() -> SentenceTransformer:
    """Download (first run only) and load the sentence-transformers model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """
    Turn texts into vectors. Vectors are L2-normalised, so the inner product
    used by FAISS equals cosine similarity (1.0 = same meaning).
    """
    model = load_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


@st.cache_resource(show_spinner="Building the FAISS vector index…")
def build_knowledge_base() -> dict:
    """
    Load documents, split them into chunks, embed the chunks and
    store them in a FAISS index. Runs once when the app starts.
    """
    documents = load_documents(DATA_DIR)

    chunks = []
    for doc in documents:
        doc_chunks = split_into_chunks(doc["text"])
        doc["chunk_count"] = len(doc_chunks)
        for number, chunk_text in enumerate(doc_chunks, start=1):
            chunks.append({
                "source": doc["name"],
                "chunk_id": number,
                "text": chunk_text,
                # The topic name is added to the embedded text so chunks that
                # contain only commands still "know" which topic they belong to.
                "embed_text": f"{doc['topic']}: {chunk_text}",
            })

    vectors = embed([c["embed_text"] for c in chunks])

    # IndexFlatIP = exact search using inner product (cosine similarity here)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return {"documents": documents, "chunks": chunks, "index": index}


# =============================================================================
# 6. RETRIEVAL
# =============================================================================

def search_index(query: str, kb: dict, k: int = TOP_K) -> dict[int, float]:
    """Search FAISS for one query. Returns {chunk index: similarity score}."""
    query_vector = embed([query])
    k = min(k, len(kb["chunks"]))
    scores, ids = kb["index"].search(query_vector, k)
    # FAISS returns -1 when there are fewer results than requested
    return {int(idx): float(score) for score, idx in zip(scores[0], ids[0]) if idx >= 0}


def looks_like_follow_up(question: str) -> bool:
    """True if the question refers back to something ("How do I configure it?")."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    return bool(words & FOLLOWUP_WORDS) or any(w in question for w in FOLLOWUP_WORDS_THAI)


def retrieve(question: str, kb: dict, previous_question: str | None = None, k: int = TOP_K) -> list[dict]:
    """
    Return the top-k chunks most similar to the question (best first).

    Follow-up questions such as "How do I configure it?" do not say what
    "it" is, so they match poorly on their own. When the question looks like
    a follow-up (weak best match, or words like "it"/"this") and there is a
    previous question, we search again with both questions joined together
    and keep the best score for every chunk.
    """
    hits = search_index(question, kb, k)

    best_score = max(hits.values(), default=0.0)
    is_follow_up = best_score < FOLLOWUP_THRESHOLD or looks_like_follow_up(question)
    if previous_question and is_follow_up:
        for idx, score in search_index(f"{previous_question} {question}", kb, k).items():
            hits[idx] = max(score, hits.get(idx, 0.0))

    # Best first, drop irrelevant chunks, keep only the top k
    ranked = sorted(hits.items(), key=lambda item: item[1], reverse=True)
    ranked = [(idx, score) for idx, score in ranked if score >= MIN_SIMILARITY][:k]

    results = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        chunk = kb["chunks"][idx]
        results.append({
            "rank": rank,
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score,
        })
    return results


def unique_documents(sources: list[dict]) -> list[str]:
    """Document names in the order they were retrieved, without duplicates."""
    return list(dict.fromkeys(s["source"] for s in sources))


def build_context(sources: list[dict]) -> str:
    """Join retrieved chunks into one context string, labelled with their source file."""
    blocks = [
        f"[Source {s['rank']}: {s['source']}]\n{s['text']}"
        for s in sources
    ]
    return "\n\n---\n\n".join(blocks)


# =============================================================================
# 7. LLM (Groq)
# =============================================================================

def get_api_key() -> str | None:
    """
    Read the Groq API key from st.secrets["GROQ_API_KEY"].
    Falls back to the GROQ_API_KEY environment variable.
    Returns None if no key is configured.
    """
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:  # no secrets file, or the key is missing
        key = os.environ.get("GROQ_API_KEY", "")
    key = str(key).strip()
    return key or None


@st.cache_resource
def get_groq_client(api_key: str) -> Groq:
    """Create the Groq client once and reuse it."""
    return Groq(api_key=api_key)


def generate_answer(api_key: str, question: str, context: str, history: list[dict]) -> str:
    """Send the prompt (system prompt + recent chat history + question) to the LLM."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, question=question)
    messages = [{"role": "system", "content": system_prompt}]

    # Short conversation memory so follow-up questions read naturally.
    # Failed and "not found" answers are skipped: if the model sees several
    # earlier refusals it tends to copy them even when the context has the answer.
    usable_history = [
        m for m in history
        if m.get("content") and m.get("status") not in ("error", "not_found", "no_docs")
    ]
    for message in usable_history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": message["role"], "content": message["content"]})

    messages.append({"role": "user", "content": question})

    response = get_groq_client(api_key).chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def is_thai(text: str) -> bool:
    """True if the text contains Thai characters (Unicode block U+0E00-U+0E7F)."""
    return bool(re.search(r"[฀-๿]", text))


def not_found_message(question: str) -> str:
    """The 'not found' reply in the same language as the question."""
    return NOT_FOUND_MESSAGE if is_thai(question) else NOT_FOUND_MESSAGE_EN


# The LLM sometimes words the refusal slightly differently, for example
# "ไม่พบบทความในเอกสารที่มี" or "No relevant information found in the documents".
# These patterns catch such variants, but only in SHORT replies, so a real
# answer that happens to contain similar words is never hidden.
NOT_FOUND_PATTERNS = [
    r"ไม่พบ.{0,20}ในเอกสาร",
    r"\b(no|not)\b.{0,40}\b(information|answer|data)\b.{0,40}\b(found|available)\b",
    r"\b(not|cannot be)\b.{0,20}\bfound\b.{0,30}\bdocuments?\b",
]
SHORT_REPLY_CHARS = 200


def is_not_found(answer: str) -> bool:
    """True if the LLM said the answer is not in the documents (in either language)."""
    if NOT_FOUND_MESSAGE in answer or NOT_FOUND_MESSAGE_EN.lower() in answer.lower():
        return True
    if len(answer) <= SHORT_REPLY_CHARS:
        return any(re.search(p, answer, flags=re.IGNORECASE) for p in NOT_FOUND_PATTERNS)
    return False


def localize_not_found(answer: str, question: str) -> str:
    """Show the 'not found' reply in the same language as the question."""
    target = not_found_message(question)
    if len(answer) <= SHORT_REPLY_CHARS:
        return target  # the whole reply is a refusal: show the standard message
    # Long reply with a refusal sentence inside: swap just that sentence
    answer = answer.replace(NOT_FOUND_MESSAGE, target)
    return re.sub(re.escape(NOT_FOUND_MESSAGE_EN), target, answer, flags=re.IGNORECASE)


def friendly_error(exc: Exception) -> str:
    """Turn API exceptions into messages a user can act on."""
    if isinstance(exc, AuthenticationError):
        return "The Groq API key was rejected. Check GROQ_API_KEY in your secrets."
    if isinstance(exc, RateLimitError):
        return "The Groq rate limit was reached. Please wait a moment and try again."
    if isinstance(exc, APIConnectionError):
        return "Could not connect to the Groq API. Check your internet connection."
    if isinstance(exc, NotFoundError):
        return (
            f"The model '{LLM_MODEL}' is not available on Groq (it may have been retired). "
            "Change LLM_MODEL in app.py to a model listed at console.groq.com/docs/models."
        )
    return f"Unexpected error: {exc}"


# =============================================================================
# 8. SESSION STATE
# =============================================================================

def init_session_state() -> None:
    """Create the chat history the first time the app runs in a browser session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "view" not in st.session_state:
        st.session_state.view = VIEW_CHAT  # selected page (Chat / Evaluation)


def queue_question(question: str) -> None:
    """Button callback: send a suggested question on the next run."""
    st.session_state.pending_question = question
    st.session_state.view = VIEW_CHAT  # jump back to the chat if another view is open


def go_home() -> None:
    """Button callback: clear the conversation and show the welcome screen."""
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.view = VIEW_CHAT


# =============================================================================
# 9. UI COMPONENTS
# =============================================================================

def inject_css() -> None:
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


def render_sidebar(kb: dict, api_ready: bool) -> None:
    """Logo, Home button, quick questions, statistics and the loaded documents."""
    documents = kb["documents"]
    with st.sidebar:
        status = (
            '<span class="ng-status ok"><span class="ng-dot"></span>Groq API key loaded</span>'
            if api_ready
            else '<span class="ng-status off"><span class="ng-dot"></span>Groq API key missing</span>'
        )
        render_html(f"""
            <div class="ng-brand">
                <div class="ng-logo">🌐</div>
                <div>
                    <div class="ng-brand-name">{APP_NAME}</div>
                    <div class="ng-brand-sub">{APP_TAGLINE}</div>
                </div>
            </div>
            <div class="ng-side-desc">
                A Retrieval-Augmented Generation assistant that answers CCNA questions
                using only its knowledge base, and shows the sources behind every answer.
            </div>
            {status}
        """)

        # Most-used actions first, so they never get lost below long lists
        st.write("")
        st.button(
            "🏠  Home · New chat",
            key="home_side",
            on_click=go_home,
            help="Back to the welcome screen and start a new conversation",
        )

        # Example questions stay available after the welcome screen is gone
        with st.expander("💡 Example questions"):
            for i, question in enumerate(SUGGESTED_QUESTIONS):
                st.button(
                    f"💬  {question}",
                    key=f"suggestion_side_{i}",
                    on_click=queue_question,
                    args=(question,),
                    disabled=not api_ready,
                )

        # Out-of-scope questions, so a reviewer can test the "answer only from
        # the documents" rule at any point in a conversation.
        with st.expander("🧪 RAG Guardrail Test", expanded=True):
            render_html(f"""
                <div class="ng-side-desc">
                    Not in the documents. Expected reply: <b>{NOT_FOUND_MESSAGE_EN}</b>
                    (Thai questions: <b>{NOT_FOUND_MESSAGE}</b>)
                </div>
            """)
            with st.container(key="ng-guardrail-side"):
                for i, question in enumerate(OUT_OF_SCOPE_QUESTIONS):
                    st.button(
                        f"🚫  {question}",
                        key=f"guardrail_side_{i}",
                        on_click=queue_question,
                        args=(question,),
                        disabled=not api_ready,
                    )

        render_html(f"""
            <div class="ng-section-title">Statistics</div>
            <div class="ng-stat-grid">
                <div class="ng-stat"><div class="ng-stat-value">{len(documents)}</div><div class="ng-stat-label">Documents loaded</div></div>
                <div class="ng-stat"><div class="ng-stat-value">{len(kb["chunks"])}</div><div class="ng-stat-label">Chunks indexed</div></div>
                <div class="ng-stat"><div class="ng-stat-value">{TOP_K}</div><div class="ng-stat-label">Top-K retrieval</div></div>
                <div class="ng-stat"><div class="ng-stat-value">{CHUNK_SIZE}</div><div class="ng-stat-label">Chunk size (chars)</div></div>
                <div class="ng-stat wide"><div class="ng-stat-value small">{EMBEDDING_MODEL}</div><div class="ng-stat-label">Embedding model</div></div>
                <div class="ng-stat wide"><div class="ng-stat-value small">{LLM_MODEL}</div><div class="ng-stat-label">LLM model (Groq)</div></div>
            </div>
        """)

        st.write("")
        with st.expander(f"📚 Knowledge Base · {len(documents)} documents"):
            doc_rows = "".join(
                f'<div class="ng-doc"><span class="ng-doc-name">📄 {esc(doc["name"])}</span>'
                f'<span class="ng-doc-meta">{doc["chunk_count"]} chunks</span></div>'
                for doc in documents
            )
            render_html(f'<div class="ng-doc-list">{doc_rows}</div>')

        render_html(
            '<div class="ng-side-footer">Built with Streamlit · Sentence-Transformers · FAISS · Groq</div>'
        )


def render_hero(compact: bool) -> None:
    """Title section at the top of the main area."""
    if compact:
        render_html(f"""
            <div class="ng-hero compact">
                <div class="ng-title">{APP_NAME}</div>
                <div class="ng-subtitle">{APP_TAGLINE}</div>
            </div>
        """)
        return

    render_html(f"""
        <div class="ng-hero">
            <span class="ng-badge">⚡ Retrieval-Augmented Generation</span>
            <div class="ng-title">{APP_NAME}</div>
            <div class="ng-subtitle">{APP_TAGLINE}</div>
            <div class="ng-hero-text">
                Ask anything about switching, routing and network services. NetGuide AI searches
                a curated CCNA knowledge base, retrieves the most relevant passages, and answers
                using only those sources.
            </div>
            <div class="ng-pills">
                <span class="ng-pill">🧠 {EMBEDDING_MODEL}</span>
                <span class="ng-pill">🗂️ FAISS vector search</span>
                <span class="ng-pill">⚡ Qwen 3.8 27B · Groq</span>
            </div>
        </div>
    """)


def render_api_key_warning() -> None:
    """Panel shown when GROQ_API_KEY is not configured."""
    render_alert(
        "warn",
        "🔑",
        "Groq API key not found",
        """
        NetGuide AI needs a <b>Groq API key</b> to generate answers. The knowledge base is
        loaded, but chat is disabled until a key is added.
        <div class="ng-steps">
            <div><span class="ng-step-num">1</span>Get a free key at
            <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a></div>
            <div><span class="ng-step-num">2</span>Local: add it to <code>.streamlit/secrets.toml</code></div>
            <div><span class="ng-step-num">3</span>Streamlit Cloud: <b>App settings → Secrets</b>, then reboot the app</div>
        </div>
        <code class="ng-code">GROQ_API_KEY = "gsk_..."</code>
        """,
    )


def render_empty_state(api_ready: bool) -> None:
    """Welcome card + suggested questions, shown before the first message."""
    cards = "".join(
        f'<div class="ng-usecase"><div class="ng-usecase-icon">{icon}</div>'
        f'<div class="ng-usecase-title">{title}</div>'
        f'<div class="ng-usecase-text">{text}</div></div>'
        for icon, title, text in USE_CASES
    )
    render_html(f"""
        <div class="ng-welcome">
            <div class="ng-ai-icon">🤖</div>
            <div class="ng-welcome-title">Welcome to NetGuide AI</div>
            <div class="ng-welcome-text">
                Your study partner for Cisco CCNA. Every answer is grounded in the knowledge base
                and comes with the exact passages it was built from, so you can check each fact.
            </div>
            <div class="ng-usecases">{cards}</div>
        </div>
        <div class="ng-section-label">💡 Try asking</div>
    """)

    # Clicking a suggestion stores it and Streamlit reruns the script,
    # which then answers it exactly like a typed question.
    columns = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        with columns[i % 2]:
            st.button(
                f"💬  {question}",
                key=f"suggestion_{i}",
                on_click=queue_question,
                args=(question,),
                disabled=not api_ready,
            )

    render_html(f"""
        <div class="ng-section-label">🧪 Test the RAG guardrail · not in the documents</div>
        <div class="ng-guardrail-note">
            The AI model already knows these answers from its training, but they are
            <b>not</b> in the knowledge base. A real RAG system must refuse instead of using outside
            knowledge: <b>{NOT_FOUND_MESSAGE_EN}</b> (or <b>{NOT_FOUND_MESSAGE}</b> for a Thai question).
        </div>
    """)
    with st.container(key="ng-guardrail-main"):
        columns = st.columns(2)
        for i, question in enumerate(OUT_OF_SCOPE_QUESTIONS):
            with columns[i % 2]:
                st.button(
                    f"🚫  {question}",
                    key=f"guardrail_main_{i}",
                    on_click=queue_question,
                    args=(question,),
                    disabled=not api_ready,
                )


def typing_indicator_html(step: str) -> str:
    """Animated dots and skeleton lines shown while an answer is being generated."""
    return f"""
        <div class="ng-assistant-marker ng-typing">
            <span class="ng-dots"><span></span><span></span><span></span></span>
            <span>{esc(step)}</span>
        </div>
        <div class="ng-skeleton" style="width: 92%"></div>
        <div class="ng-skeleton" style="width: 76%"></div>
        <div class="ng-skeleton" style="width: 54%"></div>
    """


def render_answer_stats(message: dict) -> None:
    """Response time, number of chunks and referenced documents for one answer."""
    sources = message["sources"]
    documents = unique_documents(sources)
    chips = [
        f'<span class="ng-chip">⏱️ {message["response_time"]:.2f}s response</span>',
        f'<span class="ng-chip">🧩 {plural(len(sources), "chunk")} retrieved</span>',
        f'<span class="ng-chip">📄 {plural(len(documents), "document")} referenced</span>',
    ]
    if message["status"] == "not_found":
        chips.append('<span class="ng-chip warn">🚫 Answer not in documents</span>')
    chips += [f'<span class="ng-chip doc">{esc(name)}</span>' for name in documents]
    render_html(f'<div class="ng-chips">{"".join(chips)}</div>')


def render_sources(sources: list[dict]) -> None:
    """'Sources Used' expander containing one expandable card per retrieved chunk."""
    with st.expander(f"📚 Sources Used ({len(sources)})"):
        cards = "".join(
            f"""
            <details class="ng-source">
                <summary>
                    <span class="ng-rank">#{s["rank"]}</span>
                    <span class="ng-source-head">
                        <span class="ng-source-name">📄 {esc(s["source"])} · chunk {s["chunk_id"]}</span>
                        <span class="ng-source-preview">{esc(shorten(s["text"], 150))}</span>
                    </span>
                    <span class="ng-score">{s["score"]:.2f}</span>
                </summary>
                <div class="ng-source-body">{esc_multiline(s["text"])}</div>
            </details>
            """
            for s in sources
        )
        render_html(cards)


def render_retrieval_panel(sources: list[dict]) -> None:
    """Bar chart of similarity scores for the retrieved chunks (great for demos)."""
    with st.expander("🔎 Retrieval Panel · Top Retrieved Chunks"):
        rows = []
        for s in sources:
            width = max(0.0, min(1.0, s["score"])) * 100
            level = "high" if s["score"] >= 0.6 else "mid" if s["score"] >= 0.4 else "low"
            rows.append(f"""
                <div class="ng-ret-row">
                    <div class="ng-ret-doc">
                        <span class="ng-rank">#{s["rank"]}</span>
                        <span class="ng-ret-doc-name">{esc(s["source"])}</span>
                        <span class="ng-ret-chunk">chunk {s["chunk_id"]}</span>
                    </div>
                    <div class="ng-bar"><span class="{level}" style="width: {width:.0f}%"></span></div>
                    <div class="ng-ret-score">{s["score"]:.2f}</div>
                </div>
            """)
        header = (
            '<div class="ng-ret-head">Cosine similarity between your question and each chunk '
            "(1.00 = identical meaning).</div>"
        )
        render_html(header + "".join(rows))


def render_assistant_body(message: dict) -> None:
    """Everything inside an assistant bubble: header, answer, stats and sources."""
    render_html(
        f'<div class="ng-assistant-marker ng-meta"><b>{APP_NAME}</b>'
        f'<span>·</span><span>{esc(message["time"])}</span></div>'
    )

    if message["status"] == "no_docs":
        render_alert(
            "warn",
            "🔍",
            "No relevant documents found.",
            f"{esc(message['content'])}<br>None of the knowledge-base chunks are similar enough "
            "to your question. Try rephrasing it, or ask about one of the topics in the sidebar.",
        )
    elif message["status"] == "error":
        render_alert("error", "⚠️", "Could not generate an answer", esc(message["error"]))
    else:
        # The LLM answer is Markdown, so it is rendered as Markdown (not raw HTML)
        st.markdown(message["content"])

    render_answer_stats(message)
    if message["sources"]:
        render_sources(message["sources"])
        render_retrieval_panel(message["sources"])


def render_user_message(message: dict) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        render_html(
            f'<div class="ng-user-marker ng-user-bubble">{esc_multiline(message["content"])}'
            f'<div class="ng-time">{esc(message["time"])}</div></div>'
        )


def render_chat_history() -> None:
    """Re-draw every stored message (Streamlit reruns the whole script on each action)."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            render_user_message(message)
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                render_assistant_body(message)


# =============================================================================
# 10. EVALUATION VIEW (retrieval quality on test_questions.csv)
# =============================================================================

def load_test_questions() -> list[dict]:
    """Read test_questions.csv (columns: id, question, expected_source, answerable, ...)."""
    with open(TEST_QUESTIONS_FILE, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@st.cache_data(show_spinner="Running the retrieval evaluation…")
def run_retrieval_evaluation(_kb: dict) -> dict:
    """
    Run every test question through retrieve() only (no LLM, so no API cost).
    For answerable questions, check where the expected document appears in
    the top-k chunks. The leading underscore tells Streamlit not to hash _kb.
    """
    rows = []
    for item in load_test_questions():
        sources = retrieve(item["question"], _kb)
        retrieved = [s["source"] for s in sources]
        answerable = item["answerable"].strip().lower() == "yes"
        expected = item["expected_source"]
        rank = retrieved.index(expected) + 1 if answerable and expected in retrieved else None

        if not answerable:
            result = "🧪 LLM must refuse"
        elif rank:
            result = f"✅ Hit (rank {rank})"
        else:
            result = "❌ Miss"

        rows.append({
            "ID": int(item["id"]),
            "Question": item["question"],
            "Type": "Answerable" if answerable else "Out-of-scope",
            "Expected document": expected if answerable else "—",
            "Top retrieved": retrieved[0] if retrieved else "—",
            "Best score": sources[0]["score"] if sources else 0.0,
            "Result": result,
            "_rank": rank,
        })

    in_scope = [r for r in rows if r["Type"] == "Answerable"]
    out_scope = [r for r in rows if r["Type"] == "Out-of-scope"]

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "answerable": len(in_scope),
        "out_of_scope": len(out_scope),
        "hit_at_k": sum(1 for r in in_scope if r["_rank"]),
        "hit_at_1": sum(1 for r in in_scope if r["_rank"] == 1),
        # Mean Reciprocal Rank: 1.0 if the right document is always ranked first
        "mrr": average([1 / r["_rank"] if r["_rank"] else 0.0 for r in in_scope]),
        "score_in": average([r["Best score"] for r in in_scope]),
        "score_out": average([r["Best score"] for r in out_scope]),
    }
    return {"rows": rows, "metrics": metrics}


def render_evaluation_view(kb: dict) -> None:
    """Metrics + per-question table showing how well retrieval finds the right document."""
    render_html(f"""
        <div class="ng-eval-intro">
            <div class="ng-welcome-title">📊 Retrieval Evaluation</div>
            <div class="ng-welcome-text">
                Every question in <code>test_questions.csv</code> is embedded and searched with the
                same FAISS index the chat uses. For answerable questions we check whether the
                expected document is among the top {TOP_K} retrieved chunks. This page calls
                <b>no LLM</b>, so it is instant and free.
            </div>
        </div>
    """)

    if not TEST_QUESTIONS_FILE.exists():
        render_alert("warn", "📄", "test_questions.csv not found",
                     "Add the file next to app.py to see the evaluation.")
        return

    report = run_retrieval_evaluation(kb)
    m = report["metrics"]
    n = max(m["answerable"], 1)
    render_html(f"""
        <div class="ng-eval-grid">
            <div class="ng-stat"><div class="ng-stat-value">{m["hit_at_k"]}/{m["answerable"]}</div>
                <div class="ng-stat-label">Hit@{TOP_K} · expected doc in top {TOP_K} ({m["hit_at_k"] / n:.0%})</div></div>
            <div class="ng-stat"><div class="ng-stat-value">{m["hit_at_1"]}/{m["answerable"]}</div>
                <div class="ng-stat-label">Hit@1 · expected doc ranked first ({m["hit_at_1"] / n:.0%})</div></div>
            <div class="ng-stat"><div class="ng-stat-value">{m["mrr"]:.2f}</div>
                <div class="ng-stat-label">MRR · Mean Reciprocal Rank (1.00 = perfect)</div></div>
            <div class="ng-stat"><div class="ng-stat-value">{m["score_in"]:.2f} <span class="ng-vs">vs</span> {m["score_out"]:.2f}</div>
                <div class="ng-stat-label">Avg best similarity · answerable vs out-of-scope</div></div>
        </div>
    """)

    choice = st.segmented_control(
        "Show questions",
        ["All", "Answerable", "Out-of-scope"],
        default="All",
        key="eval_filter",
    )
    rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in report["rows"]
        if choice in (None, "All") or row["Type"] == choice
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn(width="small"),
            "Question": st.column_config.TextColumn(width="large"),
            "Best score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    render_html(f"""
        <div class="ng-guardrail-note">
            <b>Why do out-of-scope questions still retrieve chunks?</b> Vector search always returns
            the closest passages, even when none of them is relevant. Their similarity is lower
            (about {m["score_out"]:.2f} on average vs {m["score_in"]:.2f} for answerable questions).
            The final safeguard is the system prompt: the LLM must answer <b>only</b> from the context,
            so it replies <b>{NOT_FOUND_MESSAGE_EN}</b> (<b>{NOT_FOUND_MESSAGE}</b> for Thai questions).
            Try the 🧪 questions in the sidebar to see this live.
        </div>
    """)


# =============================================================================
# 11. QUESTION HANDLING (the RAG pipeline end-to-end)
# =============================================================================

def handle_question(question: str, kb: dict, api_key: str) -> None:
    """Show the question, retrieve context, ask the LLM and store the answer."""
    history = list(st.session_state.messages)  # messages before this question

    user_message = {"role": "user", "content": question, "time": current_time()}
    st.session_state.messages.append(user_message)
    render_user_message(user_message)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        started = time.perf_counter()

        # Step 1: retrieval
        render_html(typing_indicator_html("Searching the knowledge base…"), placeholder)
        previous_questions = [m["content"] for m in history if m["role"] == "user"]
        sources = retrieve(question, kb, previous_question=previous_questions[-1] if previous_questions else None)

        answer = {
            "role": "assistant",
            "content": "",
            "time": current_time(),
            "sources": sources,
            "status": "ok",          # ok | not_found | no_docs | error
            "error": "",
            "response_time": 0.0,
        }

        # Step 2: generation
        if not sources:
            answer["status"] = "no_docs"
            answer["content"] = not_found_message(question)
        else:
            render_html(
                typing_indicator_html(f"Reading {len(sources)} chunks and writing an answer…"),
                placeholder,
            )
            try:
                answer["content"] = generate_answer(api_key, question, build_context(sources), history)
                if is_not_found(answer["content"]):
                    answer["status"] = "not_found"
                    # Thai question -> Thai message, otherwise English
                    answer["content"] = localize_not_found(answer["content"], question)
            except Exception as exc:  # show the problem instead of crashing the app
                answer["status"] = "error"
                answer["error"] = friendly_error(exc)

        answer["response_time"] = time.perf_counter() - started

        # Step 3: replace the typing indicator with the final answer
        placeholder.empty()
        render_assistant_body(answer)

    st.session_state.messages.append(answer)


# =============================================================================
# 12. MAIN APP
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="NetGuide AI · CCNA Assistant",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_session_state()

    # Build (or load from cache) the vector index
    try:
        kb = build_knowledge_base()
    except (FileNotFoundError, ValueError) as exc:
        render_alert("error", "📂", "Knowledge base could not be loaded", esc(exc))
        st.stop()

    api_key = get_api_key()
    render_sidebar(kb, api_ready=api_key is not None)

    # Top bar: Home button + switch between the chat and the evaluation page
    with st.container(key="ng-topbar"):
        left, middle, _ = st.columns([1, 2, 1], vertical_alignment="center")
        with left:
            st.button(
                "🏠  Home",
                key="home_top",
                on_click=go_home,
                help="Back to the welcome screen and start a new conversation",
            )
        with middle:
            view = st.segmented_control(
                "View",
                [VIEW_CHAT, VIEW_EVAL],
                key="view",  # starts as VIEW_CHAT, set in init_session_state()
                label_visibility="collapsed",
            )

    if view == VIEW_EVAL:
        render_hero(compact=True)
        render_evaluation_view(kb)
        return  # no chat input on this page

    # The chat input always stays pinned to the bottom of the page.
    typed_question = st.chat_input(
        "Ask about VLANs, STP, OSPF, ACLs, NAT…",
        disabled=api_key is None,
    )
    # A clicked suggestion takes priority over typed text
    question = (st.session_state.pending_question or typed_question or "").strip()
    st.session_state.pending_question = None

    has_conversation = bool(st.session_state.messages) or bool(question)
    render_hero(compact=has_conversation)

    if api_key is None:
        render_api_key_warning()

    if not has_conversation:
        render_empty_state(api_ready=api_key is not None)

    render_chat_history()

    if question and api_key:
        handle_question(question, kb, api_key)


if __name__ == "__main__":
    main()
