# ============================================================
#  10-K Financial Analyst RAG — Edition v7.6
#
#  v7.5 ROOT-CAUSE FIX — Statement chunks PINNED before scoring:
#    ★ FIX #1 — PINNED STATEMENTS: Statement chunks retrieved by metadata
#      are now PINNED (excluded from score-based filtering). Previously,
#      63 statement chunks were fetched but then 178→38 heuristic scoring
#      discarded most of them including ALL Alphabet balance sheet chunks.
#      Now: stmt chunks bypass scoring entirely and go straight to output.
#    ★ FIX #2 — GUARANTEED PER-COMPANY STATEMENT COVERAGE: For each
#      (company × statement_type) combination needed, at least MIN_STMT_PER
#      chunks are guaranteed in the final context regardless of scoring.
#    ★ FIX #3 — SCORING preserves company balance BEFORE filtering,
#      not after. Balance enforcement now runs on pre-scored pool.
#    ★ FIX #4 — INDEX_VERSION bumped to v7.5 to invalidate v7.4 cache.
#
#  v7.4 Fixes (still included):
#    ★ Microsoft BALANCE SHEETS / INCOME STATEMENTS pattern matching
#    ★ Cache version stamping
#    ★ Notes false-positive fix, 4000-char search window, IGNORECASE
#
#  v7.3 Fixes (still included):
#    ★ No index-time chunk truncation, per-page chunks, 12-page window,
#      60K verifier context, weighted budget, never "not available"
#
#  v7.2 Fixes (still included):
#    ★ Document-first, LaTeX math, subtotal detection, self-correction
#
#  Agentic RAG · LLM-Driven Query Analysis · HyDE · CRAG
#  MMR · Iterative Retrieval · Reranking · Dual-Model Verification
#  Self-Correction · Ollama Support
#
#  Requirements:
#    pip install streamlit langchain langchain-openai langchain-community \
#               faiss-cpu pdfplumber rank-bm25
#    Optional:  pip install langchain-ollama
# ============================================================

# ---------- Section 0: Environment ----------
import logging

logging.getLogger(
    "streamlit.runtime.scriptrunner_utils.script_run_context"
).setLevel(logging.ERROR)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------- Section 1: Imports ----------
import streamlit as st
import tempfile, time, re, json, hashlib, pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# ---------- Section 2: Configuration ----------
ZHIZENGZENG_BASE_URL = "https://api.zhizengzeng.com/v1"
CACHE_DIR = Path("./index_cache")
CACHE_DIR.mkdir(exist_ok=True)

CLOUD_LLM_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-flash-preview-04-17",
]
OLLAMA_LLM_MODELS = [
    "deepseek-r1:14b",
    "llama3:8b",
    "mistral:7b",
    "qwen2:7b",
]
CLOUD_EMBED_MODELS = [
    "text-embedding-3-large",
    "text-embedding-3-small",
]
OLLAMA_EMBED_MODELS = ["nomic-embed-text", "mxbai-embed-large"]

EMOJI_MAP = {
    "Amazon": "\U0001F7E0",
    "Alphabet_Google": "\U0001F535",
    "Microsoft": "\U0001F7E2",
    "Unknown": "\U000026AA",
}

DEFAULT_CONTEXT_BUDGET_CHARS = 120000
# ★ v7.3: Raised from 4→12 so thick 200+ page filings are fully read
MAX_STATEMENT_CONTINUATION_PAGES = 12
# ★ v7.3: REMOVED MAX_STATEMENT_CHUNK_CHARS — no more data destruction at index time
MAX_STATEMENT_CHUNK_CHARS = None  # Disabled — do NOT truncate at index time
MIN_CHUNK_LENGTH = 60
# ★ v7.6: ALL statement chunks are pinned (no quota limit).
# SCORED_POOL_EXTRA_SLOTS = how many extra non-statement chunks to add after pinned.
SCORED_POOL_EXTRA_SLOTS = 20

# --- 10-K Section Patterns ---
SECTION_PATTERNS = [
    (r"item\s*1[\.\s]+business", "Item1_Business"),
    (r"item\s*1a[\.\s]+risk\s*factors", "Item1A_RiskFactors"),
    (r"item\s*1b", "Item1B_UnresolvedComments"),
    (r"item\s*1c[\.\s]+cyber", "Item1C_Cybersecurity"),
    (r"item\s*2[\.\s]+properties", "Item2_Properties"),
    (r"item\s*3[\.\s]+legal", "Item3_LegalProceedings"),
    (r"item\s*5[\.\s]+market", "Item5_MarketInfo"),
    (r"item\s*7[\.\s]+management", "Item7_MDA"),
    (r"item\s*7a[\.\s]+quantitative", "Item7A_MarketRisk"),
    (r"item\s*8[\.\s]+financial\s*statements", "Item8_FinancialStatements"),
    (r"item\s*9a", "Item9A_Controls"),
    (r"item\s*10", "Item10_Directors"),
    (r"item\s*11", "Item11_Compensation"),
    (r"item\s*12", "Item12_SecurityOwnership"),
    (r"item\s*15", "Item15_Exhibits"),
]

STATEMENT_PATTERNS = [
    # ★ v7.4 FIX: Added non-consolidated variants for Microsoft-style 10-Ks.
    # Microsoft uses "BALANCE SHEETS", "INCOME STATEMENTS" etc. WITHOUT "consolidated" prefix.
    # Alphabet/Amazon use "CONSOLIDATED BALANCE SHEETS". Both must match.

    # Balance Sheet — consolidated + non-consolidated
    (r"consolidated\s+balance\s+sheet", "Balance_Sheet"),
    (r"consolidated\s+statements?\s+of\s+(?:financial\s+)?position", "Balance_Sheet"),
    (r"(?:^|\n)\s*balance\s+sheets?\s*(?:\n|$)", "Balance_Sheet"),          # bare "BALANCE SHEETS"

    # Income Statement — consolidated + non-consolidated
    (r"consolidated\s+statements?\s+of\s+(?:income|operations|earnings)", "Income_Statement"),
    (r"(?:^|\n)\s*income\s+statements?\s*(?:\n|$)", "Income_Statement"),     # bare "INCOME STATEMENTS"
    (r"(?:^|\n)\s*statements?\s+of\s+(?:income|operations|earnings)\s*(?:\n|$)", "Income_Statement"),

    # Cash Flow — consolidated + non-consolidated
    (r"consolidated\s+statements?\s+of\s+cash\s+flow", "Cash_Flow"),
    (r"(?:^|\n)\s*cash\s+flows?\s+statements?\s*(?:\n|$)", "Cash_Flow"),     # bare "CASH FLOWS STATEMENTS"
    (r"(?:^|\n)\s*statements?\s+of\s+cash\s+flows?\s*(?:\n|$)", "Cash_Flow"),

    # Equity Statement — consolidated + non-consolidated
    (r"consolidated\s+statements?\s+of\s+(?:stockholders|shareholders).*?(?:equity|deficit)", "Equity_Statement"),
    (r"(?:^|\n)\s*statements?\s+of\s+(?:stockholders|shareholders).*?(?:equity|deficit)", "Equity_Statement"),

    # Comprehensive Income — consolidated + non-consolidated
    (r"consolidated\s+statements?\s+of\s+comprehensive\s+(?:income|loss)", "Comprehensive_Income"),
    (r"(?:^|\n)\s*statements?\s+of\s+comprehensive\s+(?:income|loss)", "Comprehensive_Income"),
]

NOTES_PATTERNS = [
    r"notes\s+to\s+(?:consolidated\s+)?financial\s+statements",
    r"notes\s+to\s+(?:the\s+)?(?:condensed\s+)?consolidated",
    # ★ v7.4: "note\s+\d+" removed from here — it was a false positive that
    # blocked statement detection on ANY page with a footnote reference.
    # Footnote references like "Note 1" appear on balance sheet pages too.
]

# ★ v7.4: Looser note pattern used ONLY as a secondary check
NOTES_PAGE_START_PATTERN = r"^notes?\s+to\s+(?:consolidated\s+)?financial"


# ---------- Section 3: Prompts ----------

# ★ v7.2: Completely rewritten PERSONA with strict document-first rule and LaTeX
PERSONA = """You are "FinAnalyst Pro v7.6", an expert financial document analyst AI.

=== YOUR METHOD (Chain-of-Extraction) ===
You follow a strict THREE-STEP process for every question:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — DATA EXTRACTION (MANDATORY — NEVER SKIP):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Carefully scan ALL context chunks for EACH company.
  For every relevant data point, record it in this format:

  • [Company] | [Metric] | [Value] [Unit] | Source: [Page X / Statement type]
  • Mark as ✅ DIRECT if the value appears explicitly as a labeled line item
  • Mark as 🔢 DERIVED if you computed it from other values

  CRITICAL READING RULES:
  ① PAY ATTENTION TO ROW LABELS. Financial statements have subtotals AND totals:
     - "Total current liabilities" is a SUBTOTAL, NOT the grand total
     - "Total liabilities" is the GRAND TOTAL including long-term items
     - "Total current assets" ≠ "Total assets"
     Never report a subtotal when the question asks for a total.
  ② Read ALL rows of each statement, not just the first matching label.
  ③ The [Company: ...] tag at the start of each chunk tells you which company
     the data belongs to. Never mix data across companies.
  ④ If a value appears multiple times (e.g., from different pages), prefer the
     one from a chunk marked ★ STATEMENT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — CALCULATION (only when truly needed):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠️ DOCUMENT-FIRST RULE (ABSOLUTE):
    If a metric is EXPLICITLY STATED as a labeled line item in the document,
    USE THAT VALUE DIRECTLY. Do NOT recalculate it. Do NOT override it.
    Even if you think you could derive it differently — trust the document.

  Only calculate a derived metric when:
    (a) It does NOT appear as any labeled line item in the document, AND
    (b) All required component values are available in the document.

  ⚠️ NEVER SAY "NOT AVAILABLE" IF IT CAN BE CALCULATED (ABSOLUTE):
    Before writing "not available" or "not found" for any metric, you MUST
    check whether it can be derived from values you DID find. Key identities:
      • Total Liabilities = Total Assets − Stockholders' Equity  (ALWAYS works)
      • Gross Profit = Revenue − Cost of Revenue/Sales
      • Operating Income = Revenue − Total Operating Expenses
      • Net Income = Operating Income ± Other Items − Tax
    If Total Assets and Stockholders' Equity are known → Total Liabilities is KNOWN.
    Saying "not available" when it is calculable is a CRITICAL ERROR.

  When you do calculate, show ALL steps using LaTeX math notation:
    - Inline math: $formula$
    - Block math (for multi-step): $$formula$$

  Example calculation block:
  $$\\text{{Gross Margin}} = \\frac{{\\text{{Gross Profit}}}}{{\\text{{Revenue}}}} \\times 100
  = \\frac{{\\$74{,}039M}}{{\\$245{,}122M}} \\times 100 = 30.2\\%$$

  Common formulas (use ONLY when needed):
    • Gross Profit = Revenue − Cost of Revenue
    • Operating Income = Revenue − Total Operating Expenses
    • Net Income = Revenue − Expenses ± Other ± Tax
    • Total Liabilities = Total Assets − Total Stockholders' Equity
    • Current Ratio = Current Assets / Current Liabilities
    • Debt-to-Equity = Total Liabilities / Stockholders' Equity
    • Gross Margin = Gross Profit / Revenue × 100
    • Operating Margin = Operating Income / Revenue × 100
    • Net Margin = Net Income / Revenue × 100
    • Free Cash Flow = Operating Cash Flow − Capital Expenditures
    • Working Capital = Current Assets − Current Liabilities
    • ROE = Net Income / Stockholders' Equity
    • ROA = Net Income / Total Assets
    • EBITDA = Operating Income + Depreciation + Amortization
    • Revenue Growth = (Current − Prior) / Prior × 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — FINAL ANSWER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Present a clear, well-structured answer:
  • Use markdown tables for multi-company comparisons
  • Cite every number: [Source, Page X] or [Balance Sheet, Page X]
  • Indicate whether each value is ✅ Direct from document or 🔢 Calculated
  • State all monetary units clearly ($ millions or $ billions)

=== ABSOLUTE RULES ===
1. ALWAYS complete Steps 1 → 2 → 3 in order. Never skip Step 1.
2. DOCUMENT-FIRST: A value explicitly labeled in the document is GROUND TRUTH.
   Never recalculate something the document already states.
3. SUBTOTAL ≠ TOTAL: Always read the FULL statement to find the grand total row,
   not just the first subtotal you encounter.
4. NEVER confuse data between companies. Check [Company: ...] tags carefully.
5. NEVER fabricate numbers. Every number must trace to the context.
6. For multi-company questions, cover EVERY company without exception.
7. If data is genuinely unavailable and cannot be calculated, say so explicitly.
8. Prefer ★ STATEMENT chunks over narrative text for financial figures.
9. Use $ for currency. Always state units (millions / billions).
10. If a question has a false premise, correct it first."""

GENERATION_PROMPT = PromptTemplate(
    template="""{persona}

=== RETRIEVED CONTEXT (organized by company) ===
{context}
=== END CONTEXT ===

Previous conversation:
{history}

Question: {question}

Now follow your three-step method strictly.
**Remember:** If a value is labeled in the document, USE IT — do not recalculate.
**Remember:** "Total current liabilities" ≠ "Total liabilities". Read all rows carefully.
Start with Step 1 — Data Extraction.""",
    input_variables=["context", "question", "history"],
    partial_variables={"persona": PERSONA},
)

# ★ v7.2: Verifier now specifically checks for "ignored direct values" and subtotal confusion
VERIFICATION_PROMPT = """You are a rigorous financial fact-checker specializing in 10-K filings.

TASK: Verify every factual claim in the ANSWER against the SOURCE CONTEXT.

=== SOURCE CONTEXT ===
{context}
=== END SOURCE ===

QUESTION: {question}

ANSWER TO VERIFY:
{answer}

=== VERIFICATION RULES ===

A claim is SUPPORTED if:
  1. The exact value appears as an explicitly labeled line item in the context (DIRECT), OR
  2. The value is correctly calculated from context values using a valid formula (DERIVED).
     Verify arithmetic yourself step by step.

A claim is UNSUPPORTED if:
  - Value cannot be found AND cannot be correctly calculated.
  - Arithmetic is wrong.
  - Value is attributed to the WRONG company.

SPECIAL CHECKS — look for these common errors:

① SUBTOTAL vs TOTAL CONFUSION:
   Check whether the answer used a subtotal (e.g., "Total current liabilities") 
   when the document also contains a grand total (e.g., "Total liabilities").
   If the answer reported a subtotal as if it were the grand total, flag it.

② IGNORED DIRECT VALUE:
   If the document explicitly states a value as a labeled line item (e.g., 
   "Total liabilities $243,686") and the answer instead calculated a different
   number for the same metric, flag it as IGNORED_DIRECT_VALUE error.

③ UNJUSTIFIED "NOT AVAILABLE":
   If the answer says a metric is "not available" or "not found", but either:
   (a) The value appears somewhere in the provided context, OR
   (b) The value can be derived from other values in the context
       (e.g., Total Liabilities = Total Assets − Stockholders' Equity),
   then flag it as UNJUSTIFIED_NOT_AVAILABLE. This is a critical failure.

④ CROSS-COMPANY CONFUSION:
   For each claimed number, verify it comes from the correct company's section.

Allow rounding differences within 1%.
Do NOT flag correctly calculated derived metrics as unsupported if no direct
value existed in the document.

Respond in EXACTLY this format:
FAITHFULNESS_SCORE: [0-100]
ISSUES: [list each specific issue, or "None found"]
SUBTOTAL_ERRORS: [list any subtotal-vs-total confusion, or "None found"]
IGNORED_DIRECT_VALUES: [list cases where a doc value was ignored and recalculated differently, or "None found"]
UNJUSTIFIED_NOT_AVAILABLE: [list any metric claimed unavailable that can be found or calculated, or "None found"]
CROSS_COMPANY_ERRORS: [list any company attribution errors, or "None found"]
VERDICT: [FULLY_VERIFIED / MINOR_ISSUES / NEEDS_CORRECTION / UNRELIABLE]
SPECIFIC_CORRECTIONS: [list ONLY factual errors to fix with correct values, or "None"]"""

HYDE_PROMPT = """Imagine you are reading an annual 10-K financial filing. Write a \
short passage (2-3 sentences) that would contain the answer to this question. \
Use realistic financial language and specific number placeholders.

Question: {question}

Hypothetical passage:"""

RERANK_PROMPT = """You are a relevance judge for a financial document retrieval system.

QUERY: {query}

Below are {n} document chunks. For each, output a relevance score from 0 to 10:
  10 = directly answers the query with specific financial data
  7-9 = contains highly relevant data that helps answer the query
  4-6 = somewhat relevant, provides useful context
  1-3 = marginally relevant
  0 = completely irrelevant

{chunks_text}

Return ONLY a JSON array of scores in order, e.g. [8, 3, 10, 5, ...].
No explanation."""


# ---------- Section 4: Utility Functions ----------


def compute_files_hash(uploaded_files):
    h = hashlib.md5()
    for f in sorted(uploaded_files, key=lambda x: x.name):
        h.update(f.name.encode())
        h.update(str(f.size).encode())
    return h.hexdigest()[:12]


def safe_render(text: str) -> str:
    """Render markdown safely. Preserve LaTeX delimiters, only escape stray $."""
    if not text:
        return text
    # Replace dollar signs that are NOT part of LaTeX math ($ or $$)
    # We protect LaTeX by temporarily replacing $$ and $ math blocks
    # Simple approach: only escape $ that are followed by digits (currency)
    # and not preceded by another $ (LaTeX)
    result = re.sub(r'(?<!\$)\$(?!\$)(?=\s*[\d,\(])', r'&#36;', text)
    return result


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def process_llm_output(text: str, model_name: str = "") -> Tuple[str, Optional[str]]:
    if not text:
        return text, None
    reasoning = None
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text, reasoning


def detect_company_name(filename: str, page_texts: List[str]) -> str:
    fn = filename.lower()
    if any(k in fn for k in ["amazon", "amzn"]):
        return "Amazon"
    if any(k in fn for k in ["alphabet", "google", "googl", "goog"]):
        return "Alphabet_Google"
    if any(k in fn for k in ["microsoft", "msft"]):
        return "Microsoft"

    combined = " ".join((t or "") for t in page_texts[:10]).lower()[:8000]
    amazon_sigs = ["amazon.com, inc", "amazon.com inc", "amazon web services", "amzn"]
    google_sigs = ["alphabet inc", "google llc", "google cloud platform", "googl", "youtube"]
    msft_sigs = ["microsoft corporation", "microsoft corp", "azure", "msft"]
    scores = {
        "Amazon": sum(1 for s in amazon_sigs if s in combined),
        "Alphabet_Google": sum(1 for s in google_sigs if s in combined),
        "Microsoft": sum(1 for s in msft_sigs if s in combined),
    }
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "Unknown"


def detect_10k_section(text: str) -> str:
    text_lower = text[:600].lower()
    for pattern, section_name in SECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return section_name
    return "Other"


def detect_statement_type(text: str) -> Optional[str]:
    # ★ v7.4: Extended search window 2000→4000 chars, use re.IGNORECASE throughout
    # so "BALANCE SHEETS" (ALL CAPS in Microsoft 10-K) matches correctly.
    search_text = text[:4000]

    # ★ v7.4 FIX: Notes check uses ONLY the first 200 chars of the page AND only
    # blocks on explicit "Notes to Financial Statements" header — NOT on "Note 1"
    # footnote references that appear on every balance sheet page.
    page_start = text[:200].lower()
    for npat in NOTES_PATTERNS:
        if re.search(npat, page_start):
            return None
    # Also block if "Notes to Financial Statements" is anywhere in first 400 chars
    if re.search(NOTES_PAGE_START_PATTERN, text[:400], re.IGNORECASE):
        return None

    for pattern, stmt_type in STATEMENT_PATTERNS:
        if re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE):
            return stmt_type
    return None


def detect_fiscal_year(text: str) -> Optional[str]:
    text_lower = text[:2000].lower()
    m = re.search(
        r"(?:year\s+ended|as\s+of|for\s+the\s+(?:fiscal\s+)?years?\s+ended)"
        r"[^0-9]{0,30}(\d{4})",
        text_lower,
    )
    if m:
        return m.group(1)
    m = re.search(
        r"(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2},?\s+(\d{4})",
        text_lower,
    )
    if m:
        return m.group(1)
    years = re.findall(r"\b(20[12]\d)\b", text[:2000])
    if years:
        return max(years)
    return None


def _is_financial_continuation(text: str) -> bool:
    text_lower = text.lower()
    for npat in NOTES_PATTERNS:
        if re.search(npat, text_lower[:500]):
            return False
    terms = [
        "total", "net", "operating", "income", "loss", "revenue",
        "assets", "liabilities", "equity", "depreciation",
        "amortization", "shares", "earnings", "comprehensive",
        "retained", "accumulated", "provision", "deferred",
    ]
    term_hits = sum(1 for t in terms if t in text_lower)
    number_hits = len(re.findall(r"\b[\d,]{2,}\b", text))
    return term_hits >= 4 and number_hits >= 10


def validate_table(table: list) -> bool:
    if not table or len(table) < 2:
        return False
    non_empty = 0
    total = 0
    for row in table:
        for cell in row:
            total += 1
            if cell and str(cell).strip():
                non_empty += 1
    if total == 0:
        return False
    fill_ratio = non_empty / total
    if fill_ratio < 0.2:
        return False
    all_text = " ".join(str(c) for row in table for c in row if c)
    numbers = re.findall(r"\b[\d,]+(?:\.\d+)?\b", all_text)
    if len(numbers) < 2:
        return False
    return True


def clean_table_cell(cell: str) -> str:
    if not cell:
        return ""
    c = str(cell).strip()
    c = re.sub(r"\s+", " ", c)
    c = c.replace("|", "\\|")
    c = re.sub(r"^[\(\)\$]+$", "", c)
    c = re.sub(r"^\((.+)\)$", r"-\1", c)
    return c


def format_table_as_markdown(table: list, page_num: int = 0) -> str:
    if not validate_table(table):
        return ""
    cleaned = []
    for row in table:
        cleaned_row = [clean_table_cell(cell) for cell in row]
        if any(c for c in cleaned_row):
            cleaned.append(cleaned_row)
    if not cleaned:
        return ""
    max_cols = max(len(r) for r in cleaned)
    for row in cleaned:
        while len(row) < max_cols:
            row.append("")
    lines = []
    lines.append("| " + " | ".join(cleaned[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def chunk_quality_score(doc: Document) -> float:
    text = doc.page_content
    if len(text.strip()) < MIN_CHUNK_LENGTH:
        return 0.0
    score = 0.5
    numbers = re.findall(r"\b[\d,]+(?:\.\d+)?\b", text)
    if len(numbers) >= 3:
        score += 0.2
    if len(numbers) >= 10:
        score += 0.1
    fin_terms = [
        "revenue", "income", "assets", "liabilities", "equity", "cash",
        "operating", "total", "net", "shares", "earnings", "profit",
        "loss", "expense",
    ]
    hits = sum(1 for t in fin_terms if t in text.lower())
    score += min(hits * 0.03, 0.2)
    if doc.metadata.get("is_statement_chunk"):
        score += 0.3
    if doc.metadata.get("is_table_chunk") or doc.metadata.get("has_tables"):
        score += 0.1
    if len(text) < 200:
        score -= 0.2
    return max(0.0, min(1.0, score))


def format_chat_history(messages: list, max_turns: int = 4) -> str:
    if len(messages) <= 1:
        return "No previous conversation."
    out = ""
    for msg in messages[:-1][-max_turns:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:400]
        if len(msg["content"]) > 400:
            content += "..."
        out += f"{role}: {content}\n\n"
    return out


def extract_evidence_sources(docs: list) -> list:
    sources, seen = [], set()
    for doc in docs:
        company = doc.metadata.get("company", "Unknown")
        section = doc.metadata.get("section", "Other")
        page = doc.metadata.get("page", "?")
        stmt = doc.metadata.get("statement_type", "")
        key = f"{company}-{section}-{page}-{stmt}"
        if key not in seen:
            seen.add(key)
            label = section.replace("_", " ")
            if stmt:
                label = stmt.replace("_", " ")
            sources.append(
                dict(
                    company=company, section=label, page=page,
                    emoji=EMOJI_MAP.get(company, "\U000026AA"),
                    statement_type=stmt,
                )
            )
    return sorted(sources, key=lambda x: (x["company"], str(x["page"])))


def build_statement_inventory(chunks: list) -> Dict[str, List[str]]:
    inventory: Dict[str, List[str]] = {}
    for chunk in chunks:
        stmt = chunk.metadata.get("statement_type")
        if stmt:
            comp = chunk.metadata.get("company", "Unknown")
            year = chunk.metadata.get("year", "?")
            entry = f"{stmt.replace('_', ' ')} ({year})"
            inventory.setdefault(comp, [])
            if entry not in inventory[comp]:
                inventory[comp].append(entry)
    return inventory


def _smart_truncate_chunk(content: str, max_chars: int) -> str:
    """Truncate a chunk intelligently — keep table data, trim raw text."""
    if len(content) <= max_chars:
        return content

    table_start = content.find("=== STRUCTURED TABLE DATA ===")
    table_end = content.find("=== END TABLE DATA ===")

    if table_start >= 0 and table_end >= 0:
        table_section = content[table_start: table_end + len("=== END TABLE DATA ===")]
        prefix_section = content[:table_start]

        if len(table_section) <= max_chars - 500:
            remaining = max_chars - len(table_section) - 50
            if remaining > 200:
                return prefix_section[:remaining] + "\n[...text truncated]\n" + table_section
            else:
                return content[:200] + "\n[...truncated]\n" + table_section[: max_chars - 250]
        else:
            return content[:max_chars - 30] + "\n[...truncated]"
    else:
        return content[:max_chars - 30] + "\n[...truncated]"


def build_company_grouped_context(
    docs: List[Document],
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> str:
    """★ v7.3: Organize chunks by company with WEIGHTED budget allocation.

    Budget is allocated proportionally by statement chunk count per company,
    not naive equal split — fixes Alphabet/Amazon/Microsoft imbalance where
    Alphabet had 12 detected statements vs Microsoft's 2.
    """
    by_company: Dict[str, List[Document]] = {}
    for doc in docs:
        comp = doc.metadata.get("company", "Unknown")
        by_company.setdefault(comp, []).append(doc)

    n_companies = max(len(by_company), 1)
    header_overhead = 200 * n_companies
    usable_budget = budget_chars - header_overhead

    # ★ v7.3: Weight budget by number of statement chunks per company
    # (more statement chunks = company has more financial data = needs more space)
    # But enforce a FLOOR so each company gets at least 1/n of the budget.
    stmt_counts = {}
    for comp, cdocs in by_company.items():
        stmt_counts[comp] = max(1, sum(1 for d in cdocs if d.metadata.get("is_statement_chunk")))

    total_stmts = sum(stmt_counts.values())
    min_share = 1.0 / n_companies  # floor: each company gets at least equal share

    weights = {}
    for comp in by_company:
        raw_weight = stmt_counts[comp] / total_stmts
        # blend 50% equal + 50% weighted to avoid extreme imbalance
        weights[comp] = 0.5 * min_share + 0.5 * raw_weight

    # Normalize weights so they sum to 1
    total_weight = sum(weights.values())
    company_budgets = {
        comp: int(usable_budget * w / total_weight)
        for comp, w in weights.items()
    }

    parts = []

    for comp in sorted(by_company.keys()):
        comp_docs = by_company[comp]
        per_company_budget = company_budgets[comp]

        comp_docs.sort(
            key=lambda d: (
                0 if d.metadata.get("is_statement_chunk") else
                1 if d.metadata.get("is_table_chunk") else 2,
                d.metadata.get("page", 999),
            )
        )

        header = (
            f"\n{'═' * 60}\n"
            f"  {EMOJI_MAP.get(comp, '⚪')} {comp.replace('_', ' / ').upper()}\n"
            f"{'═' * 60}\n"
        )
        parts.append(header)
        company_chars_used = 0

        for doc in comp_docs:
            content = doc.page_content
            stmt = doc.metadata.get("statement_type")
            is_stmt = doc.metadata.get("is_statement_chunk")
            page = doc.metadata.get("page", "?")

            tag = f"[Page {page}"
            if is_stmt and stmt:
                tag += f" | ★ STATEMENT: {stmt.replace('_', ' ')}"
            elif doc.metadata.get("is_table_chunk"):
                tag += " | TABLE"
            tag += "]\n"

            chunk_text = tag + content + "\n\n---\n"

            if company_chars_used + len(chunk_text) > per_company_budget:
                remaining = per_company_budget - company_chars_used
                if remaining > 800:
                    truncated_content = _smart_truncate_chunk(
                        content, remaining - len(tag) - 50
                    )
                    chunk_text = tag + truncated_content + "\n\n---\n"
                    parts.append(chunk_text)
                    company_chars_used += len(chunk_text)
                break

            parts.append(chunk_text)
            company_chars_used += len(chunk_text)

    return "".join(parts)


def score_chunk_relevance(
    chunk: Document,
    query: str,
    target_companies: List[str],
    target_statements: List[str],
    query_type: str = "general",
) -> float:
    score = 0.0
    q_lower = query.lower()
    text_lower = chunk.page_content[:1000].lower()
    meta = chunk.metadata

    if meta.get("company") in target_companies:
        score += 10.0
    else:
        score -= 5.0

    norm_map = {
        "Balance_Sheet": "balance_sheet",
        "Income_Statement": "income_statement",
        "Cash_Flow": "cash_flow",
        "Equity_Statement": "equity_statement",
        "Comprehensive_Income": "comprehensive_income",
    }
    chunk_stmt = norm_map.get(meta.get("statement_type", ""), "")
    if chunk_stmt and chunk_stmt in target_statements:
        score += 15.0

    if meta.get("is_statement_chunk"):
        score += 5.0

    if meta.get("is_table_chunk") and query_type == "financial":
        score += 3.0

    section = meta.get("section", "Other")
    if query_type == "financial" and section in ("Item7_MDA", "Item8_FinancialStatements"):
        score += 4.0
    elif query_type == "risk" and section in ("Item1A_RiskFactors", "Item7A_MarketRisk"):
        score += 4.0
    elif query_type == "legal" and section == "Item3_LegalProceedings":
        score += 4.0
    elif query_type == "business" and section == "Item1_Business":
        score += 4.0

    q_words = set(re.findall(r"\b\w{3,}\b", q_lower))
    c_words = set(re.findall(r"\b\w{3,}\b", text_lower))
    if q_words:
        overlap = len(q_words & c_words) / len(q_words)
        score += overlap * 8.0

    score += chunk_quality_score(chunk) * 3.0
    return score


# ---------- Section 5: PDF Loading ----------


def load_pdf_enhanced(
    file_path: str, company_name: str, use_tables: bool = True
) -> List[Document]:
    if not (use_tables and HAS_PDFPLUMBER):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["company"] = company_name
            doc.metadata["section"] = detect_10k_section(doc.page_content)
        return docs

    pages_data = []
    current_section = "Other"
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []

                sec = detect_10k_section(text)
                if sec != "Other":
                    current_section = sec

                stmt_type = detect_statement_type(text)
                year = detect_fiscal_year(text)

                pages_data.append(
                    dict(
                        page=page_num, text=text, tables=tables,
                        section=current_section, statement_type=stmt_type,
                        year=year, has_tables=bool(tables),
                    )
                )
    except Exception:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["company"] = company_name
            doc.metadata["section"] = detect_10k_section(doc.page_content)
        return docs

    if not pages_data:
        return []

    known_year = None
    for pd_item in pages_data:
        if pd_item["year"]:
            known_year = pd_item["year"]
        elif known_year:
            pd_item["year"] = known_year

    documents = []
    consumed = set()

    i = 0
    while i < len(pages_data):
        pd_item = pages_data[i]
        if pd_item["statement_type"]:
            group_pages = [i]
            group_section = pd_item["section"]
            group_stmt = pd_item["statement_type"]
            j = i + 1
            while j < len(pages_data) and len(group_pages) < MAX_STATEMENT_CONTINUATION_PAGES:
                nxt = pages_data[j]
                is_cont = False
                if nxt["statement_type"] == group_stmt:
                    is_cont = True
                elif nxt["statement_type"] is not None and nxt["statement_type"] != group_stmt:
                    break
                elif (
                    nxt["statement_type"] is None
                    and nxt["section"] == group_section
                    and (nxt["has_tables"] or _is_financial_continuation(nxt["text"]))
                ):
                    is_cont = True
                if is_cont:
                    group_pages.append(j)
                    j += 1
                else:
                    break

            for idx in group_pages:
                consumed.add(idx)

            merged_text_parts = []
            merged_tables_md = []
            start_page = pages_data[group_pages[0]]["page"]
            end_page = pages_data[group_pages[-1]]["page"]
            year = pd_item["year"]

            for idx in group_pages:
                p = pages_data[idx]
                merged_text_parts.append(p["text"])
                if p["year"] and not year:
                    year = p["year"]
                for tbl in p["tables"]:
                    md = format_table_as_markdown(tbl, p["page"])
                    if md:
                        merged_tables_md.append(md)

            # ★ v7.3: Store statement chunks PER PAGE (not one mega-merged blob).
            # This ensures every row of every page is independently indexed and
            # retrievable. "Total liabilities" at the bottom of page N is never
            # truncated away by a size cap on the merged multi-page chunk.
            # We also store a lightweight "group header" chunk so that the LLM
            # knows all these pages belong to the same statement.
            group_header = (
                f"[Company: {company_name} | "
                f"Statement GROUP: {group_stmt.replace('_', ' ')} | "
                f"Year: {year or '?'} | "
                f"Pages: {start_page}-{end_page} | "
                f"Total pages in statement: {len(group_pages)}]\n"
                f"[IMPORTANT: Read ALL rows. 'Total current liabilities' is a SUBTOTAL. "
                f"'Total liabilities' is the GRAND TOTAL and appears further down.]\n"
            )

            # One chunk per page — fully indexed, no truncation
            for idx in group_pages:
                p = pages_data[idx]
                page_prefix = (
                    f"[Company: {company_name} | "
                    f"Statement: {group_stmt.replace('_', ' ')} | "
                    f"Year: {year or '?'} | "
                    f"Page: {p['page']} of group {start_page}-{end_page}]\n"
                    f"[NOTE: Read ALL rows. Subtotals like 'Total current liabilities' "
                    f"≠ grand total 'Total liabilities'. Find the grand total row.]\n\n"
                )
                page_body = p["text"]
                page_tables_md = []
                for tbl in p["tables"]:
                    md = format_table_as_markdown(tbl, p["page"])
                    if md:
                        page_tables_md.append(md)

                page_table_section = ""
                if page_tables_md:
                    page_table_section = (
                        "\n\n=== STRUCTURED TABLE DATA ===\n\n"
                        + "\n\n".join(page_tables_md)
                        + "\n\n=== END TABLE DATA ===\n"
                    )

                page_content = page_prefix + page_body + page_table_section

                # ★ v7.3: NO size cap here — store full page content
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=dict(
                            source=file_path, page=p["page"],
                            end_page=p["page"], company=company_name,
                            section=group_section, statement_type=group_stmt,
                            year=year, is_statement_chunk=True,
                            has_tables=bool(page_tables_md),
                            stmt_group_start=start_page,
                            stmt_group_end=end_page,
                        ),
                    )
                )
            i = j
        else:
            i += 1

    for idx, pd_item in enumerate(pages_data):
        if idx in consumed:
            continue
        text = pd_item["text"]
        tables = pd_item["tables"]

        table_text = ""
        if tables:
            for tbl in tables:
                md = format_table_as_markdown(tbl, pd_item["page"])
                if md:
                    table_text += (
                        f"\n\n[TABLE on Page {pd_item['page']}]\n{md}\n"
                        f"[END TABLE]\n"
                    )

        full_text = text
        if table_text:
            full_text += table_text

        if full_text.strip() and len(full_text.strip()) >= MIN_CHUNK_LENGTH:
            documents.append(
                Document(
                    page_content=full_text,
                    metadata=dict(
                        source=file_path, page=pd_item["page"],
                        company=company_name, section=pd_item["section"],
                        year=pd_item["year"], has_tables=bool(tables),
                        statement_type=None, is_statement_chunk=False,
                    ),
                )
            )

    return documents


# ---------- Section 6: Chunking ----------


def table_aware_chunking(
    documents: list, chunk_size: int = 1200, chunk_overlap: int = 200,
) -> List[Document]:
    statement_chunks = []
    regular_docs = []

    for doc in documents:
        if doc.metadata.get("is_statement_chunk"):
            if not doc.page_content.startswith("[Company:"):
                prefix = (
                    f"[Company: {doc.metadata.get('company', 'Unknown')} | "
                    f"Statement: {doc.metadata.get('statement_type', '?')} | "
                    f"Year: {doc.metadata.get('year', '?')} | "
                    f"Page: {doc.metadata.get('page', '?')}]\n"
                )
                doc.page_content = prefix + doc.page_content
            statement_chunks.append(doc)
        else:
            regular_docs.append(doc)

    table_chunks = []
    text_documents = []
    table_pattern = r"\[TABLE[^\]]*\].*?\[END TABLE\]"

    for doc in regular_docs:
        content = doc.page_content
        tables_found = re.findall(table_pattern, content, re.DOTALL)

        if tables_found:
            non_table = re.sub(table_pattern, "\n", content, flags=re.DOTALL)
            for table in tables_found:
                prefix = (
                    f"[Company: {doc.metadata.get('company', 'Unknown')} | "
                    f"Section: {doc.metadata.get('section', 'Other')} | "
                    f"Page: {doc.metadata.get('page', '?')}]\n"
                )
                table_chunks.append(
                    Document(
                        page_content=prefix + table,
                        metadata={**doc.metadata, "is_table_chunk": True},
                    )
                )
            if non_table.strip():
                text_documents.append(
                    Document(page_content=non_table, metadata=doc.metadata.copy())
                )
        else:
            text_documents.append(doc)

    for doc in text_documents:
        if not doc.page_content.startswith("[Company:"):
            prefix = (
                f"[Company: {doc.metadata.get('company', 'Unknown')} | "
                f"Section: {doc.metadata.get('section', 'Other')} | "
                f"Page: {doc.metadata.get('page', '?')}]\n"
            )
            doc.page_content = prefix + doc.page_content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "],
    )
    text_chunks = splitter.split_documents(text_documents)

    all_chunks = text_chunks + table_chunks + statement_chunks

    filtered = []
    for chunk in all_chunks:
        qs = chunk_quality_score(chunk)
        if qs >= 0.25:
            chunk.metadata["quality_score"] = round(qs, 3)
            filtered.append(chunk)

    return filtered


# ---------- Section 7: Retrieval Engine ----------


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, k=6):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_idx if scores[i] > 0]


def _dedup_docs(docs_list: list) -> list:
    seen, out = set(), []
    for doc in docs_list:
        h = hash(doc.page_content[:300])
        if h not in seen:
            seen.add(h)
            out.append(doc)
    return out


def retrieve_per_company(
    vector_store, bm25_retriever, query: str, companies: list,
    k_per: int = 8, use_mmr: bool = True,
) -> List[Document]:
    all_docs = []
    seen = set()
    for company in companies:
        company_query = f"{company.replace('_', ' ')} {query}"
        try:
            if use_mmr:
                candidates = vector_store.max_marginal_relevance_search(
                    company_query, k=k_per * 4, fetch_k=k_per * 10, lambda_mult=0.6,
                )
            else:
                candidates = vector_store.similarity_search(company_query, k=k_per * 4)
        except Exception:
            candidates = vector_store.similarity_search(company_query, k=k_per * 4)

        company_docs = [d for d in candidates if d.metadata.get("company") == company]

        if bm25_retriever and len(company_docs) < k_per:
            bm25_hits = bm25_retriever.search(company_query, k=k_per * 3)
            for doc in bm25_hits:
                if doc.metadata.get("company") == company:
                    h = hash(doc.page_content[:300])
                    if h not in seen:
                        company_docs.append(doc)

        for doc in company_docs[:k_per]:
            h = hash(doc.page_content[:300])
            if h not in seen:
                seen.add(h)
                all_docs.append(doc)
    return all_docs


def retrieve_multi_query(
    vector_store, bm25_retriever, queries: list,
    k: int = 12, use_mmr: bool = True,
) -> List[Document]:
    all_docs = []
    for query in queries:
        try:
            if use_mmr:
                docs = vector_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=k * 4, lambda_mult=0.7,
                )
            else:
                docs = vector_store.similarity_search(query, k=k)
        except Exception:
            docs = vector_store.similarity_search(query, k=k)
        all_docs.extend(docs)
        if bm25_retriever:
            try:
                all_docs.extend(bm25_retriever.search(query, k=k))
            except Exception:
                pass
    return _dedup_docs(all_docs)


def retrieve_by_statement(
    all_chunks: list, companies: list, statement_types: list,
) -> List[Document]:
    results = []
    norm_map = {
        "balance_sheet": "Balance_Sheet",
        "income_statement": "Income_Statement",
        "cash_flow": "Cash_Flow",
        "equity_statement": "Equity_Statement",
        "comprehensive_income": "Comprehensive_Income",
    }
    target_types = set()
    for st in statement_types:
        mapped = norm_map.get(st, st)
        target_types.add(mapped)
    for chunk in all_chunks:
        if (
            chunk.metadata.get("is_statement_chunk")
            and chunk.metadata.get("company") in companies
            and chunk.metadata.get("statement_type") in target_types
        ):
            # ★ v7.6: Filter out low-quality statement pages.
            # A real financial statement page MUST have numeric data.
            # Pages with <5 numbers are likely TOC, cover, or notes pages.
            text = chunk.page_content
            numbers = re.findall(r"\b[\d,]{3,}\b", text)
            if len(numbers) < 5:
                continue  # Skip narrative/TOC pages tagged as statements by mistake
            results.append(chunk)
    return results


def rerank_with_llm(
    llm, query: str, docs: List[Document],
    top_n: int = 20, batch_size: int = 10,
) -> List[Document]:
    if len(docs) <= top_n:
        return docs
    scored = []
    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start: batch_start + batch_size]
        chunks_text = ""
        for ci, doc in enumerate(batch):
            preview = doc.page_content[:300].replace("\n", " ")
            comp = doc.metadata.get("company", "?")
            stmt = doc.metadata.get("statement_type", "")
            tag = f"[{comp}"
            if stmt:
                tag += f" | {stmt}"
            tag += "]"
            chunks_text += f"\nChunk {ci + 1} {tag}: {preview}\n"
        prompt_text = RERANK_PROMPT.format(
            query=query, n=len(batch), chunks_text=chunks_text,
        )
        try:
            resp = llm.invoke(prompt_text)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            raw, _ = process_llm_output(raw)
            raw = raw.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            scores_arr = json.loads(raw)
            if isinstance(scores_arr, list):
                for ci, doc in enumerate(batch):
                    s = scores_arr[ci] if ci < len(scores_arr) else 5
                    scored.append((doc, float(s)))
            else:
                for doc in batch:
                    scored.append((doc, 5.0))
        except Exception:
            for doc in batch:
                scored.append((doc, 5.0))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_n]]


# ---------- Section 8: Agentic RAG Controller ----------


class AgenticRAGController:
    def __init__(
        self, vector_store, bm25_retriever, llm,
        all_chunks=None, use_mmr=True, use_hybrid=True,
        use_section_boost=True, use_reranking=False,
        context_budget=DEFAULT_CONTEXT_BUDGET_CHARS,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever if use_hybrid else None
        self.llm = llm
        self.all_chunks = all_chunks or []
        self.use_mmr = use_mmr
        self.use_section_boost = use_section_boost
        self.use_reranking = use_reranking
        self.context_budget = context_budget
        self.log = []

    def _log(self, msg: str):
        self.log.append(msg)

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict]:
        text = text.strip()
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            blob = m.group()
            blob = re.sub(r",\s*}", "}", blob)
            blob = re.sub(r",\s*]", "]", blob)
            try:
                return json.loads(blob)
            except Exception:
                pass
        return None

    def analyze_query(self, query: str, known_entities: List[str]) -> Dict:
        entity_str = ", ".join(known_entities) if known_entities else "Unknown"
        prompt = (
            "You are a query analysis agent for a financial 10-K document "
            "retrieval system.\n\n"
            f"The document collection contains 10-K annual filings from: "
            f"{entity_str}\n\n"
            "These filings include:\n"
            "- Business Overview (Item 1)\n"
            "- Risk Factors (Item 1A)\n"
            "- MD&A — Management Discussion & Analysis (Item 7)\n"
            "- Financial Statements (Item 8): Balance Sheets, Income "
            "Statements, Cash Flow Statements, Equity Statements\n\n"
            "Analyze the user's query and return a **valid JSON object** "
            "with exactly these fields:\n\n"
            '"target_entities" — array of entity names from the list above '
            "that this query is about. Use EXACT names from the list. "
            "If ambiguous or if it says 'all' / 'each' / 'every' / 'compare', include ALL.\n\n"
            '"is_comparison" — boolean, true if comparing entities.\n\n'
            '"search_queries" — array of 5-10 diverse search queries '
            "optimised for retrieval. Include:\n"
            "  • the original query\n"
            "  • synonym/alternative terminology\n"
            "  • For financial data: specific statement name + year\n"
            "  • For derived metrics: queries for EACH component\n"
            "  • Per-entity queries for multi-entity questions\n\n"
            '"query_type" — one of: "financial", "risk", "legal", '
            '"business", "technical", "general"\n\n'
            '"relevant_statements" — array from: '
            '"balance_sheet", "income_statement", "cash_flow", '
            '"equity_statement", "comprehensive_income". '
            "Empty array if not financial.\n\n"
            f"Query: {query}\n\n"
            "Return ONLY valid JSON."
        )
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            raw, _ = process_llm_output(raw)
            result = self._parse_json(raw)
            if result is not None:
                raw_ents = result.get("target_entities", [])
                if not isinstance(raw_ents, list):
                    raw_ents = []
                valid_ents = [e for e in raw_ents if e in known_entities]
                if not valid_ents:
                    valid_ents = known_entities.copy()
                result["target_entities"] = valid_ents
                result.setdefault("is_comparison", len(valid_ents) > 1)
                sq = result.get("search_queries", [])
                if not isinstance(sq, list) or not sq:
                    sq = [query]
                if query not in sq:
                    sq.insert(0, query)
                result["search_queries"] = sq[:12]
                result.setdefault("query_type", "general")
                rs = result.get("relevant_statements", [])
                if not isinstance(rs, list):
                    rs = []
                result["relevant_statements"] = rs
                self._log(
                    f"✅ LLM analysis → {len(sq)} queries, "
                    f"entities: {valid_ents}, type: {result['query_type']}, "
                    f"statements: {rs}"
                )
                return result
            else:
                self._log("⚠️ LLM returned unparseable JSON, using fallback")
        except Exception as exc:
            self._log(f"⚠️ LLM analysis error ({exc}), using fallback")
        return self._fallback_analyze(query, known_entities)

    def _fallback_analyze(self, query: str, known_entities: List[str]) -> Dict:
        q_lower = query.lower()
        target = []
        for entity in known_entities:
            words = {w for w in re.split(r"[_\s/]+", entity.lower()) if len(w) > 2}
            if any(w in q_lower for w in words):
                target.append(entity)
        if not target:
            target = known_entities.copy()

        is_comp = len(target) > 1 or any(
            w in q_lower for w in [
                "compare", "comparison", "versus", "vs", "difference",
                "each", "all", "every", "both",
            ]
        )
        rel_stmts = []
        if any(w in q_lower for w in ["asset", "liabilit", "equity", "balance sheet", "debt"]):
            rel_stmts.append("balance_sheet")
        if any(w in q_lower for w in [
            "revenue", "income", "profit", "loss", "earnings", "expense", "cost of", "margin",
        ]):
            rel_stmts.append("income_statement")
        if any(w in q_lower for w in ["cash flow", "cash from", "capital expenditure", "capex", "free cash flow"]):
            rel_stmts.append("cash_flow")

        queries = [query]
        for ent in target:
            queries.append(f"{ent.replace('_', ' ')} {query}")
        for stmt in rel_stmts:
            stmt_name = stmt.replace("_", " ")
            queries.append(f"consolidated {stmt_name} 2024")
            for ent in target:
                queries.append(f"{ent.replace('_', ' ')} {stmt_name} {query}")

        self._log(
            f"⚙️ Fallback analysis → {len(queries)} queries, "
            f"entities: {target}, statements: {rel_stmts}"
        )
        return dict(
            target_entities=target, is_comparison=is_comp,
            search_queries=queries[:12], query_type="general",
            relevant_statements=rel_stmts,
        )

    def generate_hyde_passage(self, query: str) -> Optional[str]:
        try:
            prompt = HYDE_PROMPT.format(question=query)
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text, _ = process_llm_output(text)
            return text
        except Exception:
            return None

    def check_coverage(self, docs: list, needed: list) -> Dict[str, int]:
        cov = {c: 0 for c in needed}
        for d in docs:
            c = d.metadata.get("company", "Unknown")
            if c in cov:
                cov[c] += 1
        return cov

    def check_statement_coverage(
        self, docs: list, companies: list, statement_types: list,
    ) -> Dict:
        cov = {}
        for comp in companies:
            for st in statement_types:
                key = f"{comp}|{st}"
                cov[key] = 0
        for d in docs:
            if d.metadata.get("is_statement_chunk"):
                comp = d.metadata.get("company", "Unknown")
                st_raw = d.metadata.get("statement_type", "")
                norm_map = {
                    "Balance_Sheet": "balance_sheet",
                    "Income_Statement": "income_statement",
                    "Cash_Flow": "cash_flow",
                    "Equity_Statement": "equity_statement",
                    "Comprehensive_Income": "comprehensive_income",
                }
                st = norm_map.get(st_raw, st_raw)
                key = f"{comp}|{st}"
                if key in cov:
                    cov[key] += 1
        return cov

    def _enforce_company_balance(
        self, docs: List[Document], companies: List[str], max_total: int,
    ) -> List[Document]:
        if len(companies) <= 1:
            return docs[:max_total]

        by_company: Dict[str, List[Document]] = {c: [] for c in companies}
        other_docs = []

        for doc in docs:
            comp = doc.metadata.get("company", "Unknown")
            if comp in by_company:
                by_company[comp].append(doc)
            else:
                other_docs.append(doc)

        per_company_max = max(max_total // len(companies), 4)
        balanced = []
        for comp in companies:
            balanced.extend(by_company[comp][:per_company_max])

        remaining = max_total - len(balanced)
        used_hashes = {hash(d.page_content[:300]) for d in balanced}
        all_remaining = []
        for comp in companies:
            for doc in by_company[comp][per_company_max:]:
                if hash(doc.page_content[:300]) not in used_hashes:
                    all_remaining.append(doc)
        all_remaining.extend(other_docs)

        for doc in all_remaining[:remaining]:
            balanced.append(doc)

        return balanced

    def retrieve(
        self, query: str, known_entities: List[str],
        base_k: int = 15, max_retries: int = 2,
    ) -> Tuple[List[Document], Dict]:
        self.log = []

        analysis = self.analyze_query(query, known_entities)

        query_info = dict(
            companies=analysis["target_entities"],
            is_comparison=analysis["is_comparison"],
            type=analysis["query_type"],
            needs_financial=(analysis["query_type"] == "financial"),
            relevant_statements=analysis.get("relevant_statements", []),
        )

        self._log(
            f"Query type: {query_info['type']} | "
            f"Entities: {', '.join(query_info['companies'])} | "
            f"Comparison: {query_info['is_comparison']} | "
            f"Statements: {query_info['relevant_statements']}"
        )

        sub_queries = analysis["search_queries"]
        self._log(
            f"Search queries ({len(sub_queries)}): "
            + " | ".join(q[:80] for q in sub_queries[:6])
        )

        n_ent = max(len(query_info["companies"]), 1)
        if query_info["is_comparison"]:
            k = max(base_k, n_ent * 8 + 4)
        elif query_info["needs_financial"]:
            k = max(base_k, base_k + 10)
        elif query_info["type"] in ("legal", "risk"):
            k = max(base_k, base_k + 8)
        else:
            k = base_k
        k = min(k, 50)
        k_per = max(6, k // n_ent + 3)
        self._log(f"Dynamic K: {k} total, {k_per} per entity")

        # ★ v7.5 FIX: Separate PINNED statement chunks from scored chunks.
        # Statement chunks retrieved by metadata are GUARANTEED to appear in
        # the final output — they bypass score-based filtering entirely.
        # ★ v7.6: PIN ALL statement chunks — no quota limit.
        pinned_docs: List[Document] = []
        pinned_hashes: set = set()

        if query_info["relevant_statements"] and self.all_chunks:
            stmt_docs = retrieve_by_statement(
                self.all_chunks, query_info["companies"],
                query_info["relevant_statements"],
            )
            # Sort: most recent year first, then by page number
            stmt_docs.sort(key=lambda d: (
                d.metadata.get("company", ""),
                d.metadata.get("statement_type", ""),
                -(int(d.metadata.get("year") or 0)),
                d.metadata.get("page", 999),
            ))
            # Pin ALL of them — score filter cannot discard any statement chunk
            for doc in stmt_docs:
                h = hash(doc.page_content[:300])
                if h not in pinned_hashes:
                    pinned_docs.append(doc)
                    pinned_hashes.add(h)

            self._log(
                f"★ Pinned ALL {len(pinned_docs)} statement chunks "
                f"(ALL guaranteed in output — score filter bypassed)"
            )

        # Scored (non-pinned) pool — vector + BM25 retrieval
        scored_pool: List[Document] = []

        if query_info["is_comparison"]:
            per_co = retrieve_per_company(
                self.vector_store, self.bm25, query,
                query_info["companies"], k_per=k_per, use_mmr=self.use_mmr,
            )
            for d in per_co:
                h = hash(d.page_content[:300])
                if h not in pinned_hashes:
                    scored_pool.append(d)
            extra = retrieve_multi_query(
                self.vector_store, self.bm25, sub_queries,
                k=k, use_mmr=self.use_mmr,
            )
            seen_pool = {hash(d.page_content[:300]) for d in scored_pool} | pinned_hashes
            for d in extra:
                h = hash(d.page_content[:300])
                if h not in seen_pool:
                    scored_pool.append(d)
                    seen_pool.add(h)
        else:
            vec_docs = retrieve_multi_query(
                self.vector_store, self.bm25, sub_queries,
                k=k * 2, use_mmr=self.use_mmr,
            )
            seen_pool = pinned_hashes.copy()
            for d in vec_docs:
                h = hash(d.page_content[:300])
                if h not in seen_pool:
                    scored_pool.append(d)
                    seen_pool.add(h)

        all_candidate_docs = pinned_docs + scored_pool
        self._log(f"After retrieval: {len(pinned_docs)} pinned + {len(scored_pool)} scored pool = {len(all_candidate_docs)} total")

        # Coverage check on full pool (pinned + scored)
        coverage = self.check_coverage(all_candidate_docs, query_info["companies"])
        self._log("Coverage: " + " | ".join(f"{c}: {n}" for c, n in coverage.items()))

        min_needed = 5 if query_info["needs_financial"] else 3
        retry_count = 0

        while retry_count < max_retries:
            weak = [c for c, n in coverage.items() if n < min_needed]
            if not weak:
                break
            retry_count += 1
            self._log(f"Retry {retry_count}: weak → {', '.join(weak)}")

            all_hashes = {hash(d.page_content[:300]) for d in all_candidate_docs}
            for comp in weak:
                display = comp.replace("_", " ")
                retry_qs = [f"{display} {query}"]
                for sq in sub_queries[:6]:
                    retry_qs.append(f"{display} {sq}")
                for st in query_info.get("relevant_statements", []):
                    retry_qs.append(f"{display} consolidated {st.replace('_', ' ')}")
                    retry_qs.append(f"{display} {st.replace('_', ' ')}")

                for rq in dict.fromkeys(retry_qs):
                    try:
                        hits = self.vector_store.similarity_search(rq, k=10)
                    except Exception:
                        continue
                    for doc in hits:
                        if doc.metadata.get("company") == comp:
                            h = hash(doc.page_content[:300])
                            if h not in all_hashes:
                                all_hashes.add(h)
                                scored_pool.append(doc)
                                all_candidate_docs.append(doc)
                    if self.bm25:
                        try:
                            for doc in self.bm25.search(rq, k=8):
                                if doc.metadata.get("company") == comp:
                                    h = hash(doc.page_content[:300])
                                    if h not in all_hashes:
                                        all_hashes.add(h)
                                        scored_pool.append(doc)
                                        all_candidate_docs.append(doc)
                        except Exception:
                            pass

            coverage = self.check_coverage(all_candidate_docs, query_info["companies"])
            self._log(
                f"After retry {retry_count}: "
                + " | ".join(f"{c}: {n}" for c, n in coverage.items())
            )

        still_weak = [c for c, n in coverage.items() if n < min_needed]
        if still_weak:
            self._log("Using HyDE for additional retrieval…")
            hyde = self.generate_hyde_passage(query)
            if hyde:
                try:
                    all_hashes = {hash(d.page_content[:300]) for d in all_candidate_docs}
                    for doc in self.vector_store.similarity_search(hyde, k=12):
                        h = hash(doc.page_content[:300])
                        if h not in all_hashes:
                            scored_pool.append(doc)
                            all_candidate_docs.append(doc)
                except Exception:
                    pass

        # ★ v7.6: Score and filter ONLY the non-pinned pool.
        # ALL statement chunks are already pinned. Scored pool is purely supplemental.
        self._log(f"Scoring {len(scored_pool)} non-pinned chunks...")
        scored_docs = []
        for doc in scored_pool:
            rel_score = score_chunk_relevance(
                doc, query, query_info["companies"],
                query_info.get("relevant_statements", []),
                query_info["type"],
            )
            scored_docs.append((doc, rel_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # ★ v7.6: Budget = SCORED_POOL_EXTRA_SLOTS extra non-statement chunks.
        # Total output = all pinned statement chunks + up to SCORED_POOL_EXTRA_SLOTS scored.
        # This means the total is NOT capped at k — it expands to fit all statement data.
        extra_slots = SCORED_POOL_EXTRA_SLOTS
        top_scored = [doc for doc, _ in scored_docs[:extra_slots]]

        self._log(
            f"Scored pool: kept top {len(top_scored)} of {len(scored_docs)} "
            f"supplemental chunks (SCORED_POOL_EXTRA_SLOTS={extra_slots})"
        )

        # Optional LLM reranking on scored pool only
        if self.use_reranking and len(top_scored) > extra_slots:
            self._log("🧠 LLM reranking scored pool...")
            top_scored = rerank_with_llm(
                self.llm, query, top_scored, top_n=extra_slots, batch_size=10,
            )
            self._log(f"After LLM reranking: {len(top_scored)} chunks")

        # Merge: pinned first, then top scored
        final_docs = pinned_docs + top_scored

        # Final dedup just in case
        seen_final: set = set()
        deduped: List[Document] = []
        for doc in final_docs:
            h = hash(doc.page_content[:300])
            if h not in seen_final:
                seen_final.add(h)
                deduped.append(doc)
        final_docs = deduped

        self._log(f"Final merged: {len(final_docs)} chunks ({len(pinned_docs)} pinned + {len(top_scored)} scored)")

        if query_info["relevant_statements"]:
            stmt_cov = self.check_statement_coverage(
                final_docs, query_info["companies"], query_info["relevant_statements"],
            )
            self._log(
                "Statement coverage: "
                + " | ".join(f"{k}: {v}" for k, v in stmt_cov.items())
            )

        final_coverage = self.check_coverage(final_docs, query_info["companies"])
        self._log(
            f"Final: {len(final_docs)} chunks | "
            + " | ".join(f"{c}: {n}" for c, n in final_coverage.items())
        )
        return final_docs, query_info


# ---------- Section 9: Verification & Self-Correction ----------


def verify_answer(verifier_llm, answer, context, question):
    # ★ v7.3: Use full context up to 60K chars (was 20K — caused verifier to
    # flag correct answers as unsupported because company data was past the cutoff)
    ctx = context[:60000] if len(context) > 60000 else context
    prompt = VERIFICATION_PROMPT.format(
        context=ctx, question=question, answer=answer,
    )
    try:
        resp = verifier_llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        content, _ = process_llm_output(content)

        score_m = re.search(r"FAITHFULNESS_SCORE:\s*(\d+)", content)
        verdict_m = re.search(r"VERDICT:\s*([A-Z_]+)", content)
        issues_m = re.search(
            r"ISSUES:\s*(.+?)(?=\nSUBTOTAL|\nIGNORED|\nUNJUST|\nCROSS|\nVERDICT|\nSPECIFIC|$)",
            content, re.DOTALL,
        )
        subtotal_m = re.search(
            r"SUBTOTAL_ERRORS:\s*(.+?)(?=\nIGNORED|\nUNJUST|\nCROSS|\nVERDICT|\nSPECIFIC|$)",
            content, re.DOTALL,
        )
        ignored_m = re.search(
            r"IGNORED_DIRECT_VALUES:\s*(.+?)(?=\nUNJUST|\nCROSS|\nVERDICT|\nSPECIFIC|$)",
            content, re.DOTALL,
        )
        unjustified_m = re.search(
            r"UNJUSTIFIED_NOT_AVAILABLE:\s*(.+?)(?=\nCROSS|\nVERDICT|\nSPECIFIC|$)",
            content, re.DOTALL,
        )
        cross_m = re.search(
            r"CROSS_COMPANY_ERRORS:\s*(.+?)(?=\nVERDICT|\nSPECIFIC|$)",
            content, re.DOTALL,
        )
        corr_m = re.search(r"SPECIFIC_CORRECTIONS:\s*(.+?)$", content, re.DOTALL)

        score = int(score_m.group(1)) if score_m else 70
        verdict = verdict_m.group(1).strip() if verdict_m else "UNKNOWN"
        issues = issues_m.group(1).strip() if issues_m else "Unable to parse"
        subtotal_errors = subtotal_m.group(1).strip() if subtotal_m else "None found"
        ignored_vals = ignored_m.group(1).strip() if ignored_m else "None found"
        unjustified_na = unjustified_m.group(1).strip() if unjustified_m else "None found"
        cross_errors = cross_m.group(1).strip() if cross_m else "None found"
        corrections = corr_m.group(1).strip() if corr_m else "None"

        bad_flags = []
        if subtotal_errors.lower() not in ("none", "none found", "n/a", "none found."):
            score = min(score, 50)
            verdict = "NEEDS_CORRECTION"
            bad_flags.append(f"Subtotal errors: {subtotal_errors}")

        if ignored_vals.lower() not in ("none", "none found", "n/a", "none found."):
            score = min(score, 40)
            verdict = "NEEDS_CORRECTION"
            bad_flags.append(f"Ignored direct values: {ignored_vals}")

        if unjustified_na.lower() not in ("none", "none found", "n/a", "none found."):
            score = min(score, 35)
            verdict = "NEEDS_CORRECTION"
            bad_flags.append(f"Unjustified 'not available': {unjustified_na}")

        if cross_errors.lower() not in ("none", "none found", "n/a", "none found."):
            score = min(score, 40)
            verdict = "NEEDS_CORRECTION"
            bad_flags.append(f"Cross-company errors: {cross_errors}")

        if bad_flags and corrections in ("None", "none", "N/A"):
            corrections = " | ".join(bad_flags)

        return dict(
            score=min(score, 100), verdict=verdict, issues=issues,
            subtotal_errors=subtotal_errors,
            ignored_direct_values=ignored_vals,
            unjustified_not_available=unjustified_na,
            cross_company_errors=cross_errors,
            corrections=(
                corrections
                if corrections.lower() not in ("none", "n/a", "none.")
                else None
            ),
            raw=content,
        )
    except Exception as e:
        return dict(
            score=-1, verdict="VERIFICATION_ERROR",
            issues=f"Verification failed: {e}",
            subtotal_errors=None, ignored_direct_values=None,
            unjustified_not_available=None,
            cross_company_errors=None, corrections=None, raw="",
        )


def self_correct(
    primary_llm, original_prompt, original_answer, verification,
    context_text, question, verifier_llm, max_rounds: int = 2,
):
    current_answer = original_answer
    current_vr = verification

    for round_num in range(max_rounds):
        if not current_vr.get("corrections"):
            break
        if current_vr.get("score", 100) >= 80:
            break

        correction_detail = current_vr["corrections"]
        cross_err = current_vr.get("cross_company_errors", "")
        subtotal_err = current_vr.get("subtotal_errors", "")
        ignored_err = current_vr.get("ignored_direct_values", "")
        unjustified_err = current_vr.get("unjustified_not_available", "")

        retry_prompt = (
            original_prompt
            + f"\n\n=== CORRECTION ROUND {round_num + 1} ===\n"
            + f"The previous answer had these issues:\n"
            + f"Issues: {current_vr['issues']}\n"
        )
        if subtotal_err and subtotal_err.lower() not in ("none", "none found"):
            retry_prompt += (
                f"CRITICAL — Subtotal vs Total confusion: {subtotal_err}\n"
                f"You must go back and find the GRAND TOTAL rows, not subtotals.\n"
            )
        if ignored_err and ignored_err.lower() not in ("none", "none found"):
            retry_prompt += (
                f"CRITICAL — You ignored a value that was explicitly in the document: {ignored_err}\n"
                f"Use the document value directly. Do NOT recalculate.\n"
            )
        if unjustified_err and unjustified_err.lower() not in ("none", "none found"):
            retry_prompt += (
                f"CRITICAL — You said 'not available' for something that CAN be found or calculated: {unjustified_err}\n"
                f"Remember: Total Liabilities = Total Assets − Stockholders' Equity. Always try this before saying unavailable.\n"
            )
        if cross_err and cross_err.lower() not in ("none", "none found"):
            retry_prompt += (
                f"CRITICAL — Cross-company errors: {cross_err}\n"
                f"Double-check which company each number belongs to.\n"
            )
        retry_prompt += (
            f"Corrections needed: {correction_detail}\n"
            f"Regenerate fixing ONLY these issues. Keep all correct content.\n"
        )

        try:
            resp = primary_llm.invoke(retry_prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text, _ = process_llm_output(text)
            if len(text) >= len(current_answer) * 0.4:
                current_answer = text
                current_vr = verify_answer(
                    verifier_llm, current_answer, context_text, question,
                )
            else:
                break
        except Exception:
            break

    return current_answer, current_vr


# ---------- Section 10: Streamlit App ----------

st.set_page_config(
    page_title="10-K Financial Analyst v7.6",
    page_icon="📊",
    layout="wide",
)

# ★ v7.2: Added MathJax for LaTeX rendering
st.markdown(
    """
<script>
  window.MathJax = {
    tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] },
    svg: { fontCache: 'global' }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
<style>
    .stChatMessage { max-width: 100%; }
    .evidence-badge {
        display: inline-block; padding: 2px 8px; margin: 2px;
        border-radius: 12px; font-size: 0.8em;
        background-color: #262730; border: 1px solid #464655;
    }
    .coverage-good { color: #00cc66; }
    .coverage-warn { color: #ffaa00; }
    .coverage-bad  { color: #ff4444; }
    .stmt-tag {
        display: inline-block; padding: 2px 8px; margin: 2px;
        border-radius: 4px; font-size: 0.75em;
        background-color: #1a3a2a; border: 1px solid #2d8659;
        color: #66cc99;
    }
    .warn-tag {
        display: inline-block; padding: 2px 8px; margin: 2px;
        border-radius: 4px; font-size: 0.75em;
        background-color: #3a1a1a; border: 1px solid #cc4444;
        color: #ff9999;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("10-K Financial Analyst v7.6")
st.caption(
    "★ ALL Statement Chunks Pinned · Dynamic Budget · "
    "Microsoft Pattern Fix · No Cache Poisoning · Document-First · LaTeX Math"
)


# ---------- Section 11: Sidebar ----------

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input(
        "Cloud API Key", type="password", placeholder="sk-...",
        key="api_key_input",
    )
    ollama_url = st.text_input(
        "Ollama Base URL", value="http://localhost:11434",
        key="ollama_url_input",
    )

    st.markdown("---")
    st.header("⚙️ Model Settings")

    llm_options = []
    if HAS_OPENAI and api_key:
        llm_options += [f"{m} (Cloud)" for m in CLOUD_LLM_MODELS]
    if HAS_OLLAMA:
        llm_options += [f"{m} (Ollama)" for m in OLLAMA_LLM_MODELS]

    embed_options = []
    if HAS_OPENAI and api_key:
        embed_options += [f"{m} (Cloud)" for m in CLOUD_EMBED_MODELS]
    if HAS_OLLAMA:
        embed_options += [f"{m} (Ollama)" for m in OLLAMA_EMBED_MODELS]

    if not llm_options:
        st.error("No models available. Enter a Cloud API key or install langchain-ollama.")
        st.stop()

    primary_sel = st.selectbox("🧠 Primary Model", llm_options, index=0,
                               key=f"primary_sel_{'cloud' if api_key else 'ollama'}")
    verifier_idx = min(1, len(llm_options) - 1)
    verifier_sel = st.selectbox("✅ Verifier Model", llm_options, index=verifier_idx,
                                key=f"verifier_sel_{'cloud' if api_key else 'ollama'}")

    if not embed_options:
        st.error("No embedding models available.")
        st.stop()
    embed_sel = st.selectbox("📐 Embedding Model", embed_options, index=0,
                             key=f"embed_sel_{'cloud' if api_key else 'ollama'}")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    st.header("📊 A/B Comparison")
    enable_ab = st.checkbox("Side-by-side model comparison", value=False)
    ab_model_sel = None
    if enable_ab:
        ab_options = [o for o in llm_options if o != primary_sel]
        if ab_options:
            ab_model_sel = st.selectbox("Model B", ab_options, index=0)
        else:
            st.caption("Need at least two models.")
            enable_ab = False

    st.markdown("---")
    st.header("🔧 Pipeline Settings")
    chunk_size = st.select_slider("Chunk Size", [500, 800, 1000, 1200, 1500], value=1200)
    chunk_overlap = st.select_slider("Chunk Overlap", [50, 100, 200, 300], value=200)
    base_k = st.slider("Base Top-K", 6, 30, 15)
    context_budget = st.select_slider(
        "Context Budget (chars)",
        [60000, 80000, 100000, 120000, 160000, 200000],
        value=120000,
    )

    st.markdown("---")
    st.header("🚀 Features")
    use_verification = st.checkbox("✅ Dual-Model Verification", True)
    use_self_correction = False  # ★ v7.6: Removed — slow and unreliable
    use_tables = st.checkbox(
        "📊 Table Extraction", HAS_PDFPLUMBER, disabled=not HAS_PDFPLUMBER,
    )
    use_hybrid = st.checkbox(
        "🔍 Hybrid Search (BM25)", HAS_BM25, disabled=not HAS_BM25,
    )
    use_section_boost = st.checkbox("📑 Section-Aware Boost", True)
    use_mmr = st.checkbox("🎯 MMR Diversity", True)
    use_hyde = st.checkbox("🧪 HyDE", True)
    use_reranking = st.checkbox("🏆 LLM Reranking", False)
    max_retries = st.slider("Max Retrieval Retries", 0, 3, 2)
    max_correction_rounds = 0  # disabled

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Rebuild Index"):
            import shutil
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                CACHE_DIR.mkdir(exist_ok=True)
            for key in [
                "vector_store", "bm25_index", "all_chunks",
                "company_stats", "config_hash", "statement_inventory",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption(
        f"pdfplumber: {'✅' if HAS_PDFPLUMBER else '❌'} | "
        f"BM25: {'✅' if HAS_BM25 else '❌'} | "
        f"OpenAI: {'✅' if HAS_OPENAI else '❌'} | "
        f"Ollama: {'✅' if HAS_OLLAMA else '❌'}"
    )


# ---------- Section 12: Model Init ----------


def _parse_selection(sel):
    m = re.match(r"(.+?)\s*\((Cloud|Ollama)\)", sel)
    return (m.group(1).strip(), m.group(2)) if m else (sel, "Cloud")


def create_llm(model_name, provider, temp=0.0):
    if provider == "Ollama":
        return ChatOllama(model=model_name, base_url=ollama_url, temperature=temp)
    return ChatOpenAI(
        model=model_name, base_url=ZHIZENGZENG_BASE_URL,
        api_key=api_key, temperature=temp, request_timeout=180,
    )


@st.cache_resource
def create_embeddings(model_name, provider, _api_key=None, _ollama_url=None):
    if provider == "Ollama":
        return OllamaEmbeddings(
            model=model_name, base_url=_ollama_url or "http://localhost:11434",
        )
    return OpenAIEmbeddings(
        model=model_name, base_url=ZHIZENGZENG_BASE_URL, api_key=_api_key,
    )


primary_model, primary_prov = _parse_selection(primary_sel)
verifier_model, verifier_prov = _parse_selection(verifier_sel)
embed_model, embed_prov = _parse_selection(embed_sel)

if primary_prov == "Cloud" and not api_key:
    st.info("👈 Enter your Cloud API Key in the sidebar to get started.")
    st.stop()

embeddings = create_embeddings(embed_model, embed_prov, api_key, ollama_url)
primary_llm = create_llm(primary_model, primary_prov, temperature)
verifier_llm = create_llm(verifier_model, verifier_prov, 0.0)
ab_llm = None
if enable_ab and ab_model_sel:
    ab_m, ab_p = _parse_selection(ab_model_sel)
    ab_llm = create_llm(ab_m, ab_p, temperature)


# ---------- Section 13: PDF Upload & Indexing ----------

uploaded_files = st.file_uploader(
    "📄 Upload 10-K PDF files",
    accept_multiple_files=True, type=["pdf"], key="pdf_uploader",
)

if uploaded_files:
    files_hash = compute_files_hash(uploaded_files)
    # ★ v7.4: VERSION string included in hash so old v7.2/v7.3 caches are
    # never silently reused after a code update. Change "v7.4" when making
    # future structural changes to the indexing pipeline.
    INDEX_VERSION = "v7.6"
    config_hash = hashlib.md5(
        f"{INDEX_VERSION}|{files_hash}|{embed_model}|{embed_prov}"
        f"|{chunk_size}|{chunk_overlap}|{use_tables}".encode()
    ).hexdigest()[:16]

    faiss_path = CACHE_DIR / f"faiss_{config_hash}"
    chunks_path = CACHE_DIR / f"chunks_{config_hash}.pkl"
    stats_path = CACHE_DIR / f"stats_{config_hash}.json"
    inv_path = CACHE_DIR / f"inventory_{config_hash}.json"

    need_build = (
        "vector_store" not in st.session_state
        or st.session_state.get("config_hash") != config_hash
    )

    if need_build:
        if faiss_path.exists() and chunks_path.exists():
            t0 = time.time()
            with st.spinner("⚡ Loading cached index..."):
                st.session_state.vector_store = FAISS.load_local(
                    str(faiss_path), embeddings,
                    allow_dangerous_deserialization=True,
                )
                with open(chunks_path, "rb") as f:
                    st.session_state.all_chunks = pickle.load(f)
                st.session_state.company_stats = (
                    json.load(open(stats_path)) if stats_path.exists() else {}
                )
                st.session_state.statement_inventory = (
                    json.load(open(inv_path)) if inv_path.exists() else {}
                )
                st.session_state.bm25_index = (
                    BM25Retriever(st.session_state.all_chunks) if HAS_BM25 else None
                )
                st.session_state.config_hash = config_hash
            st.success(
                f"⚡ Loaded from cache in {time.time()-t0:.1f}s — "
                f"{len(st.session_state.all_chunks)} chunks"
            )
        else:
            prog = st.progress(0, text="Starting PDF processing...")
            all_documents = []
            company_stats = {}

            with tempfile.TemporaryDirectory() as tmp:
                n_files = len(uploaded_files)
                for fi, file in enumerate(uploaded_files):
                    prog.progress(
                        fi / (n_files + 2),
                        text=f"📄 Processing {file.name}...",
                    )
                    path = os.path.join(tmp, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())

                    quick = PyPDFLoader(path).load()
                    page_texts = [d.page_content for d in quick[:10]]
                    company = detect_company_name(file.name, page_texts)

                    docs = load_pdf_enhanced(path, company, use_tables)
                    if not docs:
                        docs = quick
                        for d in docs:
                            d.metadata["company"] = company
                            d.metadata["section"] = detect_10k_section(d.page_content)

                    all_documents.extend(docs)
                    company_stats[company] = company_stats.get(company, 0) + len(docs)

                    stmt_count = sum(
                        1 for d in docs if d.metadata.get("is_statement_chunk")
                    )
                    if stmt_count:
                        prog.progress(
                            fi / (n_files + 2),
                            text=(
                                f"📄 {file.name}: {len(docs)} pages, "
                                f"★ {stmt_count} statement chunks"
                            ),
                        )

                prog.progress(0.50, text="✂️ Table-aware chunking...")
                chunks = table_aware_chunking(all_documents, chunk_size, chunk_overlap)

                inventory = build_statement_inventory(chunks)

                prog.progress(
                    0.60,
                    text=f"🧮 Building vector index ({len(chunks)} chunks)...",
                )
                BATCH = 200
                if len(chunks) > BATCH:
                    vs = FAISS.from_documents(chunks[:BATCH], embeddings)
                    rest = chunks[BATCH:]
                    n_batches = (len(rest) + BATCH - 1) // BATCH
                    for bi in range(n_batches):
                        s = bi * BATCH
                        e = min((bi + 1) * BATCH, len(rest))
                        pct = 0.60 + 0.35 * (bi + 1) / n_batches
                        prog.progress(pct, text=f"🧮 Embedding batch {bi+2}/{n_batches+1}...")
                        vs.merge_from(FAISS.from_documents(rest[s:e], embeddings))
                else:
                    vs = FAISS.from_documents(chunks, embeddings)

                st.session_state.vector_store = vs
                st.session_state.bm25_index = BM25Retriever(chunks) if HAS_BM25 else None
                st.session_state.all_chunks = chunks
                st.session_state.company_stats = company_stats
                st.session_state.statement_inventory = inventory
                st.session_state.config_hash = config_hash

                prog.progress(0.97, text="💾 Saving to cache...")
                vs.save_local(str(faiss_path))
                with open(chunks_path, "wb") as f:
                    pickle.dump(chunks, f)
                with open(stats_path, "w") as f:
                    json.dump(company_stats, f)
                with open(inv_path, "w") as f:
                    json.dump(inventory, f)

                prog.progress(1.0, text="✅ Done!")
                time.sleep(0.5)
                prog.empty()

            n_stmt = sum(1 for c in chunks if c.metadata.get("is_statement_chunk"))
            n_table = sum(1 for c in chunks if c.metadata.get("is_table_chunk"))
            n_text = len(chunks) - n_stmt - n_table
            st.success(
                f"✅ {n_files} PDFs → {len(chunks)} chunks "
                f"(★ {n_stmt} statement | 📊 {n_table} table | 📝 {n_text} text)"
            )

    # Company stat cards
    if "company_stats" in st.session_state:
        cols = st.columns(len(st.session_state.company_stats))
        inv = st.session_state.get("statement_inventory", {})
        for i, (comp, pages) in enumerate(st.session_state.company_stats.items()):
            with cols[i]:
                st.metric(
                    f"{EMOJI_MAP.get(comp, '⚪')} {comp}", f"{pages} pages",
                )
                if comp in inv and inv[comp]:
                    for stmt_label in inv[comp]:
                        st.markdown(
                            f'<span class="stmt-tag">📋 {stmt_label}</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<span class="warn-tag">⚠️ No statement chunks detected</span>',
                        unsafe_allow_html=True,
                    )

    # ---------- Section 14: Chat Interface ----------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(safe_render(msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the uploaded filings...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.status("🚀 Running Agentic RAG Pipeline v7.2...", expanded=True) as status:
            times = {}

            rag = AgenticRAGController(
                vector_store=st.session_state.vector_store,
                bm25_retriever=(st.session_state.bm25_index if use_hybrid else None),
                llm=verifier_llm,
                all_chunks=st.session_state.get("all_chunks", []),
                use_mmr=use_mmr, use_hybrid=use_hybrid,
                use_section_boost=use_section_boost,
                use_reranking=use_reranking,
                context_budget=context_budget,
            )

            # Step 1: Agentic retrieval
            t0 = time.time()
            st.write("🤖 **Step 1:** LLM query analysis → agentic retrieval + ranking...")
            known_entities = list(st.session_state.get("company_stats", {}).keys())
            retrieved, query_info = rag.retrieve(
                user_input, known_entities=known_entities,
                base_k=base_k, max_retries=max_retries,
            )
            for log_entry in rag.log:
                st.write(f"   → {log_entry}")
            times["Retrieval"] = time.time() - t0

            # Coverage display
            coverage = rag.check_coverage(retrieved, query_info["companies"])
            cov_parts = []
            for comp, count in coverage.items():
                emoji = EMOJI_MAP.get(comp, "⚪")
                if count >= 5:
                    cls = "coverage-good"
                elif count >= 3:
                    cls = "coverage-warn"
                else:
                    cls = "coverage-bad"
                cov_parts.append(f'<span class="{cls}">{emoji} {comp}: {count}</span>')
            st.markdown(
                f"   **Coverage:** {' | '.join(cov_parts)}", unsafe_allow_html=True,
            )

            stmt_in_ret = sum(1 for d in retrieved if d.metadata.get("is_statement_chunk"))
            if stmt_in_ret:
                st.write(f"   📌 **{stmt_in_ret} statement chunk(s)** pinned in context (ALL statement chunks guaranteed)")

            # Step 2: Build context
            t0 = time.time()
            st.write("📋 **Step 2:** Building company-grouped context (weighted budget)...")

            # ★ v7.6: Auto-reduce context for Ollama — local models can't handle
            # 95K chars (~24K tokens). Cap at 30K chars (~7.5K tokens) for Ollama.
            effective_budget = context_budget
            if primary_prov == "Ollama":
                OLLAMA_CONTEXT_CAP = 30_000
                if effective_budget > OLLAMA_CONTEXT_CAP:
                    effective_budget = OLLAMA_CONTEXT_CAP
                    st.write(
                        f"   ⚡ Ollama mode: context capped at {OLLAMA_CONTEXT_CAP:,} chars "
                        f"to avoid 502 timeout (local models have limited context window)"
                    )

            context_text = build_company_grouped_context(retrieved, budget_chars=effective_budget)
            context_tokens_est = estimate_tokens(context_text)
            st.write(
                f"   Context: {len(context_text):,} chars (~{context_tokens_est:,} tokens)"
            )
            # Show per-company statement chunk counts (explains budget weighting)
            all_chunks_for_weight = st.session_state.get("all_chunks", [])
            if all_chunks_for_weight:
                stmt_cnt = {}
                for c in all_chunks_for_weight:
                    comp = c.metadata.get("company", "?")
                    if c.metadata.get("is_statement_chunk"):
                        stmt_cnt[comp] = stmt_cnt.get(comp, 0) + 1
                if stmt_cnt:
                    wt_str = " | ".join(
                        f"{EMOJI_MAP.get(c,'⚪')} {c}: {n} stmt chunks"
                        for c, n in sorted(stmt_cnt.items())
                    )
                    st.write(f"   Budget weights: {wt_str}")
            times["Context"] = time.time() - t0

            # Step 3: Generate
            t0 = time.time()
            st.write(
                f"🧠 **Step 3:** Chain-of-Extraction generation with "
                f"**{primary_model}** ({primary_prov})..."
            )
            history = format_chat_history(st.session_state.messages)
            prompt_text = GENERATION_PROMPT.format(
                context=context_text, question=user_input, history=history,
            )
            resp = primary_llm.invoke(prompt_text)
            answer = resp.content if hasattr(resp, "content") else str(resp)
            answer, reasoning = process_llm_output(answer, primary_model)
            times["Generation"] = time.time() - t0

            # Step 4: Verification
            vr = None
            if use_verification:
                t0 = time.time()
                st.write(f"✅ **Step 4:** Verifying with **{verifier_model}**...")
                vr = verify_answer(verifier_llm, answer, context_text, user_input)

                if vr["score"] >= 0:
                    badge = (
                        "🟢" if vr["score"] >= 85
                        else "🟡" if vr["score"] >= 60
                        else "🔴"
                    )
                    st.write(f"   {badge} Score: **{vr['score']}/100** — {vr['verdict']}")

                    subtotal_err = vr.get("subtotal_errors", "")
                    if subtotal_err and subtotal_err.lower() not in (
                        "none", "none found", "n/a", "none found.",
                    ):
                        st.write(f"   🚨 **Subtotal/Total confusion:** {subtotal_err[:300]}")

                    ignored_err = vr.get("ignored_direct_values", "")
                    if ignored_err and ignored_err.lower() not in (
                        "none", "none found", "n/a", "none found.",
                    ):
                        st.write(f"   🚨 **Ignored direct document values:** {ignored_err[:300]}")

                    unjust_err = vr.get("unjustified_not_available", "")
                    if unjust_err and unjust_err.lower() not in (
                        "none", "none found", "n/a", "none found.",
                    ):
                        st.write(f"   🚨 **Unjustified 'not available':** {unjust_err[:300]}")

                    cross_err = vr.get("cross_company_errors", "")
                    if cross_err and cross_err.lower() not in (
                        "none", "none found", "n/a", "none found.",
                    ):
                        st.write(f"   🚨 **Cross-company errors:** {cross_err[:300]}")

                    if vr.get("issues") and vr["issues"].lower() not in (
                        "none", "none found", "n/a", "none found.",
                    ):
                        st.write(f"   ⚠️ Issues: {vr['issues'][:300]}")

                    if (
                        use_self_correction
                        and vr["score"] < 75
                        and vr.get("corrections")
                    ):
                        st.write(
                            f"   🔁 Self-correcting (up to {max_correction_rounds} rounds)..."
                        )
                        answer, vr = self_correct(
                            primary_llm, prompt_text, answer, vr,
                            context_text, user_input, verifier_llm,
                            max_rounds=max_correction_rounds,
                        )
                        answer, reasoning = process_llm_output(answer, primary_model)
                        badge2 = (
                            "🟢" if vr["score"] >= 85
                            else "🟡" if vr["score"] >= 60
                            else "🔴"
                        )
                        st.write(
                            f"   {badge2} After correction: "
                            f"**{vr['score']}/100** — {vr['verdict']}"
                        )
                times["Verification"] = time.time() - t0

            # Step 5: A/B
            ab_answer, ab_reasoning = None, None
            if enable_ab and ab_llm:
                t0 = time.time()
                ab_m, ab_p = _parse_selection(ab_model_sel)
                st.write(f"📊 **Step 5:** A/B with **{ab_m}** ({ab_p})...")
                try:
                    ab_resp = ab_llm.invoke(prompt_text)
                    ab_raw = ab_resp.content if hasattr(ab_resp, "content") else str(ab_resp)
                    ab_answer, ab_reasoning = process_llm_output(ab_raw, ab_m)
                except Exception as e:
                    st.write(f"   ❌ Model B failed: {e}")
                times["A/B"] = time.time() - t0

            total_time = sum(times.values())
            status.update(
                label=f"✅ Pipeline completed in {total_time:.1f}s", state="complete",
            )

        # Chunk viewer
        evidence = extract_evidence_sources(retrieved)
        with st.expander(
            f"📎 View {len(retrieved)} Retrieved Chunks "
            f"({stmt_in_ret} stmt) | ⏱️ {total_time:.1f}s"
        ):
            tcols = st.columns(min(len(times), 5))
            for i, (step, t) in enumerate(times.items()):
                tcols[i % len(tcols)].metric(step, f"{t:.1f}s")
            st.markdown("---")
            for i, doc in enumerate(retrieved):
                comp = doc.metadata.get("company", "Unknown")
                page = doc.metadata.get("page", "?")
                sec = doc.metadata.get("section", "Other")
                stmt = doc.metadata.get("statement_type")
                is_tbl = doc.metadata.get("is_table_chunk", False)
                is_stmt = doc.metadata.get("is_statement_chunk", False)
                year = doc.metadata.get("year", "")
                qs = doc.metadata.get("quality_score", "")
                em = EMOJI_MAP.get(comp, "⚪")

                tags = ""
                if is_stmt:
                    tags += f" ★ **{stmt.replace('_', ' ')}**"
                    if year:
                        tags += f" ({year})"
                elif is_tbl:
                    tags += " 📊 TABLE"
                if qs:
                    tags += f" | Q={qs}"

                st.markdown(
                    f"**Chunk {i+1}** | {em} **{comp}** | "
                    f"📑 {sec} | Page {page}{tags}"
                )
                preview_len = 1200 if is_stmt else 600
                st.code(doc.page_content[:preview_len], language=None)
                if len(doc.page_content) > preview_len:
                    st.caption(f"... ({len(doc.page_content)} chars total)")
                st.markdown("---")

        # Verification expander
        if vr and vr["score"] >= 0:
            badge = (
                "🟢" if vr["score"] >= 85
                else "🟡" if vr["score"] >= 60
                else "🔴"
            )
            with st.expander(f"{badge} Faithfulness: {vr['score']}/100 — {vr['verdict']}"):
                st.markdown(f"**Verifier:** {verifier_model}")
                st.markdown(f"**Score:** {vr['score']}/100")
                st.markdown(f"**Verdict:** {vr['verdict']}")
                st.markdown(f"**Issues:** {vr['issues']}")
                subtotal_err = vr.get("subtotal_errors", "")
                if subtotal_err:
                    st.markdown(f"**Subtotal/Total Errors:** {subtotal_err}")
                ignored_err = vr.get("ignored_direct_values", "")
                if ignored_err:
                    st.markdown(f"**Ignored Direct Values:** {ignored_err}")
                unjust_err = vr.get("unjustified_not_available", "")
                if unjust_err:
                    st.markdown(f"**Unjustified 'Not Available':** {unjust_err}")
                cross_err = vr.get("cross_company_errors", "")
                if cross_err:
                    st.markdown(f"**Cross-Company Errors:** {cross_err}")
                if vr.get("raw"):
                    st.code(vr["raw"][:2000], language=None)

        # Reasoning expander
        if reasoning:
            with st.expander("🧠 Model Reasoning (Chain of Thought)"):
                st.markdown(reasoning)

        # Answer display
        if enable_ab and ab_answer:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"### 🅰️ {primary_model}")
                with st.chat_message("assistant"):
                    st.markdown(safe_render(answer), unsafe_allow_html=True)
            with col_b:
                ab_m, _ = _parse_selection(ab_model_sel)
                st.markdown(f"### 🅱️ {ab_m}")
                with st.chat_message("assistant"):
                    st.markdown(safe_render(ab_answer), unsafe_allow_html=True)
                    if ab_reasoning:
                        with st.expander("🧠 Model B Reasoning"):
                            st.markdown(ab_reasoning)
        else:
            with st.chat_message("assistant"):
                st.markdown(safe_render(answer), unsafe_allow_html=True)

                if evidence:
                    st.markdown("---")
                    st.markdown("**📎 Evidence Sources:**")
                    badges = "  ".join(
                        f"`{s['emoji']} {s['company']} — {s['section']} — p.{s['page']}`"
                        for s in evidence[:15]
                    )
                    st.markdown(badges, unsafe_allow_html=True)

                if vr and 0 <= vr["score"] < 60:
                    st.warning(
                        "⚠️ Low confidence. Some claims may not be "
                        "fully supported. Please verify critical figures."
                    )

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info("""
### 👋 Welcome! Upload 10-K annual reports to get started.

**Upload PDFs for:** Alphabet (Google), Amazon, Microsoft — or any company.

**Sample Questions:**
- 💰 What is the gross profit for each company?
- 📊 Compare total assets, total liabilities, and stockholders' equity for all three companies.
- ☁️ What is the cloud revenue for Amazon, Google, and Microsoft?
- ⚠️ What AI-related risks do these companies mention?
- ⚖️ What regulatory or legal proceedings is each company involved in?
- 📈 How did Amazon's revenue change from 2023 to 2024?
- 🏦 Calculate the debt-to-equity ratio for each company.
        """)
    with col_b:
        st.markdown("### 🏆 v7.6 Root-Cause Fix")
        st.markdown("""
📌 **ALL Statement Chunks Pinned** — Every statement chunk (not just 3 per company) is guaranteed in the final context, bypassing score filter entirely

📈 **Dynamic Budget** — Total output = ALL pinned statements + 20 supplemental scored chunks. No fixed k=28 cap cutting off data

✅ **All v7.5 fixes** — Pinned architecture, year int cast fix

✅ **All v7.4 fixes** — Microsoft pattern matching, cache versioning, notes false-positive fix
        """)
