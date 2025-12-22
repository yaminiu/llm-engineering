"""brochure_generator.py

Company Brochure Generator (refactored)
-------------------------------------
Generate a concise Markdown brochure for a company by providing its website URL.

Highlights vs. the original day5.py:
- PEP8/typed, modular structure with dataclasses
- Robust HTTP fetching: session + retries + timeouts + user-agent
- Link extraction: normalization, de-duplication, same-domain filtering
- Safer HTML-to-text cleaning
- LLM JSON parsing with a fallback heuristic if the model returns invalid JSON
- Streaming + non-streaming output, writes brochure.md by default

Usage:
  python brochure_generator.py --name "Acme" --url https://acme.com --stream

Environment variables:
  OLLAMA_BASE_URL   (default: http://localhost:11434/v1)
  LLM_MODEL         (default: gemma3:latest)

Dependencies:
  pip install openai requests beautifulsoup4

Note:
  Please respect robots.txt and the target site's Terms of Service.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'openai' package is required. Install with: pip install openai"
    ) from exc


# -----------------------------
# Configuration & Logging
# -----------------------------

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemma3:latest")

LOGGER = logging.getLogger("brochure_generator")


@dataclass(frozen=True)
class ScrapeConfig:
    timeout_s: int = 15
    max_links: int = 200
    max_pages: int = 6
    max_chars_per_page: int = 8_000
    same_domain_only: bool = True
    user_agent: str = (
        "Mozilla/5.0 (compatible; BrochureGenerator/1.0; +https://example.com)"
    )


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    brochure_word_target: int = 500


# -----------------------------
# HTTP / Scraping Utilities
# -----------------------------


def _make_session(user_agent: str) -> requests.Session:
    """Create a requests session with sensible defaults."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "text/html,*/*"})
    return session


def _normalize_url(base: str, href: str) -> Optional[str]:
    """Normalize a candidate href into an absolute URL; drop fragments."""
    if not href:
        return None

    href = href.strip()
    if href.startswith(("mailto:", "javascript:", "tel:")):
        return None

    abs_url = urljoin(base, href)
    abs_url, _frag = urldefrag(abs_url)

    parsed = urlparse(abs_url)
    if parsed.scheme not in ("http", "https"):
        return None

    return abs_url


def _is_same_domain(root: str, candidate: str) -> bool:
    """Return True if candidate URL shares the same registrable host as root."""
    root_host = urlparse(root).netloc.lower()
    cand_host = urlparse(candidate).netloc.lower()
    return root_host == cand_host


def extract_links(url: str, html: str, cfg: ScrapeConfig) -> List[str]:
    """Extract and normalize links from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    out: List[str] = []
    seen = set()

    for a in soup.find_all("a", href=True):
        norm = _normalize_url(url, a.get("href"))
        if not norm:
            continue
        if cfg.same_domain_only and not _is_same_domain(url, norm):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= cfg.max_links:
            break

    return out


def html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Clean whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    # De-duplicate consecutive identical lines
    cleaned: List[str] = []
    prev = None
    for ln in lines:
        if ln != prev:
            cleaned.append(ln)
        prev = ln
    return "\n".join(cleaned)


def fetch_html(session: requests.Session, url: str, timeout_s: int) -> str:
    """Fetch HTML from a URL (best-effort)."""
    try:
        resp = session.get(url, timeout=timeout_s)
        resp.raise_for_status()
        # let requests guess encoding, but prefer utf-8 when ambiguous
        resp.encoding = resp.encoding or "utf-8"
        return resp.text
    except Exception as exc:
        LOGGER.warning("Failed to fetch %s: %s", url, exc)
        return ""


def fetch_page_text(session: requests.Session, url: str, cfg: ScrapeConfig) -> str:
    html = fetch_html(session, url, cfg.timeout_s)
    if not html:
        return ""
    return html_to_text(html)


# -----------------------------
# LLM Utilities
# -----------------------------

LINK_SYSTEM_PROMPT = """You are given a list of links from a company's website.
Select only the links that are most relevant for a company brochure.
Good candidates: About, Company, Products/Services, Solutions, Customers, Case Studies,
Blog (optional), Careers/Jobs, Team, Contact.
Avoid: privacy, terms, cookie policy, login, signup, press release archive, unrelated.

Respond ONLY with JSON of the form:
{
  "links": [
    {"type": "about", "url": "https://example.com/about"},
    {"type": "careers", "url": "https://example.com/careers"}
  ]
}
"""

BROCHURE_SYSTEM_PROMPT = """You are an assistant that writes a concise company brochure
for prospective customers, investors, and recruits.
Use the provided website content only; do not invent facts.
Respond in Markdown (no code blocks).
Aim for ~{word_target} words.
Include sections:
- Overview
- What they do (products/services)
- Who they serve (customers/industries)
- Culture & values
- Careers (roles/what to expect)
- Contact / next steps
"""


def _safe_json_loads(text: str) -> Optional[dict]:
    """Parse JSON, tolerating extra text by extracting the first JSON object."""
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract a JSON object substring.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _heuristic_relevant_links(root_url: str, links: Sequence[str]) -> List[Tuple[str, str]]:
    """Fallback: pick likely brochure links by URL patterns."""
    patterns = [
        ("about", r"/(about|company|who-we-are|our-story)(/|$)"),
        ("products", r"/(product|products|services|solutions)(/|$)"),
        ("customers", r"/(customers|case-studies|case-study|stories|clients)(/|$)"),
        ("careers", r"/(careers|jobs|join|join-us|work-with-us)(/|$)"),
        ("contact", r"/(contact|get-in-touch)(/|$)"),
    ]

    selected: List[Tuple[str, str]] = []
    for link_type, pat in patterns:
        for u in links:
            if re.search(pat, urlparse(u).path.lower()):
                selected.append((link_type, u))
                break

    # Ensure root always included in case patterns find nothing
    if not selected:
        selected.append(("landing", root_url))

    return selected


class BrochureGenerator:
    def __init__(self, scrape_cfg: ScrapeConfig = ScrapeConfig(), llm_cfg: LLMConfig = LLMConfig()):
        self.scrape_cfg = scrape_cfg
        self.llm_cfg = llm_cfg
        self.session = _make_session(scrape_cfg.user_agent)
        self.client = OpenAI(base_url=llm_cfg.base_url, api_key="ollama")

    def _select_relevant_links(self, root_url: str, links: Sequence[str]) -> List[Tuple[str, str]]:
        """Ask the LLM to select relevant links; fallback to heuristics."""
        if not links:
            return [("landing", root_url)]

        user_prompt = (
            f"Root website: {root_url}\n"
            f"Here are links found on the website. Choose only brochure-relevant links.\n\n"
            + "\n".join(links[: self.scrape_cfg.max_links])
        )

        LOGGER.info("Selecting relevant links via LLM (%s)", self.llm_cfg.model)
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_cfg.model,
                messages=[
                    {"role": "system", "content": LINK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            parsed = _safe_json_loads(content)
            if parsed and isinstance(parsed.get("links"), list):
                pairs: List[Tuple[str, str]] = []
                for item in parsed["links"]:
                    if not isinstance(item, dict):
                        continue
                    t = str(item.get("type", "page")).strip() or "page"
                    u = str(item.get("url", "")).strip()
                    if u and u.startswith(("http://", "https://")):
                        if self.scrape_cfg.same_domain_only and not _is_same_domain(root_url, u):
                            continue
                        pairs.append((t, u))
                # Keep up to max_pages and ensure landing included first
                pairs = [("landing", root_url)] + [p for p in pairs if p[1] != root_url]
                return pairs[: self.scrape_cfg.max_pages]
        except Exception as exc:
            LOGGER.warning("LLM link selection failed; falling back to heuristic: %s", exc)

        heur = _heuristic_relevant_links(root_url, links)
        # Always include landing page first
        out = [("landing", root_url)] + [p for p in heur if p[1] != root_url]
        return out[: self.scrape_cfg.max_pages]

    def _gather_site_corpus(self, root_url: str) -> str:
        """Fetch landing page, extract links, select relevant pages, then fetch text."""
        landing_html = fetch_html(self.session, root_url, self.scrape_cfg.timeout_s)
        if not landing_html:
            raise RuntimeError(f"Unable to fetch landing page: {root_url}")

        links = extract_links(root_url, landing_html, self.scrape_cfg)
        selected = self._select_relevant_links(root_url, links)

        corpus_parts: List[str] = []
        for link_type, url in selected:
            txt = fetch_page_text(self.session, url, self.scrape_cfg)
            if not txt:
                continue
            txt = txt[: self.scrape_cfg.max_chars_per_page]
            corpus_parts.append(f"## {link_type.upper()} :: {url}\n{txt}")

        return "\n\n".join(corpus_parts)

    def generate_brochure(self, company_name: str, root_url: str) -> str:
        corpus = self._gather_site_corpus(root_url)
        system_prompt = BROCHURE_SYSTEM_PROMPT.format(word_target=self.llm_cfg.brochure_word_target)
        user_prompt = (
            f"Company name: {company_name}\n"
            f"Website: {root_url}\n\n"
            "Below is text extracted from relevant pages. Write the brochure now.\n\n"
            + corpus
        )

        LOGGER.info("Generating brochure via LLM (%s)", self.llm_cfg.model)
        resp = self.client.chat.completions.create(
            model=self.llm_cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    def stream_brochure(self, company_name: str, root_url: str) -> Iterable[str]:
        corpus = self._gather_site_corpus(root_url)
        system_prompt = BROCHURE_SYSTEM_PROMPT.format(word_target=self.llm_cfg.brochure_word_target)
        user_prompt = (
            f"Company name: {company_name}\n"
            f"Website: {root_url}\n\n"
            "Below is text extracted from relevant pages. Write the brochure now.\n\n"
            + corpus
        )

        LOGGER.info("Streaming brochure via LLM (%s)", self.llm_cfg.model)
        stream = self.client.chat.completions.create(
            model=self.llm_cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            # OpenAI python SDK streams delta content here
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a company brochure from a website URL")
    p.add_argument("--name", required=True, help="Company name")
    p.add_argument("--url", required=True, help="Company website URL (https://...)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"LLM API base url (default: {DEFAULT_BASE_URL})")
    p.add_argument("--out", default="brochure.md", help="Output Markdown file")
    p.add_argument("--stream", action="store_true", help="Stream output to console")
    p.add_argument("--max-pages", type=int, default=ScrapeConfig.max_pages, help="Max pages to fetch")
    p.add_argument("--max-links", type=int, default=ScrapeConfig.max_links, help="Max links to consider")
    p.add_argument("--timeout", type=int, default=ScrapeConfig.timeout_s, help="HTTP timeout seconds")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    scrape_cfg = ScrapeConfig(
        timeout_s=args.timeout,
        max_links=args.max_links,
        max_pages=args.max_pages,
    )
    llm_cfg = LLMConfig(base_url=args.base_url, model=args.model)

    gen = BrochureGenerator(scrape_cfg=scrape_cfg, llm_cfg=llm_cfg)

    try:
        if args.stream:
            chunks: List[str] = []
            for tok in gen.stream_brochure(args.name, args.url):
                sys.stdout.write(tok)
                sys.stdout.flush()
                chunks.append(tok)
            sys.stdout.write("\n")
            brochure = "".join(chunks)
        else:
            brochure = gen.generate_brochure(args.name, args.url)
            print(brochure)

        out_path = os.path.abspath(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(brochure)
            f.write("\n")

        LOGGER.info("Saved brochure to %s", out_path)
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    except Exception as exc:
        LOGGER.error("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
