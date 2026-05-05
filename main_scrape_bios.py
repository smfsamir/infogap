import click
import os
import re
import requests
import tqdm
import dill
from wikidata.client import Client
from functools import partial
import pywikibot
import pandas as pd
import ipdb
import polars as pl
from collections import OrderedDict
from typing import Optional, List, Tuple, Iterable, Dict
import loguru
import unicodedata
import urllib.parse
import yaml
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry

from flowmason.flowmason import conduct, load_artifact, load_artifact_with_step_name, SingletonStep

from packages.constants import BIO_SAVE_DIR, SCRATCH_DIR
from wikigap_topics_scrape import selected_topics
from wiki_text_process_test.examine_cache import examine_cache

logger = loguru.logger

# ----------------------------
# HTTP utilities (robust requests)
# ----------------------------
HEADERS = {"User-Agent": "WikiGapPipeline/0.1 (mailto:you@example.com)"}


def _session() -> requests.Session:
    """Return a requests session with retry/backoff for resilient API calls.

    We centralize retries and headers here so every function that hits
    MediaWiki or Wikidata APIs shares the same behavior.
    """
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s


# ----------------------------
# Core helpers (existing flow, made safer)
# ----------------------------

def get_wikidata_id(topic: str, lang: str, **kwargs) -> Optional[str]:
    """Resolve a Wikipedia title in a given language to its Wikidata QID.

    Uses wbgetentities with `sites={lang}wiki&titles={topic}` and returns the
    first entity id if available. Returns None and logs on failure.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "sites": f"{lang}wiki",
        "titles": topic,
        "props": "info",
        "format": "json",
    }
    r = _session().get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    entities = data.get("entities", {})
    if entities:
        return list(entities.keys())[0]
    logger.error(f"Could not find Wikidata ID for topic='{topic}' ({lang})")
    return None


def get_interlanguage_links(wikidata_id: str, **kwargs) -> Dict[str, str]:
    """Fetch interlanguage sitelinks (urls) for a given QID.

    Returns a mapping like {"enwiki": url, "frwiki": url, ...}.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "props": "sitelinks/urls",
        "format": "json",
    }
    r = _session().get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    sitelinks = data.get("entities", {}).get(wikidata_id, {}).get("sitelinks", {})
    return {site: details.get("url", "") for site, details in sitelinks.items()}


def extract_article_title_from_urls(urls: Dict[str, str], lang: str, **kwargs) -> Optional[str]:
    """Extract and return the page title for a specific language from Wikidata sitelinks.

    We look up `{lang}wiki` in the sitelinks dict, then parse the title from the URL.
    Returns None and logs if the title/URL is missing or unparsable.
    """
    lang_wiki = f"{lang}wiki"
    url = urls.get(lang_wiki)
    if not url:
        logger.error(f"No sitelink for {lang_wiki}")
        return None
    match = re.search(r"/wiki/([^#?]*)", url)
    if match:
        return match.group(1).replace("_", " ")
    logger.error(f"Can't extract title for url: {url}")
    return None


def get_wikipedia_text(article_title: str, lang: str, **kwargs) -> Optional[str]:
    """Fetch the plain-text content of a Wikipedia article via Action API extracts.

    Returns the plain text string or None if not found.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        # Important: keep human-readable title, let requests handle encoding via params
        "titles": article_title,
    }
    r = _session().get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for _, page_data in pages.items():
        if "extract" in page_data:
            return page_data["extract"]
    logger.error(f"Could not find extract for title: {article_title} ({lang})")
    return None


def make_lang_article_dict(en_article_title: Optional[str], en_lang: str,
                            tgt_article_title: Optional[str], tgt_lang: str, **kwargs) -> Dict[str, Optional[str]]:
    """Return a small mapping of two languages → titles (used by the existing flow).

    This keeps compatibility with the current flowmason pipeline steps.
    """
    return {en_lang: en_article_title, tgt_lang: tgt_article_title}


def clean_text(text: str) -> str:
    """Normalize and remove control/non-printable characters while preserving Unicode.

    - strips invisible unicode control marks
    - NFKC normalizes
    - removes non-printables except common Unicode ranges
    """
    text = re.sub(r"[\u200f\u200e\u200d\u202c\u202d\u202e\u2066\u2067\u2068\u2069]", "", text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text


# --- Helper: sanitize string for folder/file paths ---
def _sanitize_for_path(name: str) -> str:
    """Sanitize a string for safe use in folder/file paths.
    Removes/normalizes path separators and common forbidden characters.
    """
    if not name:
        return "unknown"
    # Replace path separators and trim whitespace
    name = name.replace(os.sep, "_").replace("/", "_").strip()
    # Remove characters that are problematic on various filesystems
    name = re.sub(r"[\\:*?\"<>|]", "_", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def process_wikipedia_text(text: str, lang: str, **kwargs) -> List[Dict[str, str]]:
    """Split MediaWiki plaintext into simple header/paragraph blocks.

    The output remains the same structure you already use downstream: a list of
    dicts like {"header_1": ...} or {"paragraph": ...}.
    """
    ignore_headers = {
        "en": ["see also", "references", "external links"],
        "zh": ["参见", "参考资料", "参考文献", "外部链接"],
        "fr": ["voir aussi", "références"],
        "ru": ["см. также", "литература"],
        "ko": ["같이 보기", "참고 자료", "외부 링크"],
        "ja": ["関連項目", "参考文献", "外部リンク"],
        "he": ["מפיד", "הערות שוליים", "קישורים חיצוניים"],
        "bn": ["আরও দেখুন", "তথ্যসূত্র", "বহিঃসংযোগ"],
    }

    lines = text.split('\n')
    processed_paragraphs: List[Dict[str, str]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Header markup like == History ==, etc.
        header_match = re.match(r'^(=+)(.*?)\1$', line)
        if header_match:
            level = len(header_match.group(1)) - 1
            header_text = header_match.group(2).strip().lower()
            if lang in ignore_headers and header_text in ignore_headers[lang]:
                break  # stop at References/External links, etc.
            processed_paragraphs.append({f"header_{level}": header_text})
            continue

        # Long-ish paragraph lines
        if len(line) >= 6:
            line = clean_text(line)
            processed_paragraphs.append({"paragraph": line})

    return processed_paragraphs


def step_load_both_bios(lang_article_dict: Dict[str, Optional[str]], **kwargs) -> None:
    """Fetch, process, and save blocks for each language→title pair as .pkl.

    Saves to BIO_SAVE_DIR/<pillar>/<english_title>/<lang>.pkl so that you have
    an English-named topic folder, containing one file per language version.
    Falls back to BIO_SAVE_DIR/<pillar>/qid_<QID>/ if English title is absent.
    """
    pillar = kwargs.get("pillar", "misc")
    qid = kwargs.get("qid")

    try:
        os.makedirs(BIO_SAVE_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not ensure BIO_SAVE_DIR exists: {e}")

    # Determine the English anchor folder name
    en_title = lang_article_dict.get("en")
    if not en_title:
        anchor = f"qid_{qid}" if qid else "unknown"
    else:
        anchor = _sanitize_for_path(en_title)

    topic_dir = os.path.join(BIO_SAVE_DIR, _sanitize_for_path(str(pillar)), anchor)
    os.makedirs(topic_dir, exist_ok=True)

    failed = []
    progress = tqdm.tqdm(total=len(lang_article_dict))
    for lang, article_title in lang_article_dict.items():
        if not article_title:
            logger.error(f"Empty title for lang={lang}; skipping")
            failed.append((lang, article_title))
            progress.update(1)
            continue

        logger.info(f"Retrieving content blocks for {lang} {article_title}")
        text = get_wikipedia_text(article_title, lang)
        if not text:
            logger.error(f"Empty extract for {lang} {article_title}")
            failed.append((lang, article_title))
            progress.update(1)
            continue

        blocks = process_wikipedia_text(text, lang)
        # Save one pickle per language inside the English-named folder
        out_path = os.path.join(topic_dir, f"{lang}.pkl")
        with open(out_path, 'wb') as f:
            dill.dump(blocks, f)
        logger.info(f"Saved {len(blocks)} blocks → {out_path}")
        progress.update(1)

    progress.close()
    logger.info(f"Failed articles: {failed}")


# ----------------------------
# Existing interactive command (kept intact)
# ----------------------------
@click.command()
def scrape_bios():
    """Existing interactive path: ask for a target language, iterate selected_topics.

    This command is left intact for backwards-compatibility with your current
    experiments. It fetches English + target lang titles via sitelinks and saves
    .pkl blocks using the same processing function.
    """
    tgt_lang = input("Enter the target language code you would like to scrape for these topics: ")
    lang = "en"
    en_tgt_title_pairs = []

    for topic in selected_topics:
        print(f"Processing topic: {topic}")
        step_dict = OrderedDict()
        step_dict['get_wikidata_id'] = SingletonStep(get_wikidata_id, {
            'topic': topic,
            'lang': lang,
            'version': '001'
        })
        step_dict['get_interlanguage_links'] = SingletonStep(get_interlanguage_links, {
            'wikidata_id': 'get_wikidata_id',
            'version': '001'
        })
        step_dict['extract_en_article_title_from_url'] = SingletonStep(extract_article_title_from_urls, {
            'urls': 'get_interlanguage_links',
            'lang': 'en',
            'version': '001'
        })
        step_dict['extract_tgt_article_title_from_url'] = SingletonStep(extract_article_title_from_urls, {
            'urls': 'get_interlanguage_links',
            'lang': tgt_lang,
            'version': '001'
        })
        step_dict['lang_article_dict'] = SingletonStep(make_lang_article_dict, {
            'en_article_title': 'extract_en_article_title_from_url',
            'en_lang': 'en',
            'tgt_article_title': 'extract_tgt_article_title_from_url',
            'tgt_lang': tgt_lang,
            'version': '001'
        })
        step_dict['step_load_both_bios'] = SingletonStep(step_load_both_bios, {
            'lang_article_dict': 'lang_article_dict',
            'version': '001'
        })
        metadata = conduct(
            os.path.join(SCRATCH_DIR, "bio_scrape_cache"),
            step_dict,
            f"new_scrape_en_{tgt_lang}_bios_{topic.replace(' ', '_')}"
        )
        cache_path_src_title = metadata[2][1]['cache_path']
        cache_path_tgt_title = metadata[3][1]['cache_path']

        src_title = examine_cache(cache_path_src_title)
        decoded_src_title = urllib.parse.unquote(src_title)
        tgt_title = examine_cache(cache_path_tgt_title)
        decoded_tgt_title = urllib.parse.unquote(tgt_title)
        en_tgt_title_pairs.append((decoded_src_title, decoded_tgt_title))

    output_path = f"packages/scraped_titles_{tgt_lang}.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated file containing (English, Target language) topic tuples\n\n")
        f.write("en_tgt_title_pairs = [\n")
        for en, tgt in en_tgt_title_pairs:
            f.write(f"    ({repr(en)}, {repr(tgt)}),\n")
        f.write("]\n")

    print(f"\n✅ Saved {len(en_tgt_title_pairs)} topic pairs to {output_path}")


# ----------------------------
# YAML-driven pipeline (NEW): languages + topics with optional SPARQL
# ----------------------------

def load_langs_from_yaml(path: str) -> List[str]:
    """Load a list of language codes from YAML: {langs: [en, fr, ...]}."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("langs", [])


def load_pillars_from_yaml(path: str) -> List[dict]:
    """Load pillars from YAML: each with name, optional seeds, optional class_qid, optional limit_per_class."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("pillars", [])


def fetch_sitelinks_for_qid(qid: str) -> Dict[str, dict]:
    """Fetch the full sitelinks block for a QID (all languages)."""
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbgetentities", "ids": qid, "props": "sitelinks/urls", "format": "json"}
    r = _session().get(url, params=params, timeout=30)
    r.raise_for_status()
    ent = r.json().get("entities", {}).get(qid, {})
    return ent.get("sitelinks", {})


#
# Accept **kwargs because Flowmason includes bookkeeping params like 'version' in the step kwargs.
def titles_for_langs_from_qid(qid: str, langs: List[str], **kwargs) -> OrderedDict:
    """Build a {lang: title or None} mapping for requested languages using sitelinks URLs."""
    sitelinks = fetch_sitelinks_for_qid(qid)
    out: OrderedDict[str, Optional[str]] = OrderedDict()
    for lang in langs:
        key = f"{lang}wiki"
        title = None
        if key in sitelinks:
            url = sitelinks[key].get("url", "")
            m = re.search(r"/wiki/([^#?]*)", url)
            if m:
                title = urllib.parse.unquote(m.group(1)).replace("_", " ")
        out[lang] = title
    return out


def sparql_instances_of(class_qid: str, limit: Optional[int] = None) -> List[str]:
    """Enumerate QIDs that are instances of a given Wikidata class via SPARQL.

    If `limit` is provided, we cap the result size for quick tests.
    """
    WD_SPARQL = "https://query.wikidata.org/sparql"
    lim = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
    SELECT ?item WHERE {{
      ?item wdt:P31 wd:{class_qid} .
    }} {lim}
    """
    r = _session().get(WD_SPARQL, params={"query": query, "format": "json"}, timeout=60)
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    return [b["item"]["value"].split("/")[-1] for b in rows]


@click.command()
@click.option("--topics-file", default="config/topics.yaml", show_default=True)
@click.option("--languages-file", default="config/languages.yaml", show_default=True)
def scrape_topics_yaml(topics_file: str, languages_file: str):
    """YAML-driven scraping: for each pillar, get QIDs from `seeds` or SPARQL, then save .pkl per (title, lang).

    This reuses `step_load_both_bios` so the on-disk format stays identical to
    your current pipeline (blocks list pickled per article/language).
    """
    langs = load_langs_from_yaml(languages_file)
    pillars = load_pillars_from_yaml(topics_file)

    if not langs:
        raise click.ClickException("No languages found in languages YAML.")
    if not pillars:
        raise click.ClickException("No pillars found in topics YAML.")

    summary = []
    for pillar in pillars:
        name = pillar.get("name")
        seeds = pillar.get("seeds", [])
        class_qid = pillar.get("class_qid")
        limit = pillar.get("limit_per_class")

        # Choose QIDs: prefer explicit seeds; otherwise use SPARQL
        if seeds:
            qids = seeds
        elif class_qid:
            qids = sparql_instances_of(class_qid, limit=limit)
        else:
            logger.warning(f"pillar '{name}' has neither seeds nor class_qid; skipping")
            continue

        logger.info(f"Processing pillar={name} with {len(qids)} QIDs; langs={langs}")
        pbar = tqdm.tqdm(total=len(qids), desc=f"{name}")

        for qid in qids:
            # Build {lang: title} mapping for the requested languages
            lang_article_dict = titles_for_langs_from_qid(qid, langs)
            if not any(lang_article_dict.values()):
                logger.warning(f"{qid}: no titles available in requested langs; skipping")
                pbar.update(1)
                continue

            # Reuse flowmason to stay consistent with your cache/metadata style
            step_dict = OrderedDict()
            step_dict['lang_article_dict'] = SingletonStep(
                titles_for_langs_from_qid,
                {
                    'qid': qid,           # include qid so cache differs per item
                    'langs': tuple(langs),
                    'version': f'001_{qid}'  # version also includes qid for extra safety
                }
            )
            step_dict['step_load_both_bios'] = SingletonStep(
                step_load_both_bios,
                {
                    'lang_article_dict': 'lang_article_dict',
                    'pillar': name,   # for BIO_SAVE_DIR/<pillar>/...
                    'qid': qid,       # fallback folder name if no English title
                    'version': '001'
                }
            )
            conduct(
                os.path.join(SCRATCH_DIR, "yaml_scrape_cache"),
                step_dict,
                f"{name}_{qid}"
            )
            pbar.update(1)

        pbar.close()
        summary.append((name, len(qids)))

    logger.info(f"Done. Pillars processed: {summary}")


@click.group()
def main():
    """CLI group for scraping commands (interactive and YAML-driven)."""
    pass


# Register commands
main.add_command(scrape_bios)           # existing path
main.add_command(scrape_topics_yaml)    # new YAML-driven path


if __name__ == '__main__':
    main()
    # bio_frame = load_artifact_with_step_name(metadata, "step_load_pairs_common")
    # bio_ids = load_bios()
    # extract_bios_en_fr(bio_ids)