import json
import os
import sys
import urllib.parse
import dill
import importlib
import pandas as pd
import ast
import requests
from datetime import datetime
from deep_translator import GoogleTranslator
import openai
import loguru
from tqdm import tqdm

logger = loguru.logger

# ---------------------------
# 1) GLOBALS & CONFIG
# ---------------------------
BIO_SAVE_DIR = "scratch/wiki_articles"
LANG_CODE_MAPPING_HEADER = {
    "en": "en",
    "fr": "fr",
    "ru": "ru",
    "zh": "zh-TW"
}
LANG_CODE_MAPPING = {
    "en": "English",
    "fr": "French",
    "ru": "Russian",
    "zh": "Chinese"
}

class BioFilenotFoundError(Exception):
    """Custom exception for missing bio file."""
    pass

# Configure your Azure OpenAI client
client = openai.AzureOpenAI(
    api_key=os.getenv("THE_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("URL_ENDPOINT"),
)

SRC_LANGUAGE_FILTER = 'en'  # The primary language to skip in final JSON if desired


# ---------------------------
# 2) EXTRACTION & PARSING
# ---------------------------
def extract_values(data, target_names):
    """
    Recursively traverse the JSON object to extract 'values' for each 'name' in target_names.
    Returns { target_name: [list_of_values, ...], ... }.
    """
    extracted_values = {}

    def traverse(obj):
        if isinstance(obj, dict):
            # If this dict has a 'name' matching our targets, gather its 'values'
            if 'name' in obj and obj['name'] in target_names and 'values' in obj:
                extracted_values[obj['name']] = obj['values']
            for _, value in obj.items():
                traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)

    traverse(data)
    return extracted_values


def clean_src_context_nested_values(df):
    """
    If a row's 'src_context' is a dict with a 'values' key, replace it with that list of values.
    """
    if 'src_context' in df.columns:
        df['src_context'] = df['src_context'].apply(
            lambda x: x['values'] if isinstance(x, dict) and 'values' in x else x
        )
    return df


def extract_nested_values(column_data):
    """
    'tgt_contexts' can be a dict { 'values': [ { 'values': [] }, ... ] }.
    Flatten out only 'values' from each item in the list.
    """
    if not isinstance(column_data, dict):
        return column_data  # If it's not the expected dict shape, just return as is

    values = column_data.get('values', [])
    new_values = []
    for val in values:
        # Each val should be a dict with 'values'
        inner_list = val.get('values', [])
        new_values.extend(inner_list)
    return new_values

# Function to process a single JSON file
def process_json_file(file_path, target_names):
    with open(file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Extract values
    extracted_data = extract_values(json_data, target_names)

    # Convert the extracted data into a structured column-wise format
    max_length = max(len(v) for v in extracted_data.values()) if extracted_data else 0

    # Ensure all columns have the same number of rows by padding with None
    for key in extracted_data:
        while len(extracted_data[key]) < max_length:
            extracted_data[key].append(None)

    # Convert to DataFrame
    df_structured = pd.DataFrame(extracted_data)

    # Function to extract only the 'values' from nested objects in specific columns
    df_structured = clean_src_context_nested_values(df_structured)
    # Apply the function to the 'tgt_contexts' column
    if 'tgt_contexts' in df_structured.columns:
        df_structured['tgt_contexts'] = df_structured['tgt_contexts'].apply(extract_nested_values)
    if 'tgt_fact_aligned_sentences' in df_structured.columns:
        df_structured['tgt_fact_aligned_sentences'] = df_structured['tgt_fact_aligned_sentences'].apply(extract_nested_values)


    return df_structured

def process_single_json_file(directory_path, filename, target_names, output_csv_path):
    """
    1) process_json_file to get a DataFrame
    2) Save to CSV for inspection (optional).
    3) Return the DataFrame.
    """
    file_path = os.path.join(directory_path, filename)
    df = process_json_file(file_path, target_names)
    if df.empty:
        return df

    df["source_file"] = filename
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return df

def retrieve_title(topic, tgt_lang):
    """
    Dynamically import en_tgt_title_pairs from packages.scraped_titles_{tgt_lang}
    and return the target title that matches `topic`.
    """
    module_name = f"packages.scraped_titles_{tgt_lang}"
    mod = importlib.import_module(module_name)  # import packages.scraped_titles_ru, for example
    en_tgt_title_pairs = mod.en_tgt_title_pairs

    for src_topic, tgt_topic_val in en_tgt_title_pairs:
        if src_topic == topic:
            return tgt_topic_val
    return None

# Usage example:
# Suppose en_tgt_title_pairs = [('apple', 'яблоко'), ('banana', 'банан')]
# retrieve_title('apple', 'ru', en_tgt_title_pairs)  # returns 'яблоко'


# ---------------------------
# 3) BIO FILE RETRIEVAL
# ---------------------------
def step_retrieve_prescraped_en_content_blocks(en_bio_id, save_dir=BIO_SAVE_DIR):
    path = os.path.join(save_dir, f"{en_bio_id}_en.pkl")
    if not os.path.exists(path):
        raise BioFilenotFoundError(f"Could not find the prescraped bio file for {en_bio_id}")
    with open(path, 'rb') as f:
        return dill.load(f)


def step_retrieve_prescraped_tgt_content_blocks(tgt_bio_id, tgt_lang, save_dir=BIO_SAVE_DIR):
    path = os.path.join(save_dir, f"{tgt_bio_id}_{tgt_lang}.pkl")
    print(path)
    if not os.path.exists(path):
        raise BioFilenotFoundError(f"Could not find the prescraped bio file for {tgt_bio_id}")
    logger.info(f"Retrieving {tgt_bio_id}_{tgt_lang} from {save_dir}")
    with open(path, 'rb') as f:
        return dill.load(f)


# ---------------------------
# 4) PARAGRAPH & HEADER HANDLING
# ---------------------------
def process_paragraphs_with_headers(data):
    """
    Convert a list of dicts (with 'header_x' or 'paragraph') into a list of paragraphs,
    each referencing the most recent headers at various levels.
    """
    processed_data = []
    current_headers = {}  # level -> text
    paragraph_index = 0

    for item in data:
        # Detect a header, e.g., 'header_1'
        header_match = next((k for k in item.keys() if k.startswith("header")), None)
        if header_match:
            level_str = header_match.split("_")[1] if "_" in header_match else "1"
            level = int(level_str) if level_str.isdigit() else 1
            header_text = item[header_match]
            current_headers[level] = header_text
            # Remove deeper headers if a higher-level one is set
            current_headers = {k: v for k, v in current_headers.items() if k <= level}

        elif "paragraph" in item:
            paragraph_entry = {
                "index": paragraph_index,
                "paragraph": item["paragraph"],
            }
            # Include all relevant headers
            for lvl in sorted(current_headers.keys()):
                paragraph_entry[f"header_{lvl}"] = current_headers[lvl]

            processed_data.append(paragraph_entry)
            paragraph_index += 1

    return processed_data


def get_header_by_paragraph_index(paragraph_idx, data, lvl):
    for p in data:
        if p["index"] == paragraph_idx:
            return p.get(lvl, None)
    return None


# ---------------------------
# 5) TRANSLATION UTILITIES
# ---------------------------
def translate_headers(df, src_lang, tgt_lang):
    """
    Translate header_1 and header_2 from src_lang to tgt_lang using GoogleTranslator.
    """
    translator = GoogleTranslator(source=src_lang, target=tgt_lang)

    def safe_translate(text):
        if pd.notna(text) and text not in ["None", None]:
            return translator.translate(text)
        return text

    # You could vectorize this, but for clarity we do a row-based approach:
    rows = df.to_dict('records')
    header_1_translated = []
    header_2_translated = []

    for row in tqdm(rows, desc="Translating headers"):
        h1 = safe_translate(row.get("header_1"))
        h2 = safe_translate(row.get("header_2"))
        header_1_translated.append(h1)
        header_2_translated.append(h2)

    df["header_1_translated"] = header_1_translated
    df["header_2_translated"] = header_2_translated
    return df


def translate_facts(df, src_lang, tgt_lang):
    """
    Translate the `fact` column with GPT. If there's no 'fact' column, does nothing.
    """
    if "fact" not in df.columns:
        return df

    def gpt_translate(text):
        if pd.notna(text) and text not in ["None", None]:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Translate the following content: '{text}' "
                                f"from {src_lang} to {tgt_lang}. "
                                "Return only the translation."
                            )
                        }
                    ]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Translation Error: {e}"
        return text

    rows = df.to_dict('records')
    fact_translated = []

    for row in tqdm(rows, desc="Translating facts"):
        translated = gpt_translate(row.get("fact"))
        # Try to unescape if it looks like a JSON-encoded string
        if isinstance(translated, str):
            try:
                # If translated was something like "\"Galof rice...\""
                # json.loads() will convert it to a clean string
                possibly_unescaped = json.loads(translated)
                if isinstance(possibly_unescaped, str):
                    translated = possibly_unescaped
            except json.JSONDecodeError:
                # Not a JSON-encoded string, so just keep the original
                pass

        fact_translated.append(translated)

    df["fact_translated"] = fact_translated
    return df


# --- Helper utilities for final JSON ---
def safe_value(value):
    """ Convert NaN to None so JSON doesn't store 'NaN'. """
    return None if pd.isna(value) else value

import ast
import numpy as np
import pandas as pd

def parse_as_list(value):
    """
    Convert various data types into a Python list:
      - If value is a NumPy array or Pandas Series, convert it to a list.
      - If it's None or float('NaN'), return None.
      - If it's already a list, return it as-is.
      - If it's a string that looks like a list, parse with ast.literal_eval.
      - Otherwise, make it a single-element list of the stringified value.
    """

    # 1) Handle arrays/Series by converting them into Python lists
    if isinstance(value, (pd.Series, np.ndarray)):
        value = value.tolist()

    # 2) If the value is None, or if it's a float that's NaN, return None
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    # 3) If it’s already a Python list, we're good
    if isinstance(value, list):
        return value

    # 4) If it's a string, attempt to parse it as a Python literal
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
        except (SyntaxError, ValueError):
            # If it's not a valid Python literal, just wrap the raw string in a list
            return [value]

    # 5) Fallback: wrap the object (int, dict, etc.) in a single-element list
    return [str(value)]

# --- Wikidata lookups (optional) ---
def get_wikidata_id(article_title, lang='en'):
    """Get the Wikidata QID for a given Wikipedia article title."""
    url = (
        f"https://www.wikidata.org/w/api.php?"
        f"action=wbgetentities&sites={lang}wiki&titles={article_title}"
        f"&props=info&format=json"
    )
    resp = requests.get(url)
    data = resp.json()
    entities = data.get("entities", {})
    if entities:
        return list(entities.keys())[0]  # e.g. "Q12345"
    return None

def get_interlanguage_links(wikidata_id):
    url = (
        f"https://www.wikidata.org/w/api.php?"
        f"action=wbgetentities&ids={wikidata_id}&props=sitelinks/urls&format=json"
    )
    resp = requests.get(url)
    data = resp.json()
    sitelinks = data.get("entities", {}).get(wikidata_id, {}).get("sitelinks", {})
    return {site: info['url'] for site, info in sitelinks.items()}

def get_tgt_wiki_link(name, src_lang, tgt_lang):
    """Resolve a target-language Wikipedia link from Wikidata."""
    wikidata_id = get_wikidata_id(name, src_lang)
    if not wikidata_id:
        logger.warning(f"No Wikidata ID found for: {name}")
        return name  # fallback

    tgt_lang_wiki = f"{tgt_lang}wiki"
    interlinks = get_interlanguage_links(wikidata_id)
    if tgt_lang_wiki in interlinks:
        return interlinks[tgt_lang_wiki]
    return name


def df_to_nested_json(df):
    """
    Build a nested dict with structure:
      { person_name: {
          languages: {
             lang: {
               headers: {
                 header_1_text: {
                   'entries': [ { details }, ... ]
                 }
               }
             }
          }
        }
      }
    Skips rows with language == SRC_LANGUAGE_FILTER if desired.
    """
    if df.empty:
        return {}

    nested_dict = {}
    # Use the first 'person_name' as a fallback reference
    person = df['person_name'].iloc[0]

    # Collect all languages encountered that are not the source language
    non_src_langs = df.loc[df['language'] != SRC_LANGUAGE_FILTER, 'language'].unique()

    # Pre-fetch the wiki links for each language
    # (Use dict comprehension if you want to do it once per person!)
    languages_dict = {}
    for lang in non_src_langs:
        languages_dict[lang] = get_tgt_wiki_link(person, SRC_LANGUAGE_FILTER, lang)

    for _, row in df.iterrows():
        # Possibly skip rows with language == SRC_LANGUAGE_FILTER
        if row['language'] == SRC_LANGUAGE_FILTER:
            continue

        person = row['person_name']
        language = row['language']
        tgt_wiki_link = languages_dict.get(language, row['person_name'])

        header_1_val = safe_value(row.get('header_1')) or "General description"

        # Make sure nested dict structure exists
        nested_dict.setdefault(person, {}).setdefault('languages', {})\
                   .setdefault(language, {}).setdefault('headers', {})\
                   .setdefault(header_1_val, {'entries': []})

        src_context = parse_as_list(row.get('src_context'))
        tgt_sentences = parse_as_list(row.get('tgt_fact_aligned_sentences'))

        entry = {
            'header_2': {
                'original': safe_value(row.get('header_2')),
                'translated': safe_value(row.get('header_2_translated'))
            },
            'header_1': {
                'original': header_1_val,
                'translated': safe_value(row.get('header_1_translated'))
            },
            'fact': {
                'original': safe_value(row.get('fact')),
                'translated': safe_value(row.get('fact_translated')),
                'fact_aligned_sentence': safe_value(row.get('fact_aligned_sentence')),
                'wiki_link': tgt_wiki_link
            },
            'src_context': src_context,
            'tgt_fact_aligned_sentences': tgt_sentences
        }

        nested_dict[person]['languages'][language]['headers'][header_1_val]['entries'].append(entry)

    return nested_dict


# ---------------------------
# 6) SAMPLING HELPER
# ---------------------------
def weighted_sampling_by_header(df, header_column="header_1", sample_size=15):
    """
    Example of weighting by the frequency of header_1 to prefer more/less common headers.
    Adjust logic as needed. If sample_size > len(df), it will return the entire DF.
    """
    if df.empty or sample_size <= 0:
        return df

    freq = df[header_column].value_counts(normalize=True, dropna=False)
    weights = df[header_column].map(freq)
    # We clamp sample_size so we don't overshoot
    sample_size = min(sample_size, len(df))
    return df.sample(n=sample_size, weights=weights, random_state=42)


# ---------------------------
# 7) MAIN WORKFLOW
# ---------------------------
def main():
    """
    High-level steps:
      1) For each (en_title, tgt_title) in en_tgt_title_pairs, load & parse JSON -> DF.
      2) Retrieve paragraph blocks, attach header info.
      3) Translate, filter, sample.
      4) Collect all rows, convert to nested JSON, write out.
    """
    # Example placeholders
    TARGET_LANGUAGES = ['ru', 'fr', 'zh']
    json_directory = "scratch/ethics_annotation_save/wikigap_data"
    output_csv = "wikigap_data_temp.csv"
    target_names = {
        'fact',
        'fact_aligned_sentence',
        'src_context',
        'person_name',
        'tgt_contexts',
        'tgt_fact_aligned_sentences',
        'intersection_label',
        'language',
        'paragraph_index'
    }
    from wikigap_topics_scrape import selected_topics
    for topic in selected_topics:
        today = '2025-03-24'
        sample_size = 20  # Example sample size
        all_dfs = []
        for tgt_lang in TARGET_LANGUAGES:
            tgt_title = retrieve_title(topic, tgt_lang)
            en_title = topic
            file_name = f"annotation_{today}_{en_title}_{tgt_lang}.json"
            df = process_single_json_file(json_directory, file_name, target_names, output_csv)
            if df.empty:
                logger.warning(f"No data extracted from {file_name}, skipping.")
                continue

            try:
                en_blocks = step_retrieve_prescraped_en_content_blocks(en_title)
                tgt_blocks = step_retrieve_prescraped_tgt_content_blocks(tgt_title, tgt_lang)
            except BioFilenotFoundError as e:
                logger.warning(e)
                continue

            # Convert paragraph blocks to a simpler list with indices
            processed_en_blocks = process_paragraphs_with_headers(en_blocks)
            processed_tgt_blocks = process_paragraphs_with_headers(tgt_blocks)

            # We only care about non-EN rows in this example
            df_tgt = df[df["language"] != "en"].copy()
            if df_tgt.empty:
                continue

            # Attach header_1 and header_2 by paragraph_index
            df_tgt["header_1"] = df_tgt["paragraph_index"].apply(
                get_header_by_paragraph_index, args=(processed_tgt_blocks, "header_1")
            )
            df_tgt["header_2"] = df_tgt["paragraph_index"].apply(
                get_header_by_paragraph_index, args=(processed_tgt_blocks, "header_2")
            )

            # df_filtered = df_tgt[df_tgt['intersection_label'] == 'no']
            # # Weighted sampling if desired
            # df_tgt_sampled = weighted_sampling_by_header(df_filtered, header_column="header_1", sample_size=sample_size)

            # # Convert NaNs to None to avoid translator issues
            # df_tgt_sampled = df_tgt_sampled.where(pd.notna(df_tgt_sampled), None)

            # # Translate headers, facts from RU -> EN
            # df_tgt_sampled = translate_headers(df_tgt_sampled, LANG_CODE_MAPPING_HEADER[tgt_lang], "en")
            # df_tgt_sampled = translate_facts(df_tgt_sampled, LANG_CODE_MAPPING[tgt_lang], "English")

            # if not df_tgt_sampled.empty:
            #     all_dfs.append(df_tgt_sampled)
            df_filtered = df_tgt[df_tgt['intersection_label'] == 'no']
            df_filtered = df_filtered.where(pd.notna(df_filtered), None)

            # Translate headers and facts
            df_translated = translate_headers(df_filtered, LANG_CODE_MAPPING_HEADER[tgt_lang], "en")
            df_translated = translate_facts(df_translated, LANG_CODE_MAPPING[tgt_lang], "English")

            if not df_translated.empty:
                all_dfs.append(df_translated)

        # Combine everything into one DataFrame
        if not all_dfs:
            logger.warning("No data to combine. Exiting.")
            return

        df_merged = pd.concat(all_dfs, ignore_index=True)
        df_merged = df_merged.where(pd.notna(df_merged), None)
        print(df_merged)
        # Convert to nested JSON
        nested_json = df_to_nested_json(df_merged)

        # Save final JSON
        output_json = f"scratch/ethics_annotation_save/wikigap_data/json/{topic}.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(nested_json, f, indent=4, ensure_ascii=False)
        print(f"JSON file saved successfully: {output_json}")


# Pythonic entry point
if __name__ == "__main__":
    main()