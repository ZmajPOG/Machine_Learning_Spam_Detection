import os
import re
import glob
import email
import pandas as pd
from email import policy
from email.parser import BytesParser

def _read_text_file(path: str) -> str:
    try:
        with open(path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    try:
                        parts.append(part.get_content())
                    except Exception:
                        continue
            return "\n".join([str(p) for p in parts])
        else:
            return str(msg.get_content())
    except Exception:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""

def load_spamassassin(base_dir: str) -> pd.DataFrame:
    ham_dirs  = ["ham", "ham2", "easy_ham", "easy_ham_2", "hard_ham"]
    spam_dirs = ["spam", "spam_2"]

    rows = []
    for name, label in [(d, "ham") for d in ham_dirs] + [(d, "spam") for d in spam_dirs]:
        folder = os.path.join(base_dir, name)
        if not os.path.isdir(folder):
            continue
        for file in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
            if os.path.isfile(file):
                text = _read_text_file(file)
                if text.strip():
                    rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)

def load_enron(base_dir: str, max_files: int = None) -> pd.DataFrame:
    rows, count = [], 0
    for file in glob.glob(os.path.join(base_dir, "**", "*"), recursive=True):
        if os.path.isfile(file):
            text = _read_text_file(file)
            if text.strip():
                rows.append({"text": text, "label": "ham"})
                count += 1
                if max_files and count >= max_files:
                    break
    return pd.DataFrame(rows)

def basic_clean(text: str) -> str:
    text = re.sub(r"\r\n|\r|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def prepare_dataset(enron_dir: str, sa_dir: str, out_csv: str, sample_size: int = None) -> pd.DataFrame:
    df_enron = load_enron(enron_dir)
    df_sa = load_spamassassin(sa_dir)
    df = pd.concat([df_enron, df_sa], ignore_index=True)
    df["text"] = df["text"].astype(str).map(basic_clean)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
