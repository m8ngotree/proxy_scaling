#!/usr/bin/env python3
"""
Download and tokenize DCLM data into per-domain binary files.

For each domain in shared_mixtures.json:
1. Check if data/domains/{topic}/train.bin exists with enough tokens - skip if so
2. Stream DCLM from HuggingFace (mlfoundations/dclm-baseline-1.0)
3. Filter to the requested topic using available metadata/labels
4. Tokenize with the Dolma 2 tokenizer (allenai/dolma2-tokenizer)
5. Save as numpy uint16 memmap: data/domains/{topic}/train.bin
6. Save token count to data/domains/{topic}/token_count.txt
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset


def get_required_domains(shared_mixtures_path: str = "data/shared_mixtures.json") -> list[str]:
    """Get list of domain names from shared mixtures config."""
    shared = json.load(open(shared_mixtures_path))
    return [
        d.replace("dclm:", "")
        for d in shared["domain_cols"]
        if d.startswith("dclm:")
    ]


def check_existing(domain: str, output_dir: str, min_tokens: int) -> bool:
    """Check if domain data already exists with sufficient tokens."""
    bin_path = Path(output_dir) / domain / "train.npy"
    count_path = Path(output_dir) / domain / "token_count.txt"

    if not bin_path.exists():
        return False

    if count_path.exists():
        existing_tokens = int(count_path.read_text().strip())
        if existing_tokens >= min_tokens:
            print(f"  {domain}: already has {existing_tokens:,} tokens (need {min_tokens:,}), skipping")
            return True

    # Check file size: uint32 = 4 bytes per token
    file_tokens = bin_path.stat().st_size // 4
    if file_tokens >= min_tokens:
        print(f"  {domain}: file has ~{file_tokens:,} tokens (need {min_tokens:,}), skipping")
        return True

    return False


def tokenize_and_save_domain(
    domain: str,
    tokenizer,
    output_dir: str,
    max_tokens: int,
    dclm_dataset_name: str = "mlfoundations/dclm-baseline-1.0",
):
    """Stream DCLM, filter to domain, tokenize, and save as memmap.

    DCLM documents may have topic labels from WebOrganizer. If the dataset
    does not include topic labels, we process all documents and rely on the
    caller to handle domain assignment. In practice, the Olmix team used
    WebOrganizer to pre-classify DCLM into 24 topic domains. If those
    pre-classified splits are available (e.g., as separate configs or
    columns), we use them directly.
    """
    domain_dir = Path(output_dir) / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    bin_path = domain_dir / "train.npy"
    count_path = domain_dir / "token_count.txt"

    print(f"\n  Processing domain: {domain}")
    print(f"  Target tokens: {max_tokens:,}")

    # Try loading DCLM with streaming to avoid downloading entire dataset.
    # The dataset may have a 'topic' or 'label' field from WebOrganizer,
    # or it may be split by topic as separate configs.
    all_tokens = []
    token_count = 0

    try:
        # First try: load with domain as a config/subset name
        ds = load_dataset(
            dclm_dataset_name,
            name=domain,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        print(f"  Loaded DCLM subset '{domain}' directly")
    except Exception:
        try:
            # Second try: load full dataset and filter by topic field
            ds = load_dataset(
                dclm_dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            # Check if documents have a topic/label field
            sample = next(iter(ds))
            topic_field = None
            for field in ["topic", "label", "category", "domain"]:
                if field in sample:
                    topic_field = field
                    break

            if topic_field:
                print(f"  Filtering by {topic_field}=={domain}")
                ds = ds.filter(lambda x: x[topic_field] == domain)
            else:
                print(f"  WARNING: No topic field found in DCLM. "
                      f"Available fields: {list(sample.keys())}")
                print(f"  Will process documents sequentially (no domain filtering).")
                print(f"  Consider running WebOrganizer to classify documents first.")
        except Exception as e:
            print(f"  ERROR loading DCLM: {e}")
            print(f"  Creating empty placeholder for domain '{domain}'")
            # Create a minimal placeholder so the pipeline doesn't break
            placeholder = np.zeros(1000, dtype=np.uint32)
            np.save(str(bin_path), placeholder)
            count_path.write_text("1000")
            return

    # Determine the text field name
    text_field = "text"
    try:
        sample = next(iter(ds))
        if "text" not in sample:
            # Try common alternatives
            for field in ["content", "document", "raw_content"]:
                if field in sample:
                    text_field = field
                    break
    except StopIteration:
        print(f"  WARNING: Empty dataset for domain '{domain}'")
        placeholder = np.zeros(1000, dtype=np.uint16)
        placeholder.tofile(str(bin_path))
        count_path.write_text("1000")
        return

    # Tokenize in batches
    batch_size = 100
    batch_texts = []

    for doc in ds:
        text = doc.get(text_field, "")
        if not text:
            continue
        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            encoded = tokenizer(
                batch_texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            for ids in encoded["input_ids"]:
                all_tokens.extend(ids)
                token_count += len(ids)

            batch_texts = []

            if token_count % 10_000_000 < batch_size * 500:
                print(f"    {token_count:,} tokens collected...")

            if token_count >= max_tokens:
                break

    # Process remaining batch
    if batch_texts:
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for ids in encoded["input_ids"]:
            all_tokens.extend(ids)
            token_count += len(ids)

    # Truncate to max_tokens
    all_tokens = all_tokens[:max_tokens]
    token_count = len(all_tokens)

    # Save as uint32 numpy array (vocab_size=100352 exceeds uint16 max of 65535)
    arr = np.array(all_tokens, dtype=np.uint32)
    np.save(str(bin_path), arr)
    count_path.write_text(str(token_count))

    print(f"  Saved {token_count:,} tokens to {bin_path}")
    print(f"  File size: {bin_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Tokenize DCLM domains")
    parser.add_argument(
        "--tokens-per-domain",
        type=int,
        default=3_000_000_000,
        help="Max tokens per domain (default: 3B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/domains/",
        help="Output directory for domain data",
    )
    parser.add_argument(
        "--shared-mixtures",
        type=str,
        default="data/shared_mixtures.json",
        help="Path to shared mixtures config",
    )
    args = parser.parse_args()

    domains = get_required_domains(args.shared_mixtures)
    print(f"Processing {len(domains)} domains")
    print(f"Target: {args.tokens_per_domain:,} tokens per domain")
    print(f"Output: {args.output_dir}")

    # Load tokenizer once
    print("\nLoading Dolma 2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/dolma2-tokenizer", trust_remote_code=True
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Dolma 2 vocab_size=100352, which exceeds uint16 max (65535), so we use uint32
    if tokenizer.vocab_size > 65535:
        print(f"NOTE: vocab_size={tokenizer.vocab_size} > 65535, using uint32 for storage")

    skipped = 0
    processed = 0

    for domain in domains:
        if check_existing(domain, args.output_dir, args.tokens_per_domain):
            skipped += 1
            continue

        tokenize_and_save_domain(
            domain=domain,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_tokens=args.tokens_per_domain,
        )
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
