#!/usr/bin/env python3
"""
uniprot_download_and_clean.py

Two download methods:
  1) FTP/HTTP bulk files from UniProt (recommended for full releases)
  2) UniProt REST API queries (recommended for subsets / customized columns)

Cleaning performed:
  - parse TSV or FASTA
  - drop exact duplicate accessions
  - drop sequences flagged as 'fragment' (protein name contains 'fragment')
  - drop sequences with non-standard amino acids or internal stop codons ('*')
  - normalize organism names and accession column
  - output cleaned TSV and FASTA

References / notes:
  - UniProt provides FTP bulk downloads and a REST API for programmatic access.
    See UniProt documentation for FTP/regular downloads and the REST API.
"""

import os
import gzip
import shutil
import logging
from typing import Optional, Iterable
import requests
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------- Configuration ----------
# Example FTP URLs (mirror): these are standard UniProt paths for bulk releases.
# If you want a different mirror, change the base_url accordingly.
FTP_BASE = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/"
# Common files (gzipped) available on the FTP site:
EXAMPLE_SPROT_FASTA = FTP_BASE + "uniprot_sprot.fasta.gz"      # reviewed (Swiss-Prot)
EXAMPLE_TREMBL_FASTA = FTP_BASE + "uniprot_trembl.fasta.gz"    # unreviewed (TrEMBL)
EXAMPLE_UNIPROT_TSV = FTP_BASE + "uniprot_sprot.dat.gz"        # older flat format (example)
# ---------- End configuration ----------

def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024):
    """
    Download a (possibly large) file with streaming and resume-friendly behavior.
    """
    logging.info("Downloading %s -> %s", url, dest_path)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    logging.info("Download complete: %s", dest_path)


def maybe_gunzip(src_gz: str, dest: Optional[str] = None) -> str:
    """
    If src_gz ends with .gz, decompress to dest (or remove .gz if dest not provided).
    Returns path to decompressed file.
    """
    if not src_gz.endswith(".gz"):
        return src_gz
    if dest is None:
        dest = src_gz[:-3]
    logging.info("Decompressing %s -> %s", src_gz, dest)
    with gzip.open(src_gz, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


# ---------------- REST API helper ----------------
def download_uniprot_via_rest(query: str = "*",
                              fields: Optional[Iterable[str]] = None,
                              out_tsv: str = "uniprot_query_result.tsv",
                              batch_size: int = 500) -> str:
    """
    Download UniProt query results using the UniProt REST API in TSV format.
    - query: UniProt query string. "*" => all entries (beware large result!)
    - fields: list of return fields (see UniProt docs for field names).
    - out_tsv: output filename.
    Note: REST API has paging limits; this implementation fetches in pages.
    """
    if fields is None:
        fields = ["accession", "id", "protein_name", "organism_name", "sequence", "reviewed"]

    base = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "tsv",
        "fields": ",".join(fields),
        "size": str(batch_size)
    }
    logging.info("Querying UniProt REST API: query=%r, fields=%s", query, ",".join(fields))

    # We'll iterate pages using 'next' link if present.
    rows = []
    with requests.Session() as s:
        url = base
        while url:
            resp = s.get(url, params=params if url == base else None, stream=False)
            resp.raise_for_status()
            text = resp.text
            # First call: create file and write header
            if not rows and text:
                # Save first page header+rows
                with open(out_tsv, "w", encoding="utf-8") as fh:
                    fh.write(text)
                logging.info("Wrote initial page to %s", out_tsv)
            else:
                # append next page results but drop header line
                with open(out_tsv, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(text.splitlines()[1:]) + "\n")

            # Check for 'Link' header for pagination
            link = resp.headers.get("Link")
            next_url = None
            if link:
                # Link header looks like: <https://...&cursor=...>; rel="next"
                parts = link.split(",")
                for p in parts:
                    if 'rel="next"' in p:
                        # extract between <>
                        start = p.find("<")
                        end = p.find(">")
                        if start != -1 and end != -1:
                            next_url = p[start+1:end]
            if next_url:
                logging.info("Following next page: %s", next_url)
                url = next_url
                params = None  # subsequent pages already encoded in next_url
            else:
                url = None
    logging.info("REST download finished; saved to %s", out_tsv)
    return out_tsv


# ---------------- Cleaning routines ----------------
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")  # canonical 20 amino acids (uppercase)


def is_valid_protein_sequence(seq: str, allow_x: bool = False) -> bool:
    """
    Valid sequence test:
      - No internal '*' (stop codon)
      - Only standard amino acids (optionally allowing X)
    """
    if "*" in seq:
        return False
    seq_set = set(seq.upper())
    if allow_x:
        seq_set -= {"X"}
    return seq_set.issubset(STANDARD_AA)


def clean_uniprot_tsv(in_tsv: str,
                      accession_col: str = "Entry",
                      protein_name_col: str = "Protein names",
                      sequence_col: str = "Sequence",
                      reviewed_col: Optional[str] = None,
                      drop_fragments: bool = True,
                      allow_x: bool = False) -> pd.DataFrame:
    """
    Load a UniProt TSV file into a DataFrame and apply cleaning:
      - rename standard accession column to 'accession'
      - drop duplicates, keeping first
      - remove entries with 'fragment' in the protein name if drop_fragments True
      - remove invalid sequences (containing '*' or non-standard AA)
    """
    logging.info("Reading TSV: %s", in_tsv)
    # Try to infer separator and encoding
    df = pd.read_csv(in_tsv, sep="\t", dtype=str, na_filter=False)
    logging.info("TSV loaded: %d rows, %d cols", df.shape[0], df.shape[1])

    # Normalize columns: try several common names
    col_map = {}
    # Map accession
    if "accession" in df.columns:
        col_map["accession"] = "accession"
    elif accession_col in df.columns:
        col_map[accession_col] = "accession"
    elif "Entry" in df.columns:
        col_map["Entry"] = "accession"

    # Map sequence
    if sequence_col in df.columns:
        col_map[sequence_col] = "sequence"
    elif "sequence" in df.columns:
        col_map["sequence"] = "sequence"
    elif "Sequence" in df.columns:
        col_map["Sequence"] = "sequence"

    # Map protein name
    if protein_name_col in df.columns:
        col_map[protein_name_col] = "protein_name"
    elif "Protein names" in df.columns:
        col_map["Protein names"] = "protein_name"
    elif "protein_name" in df.columns:
        col_map["protein_name"] = "protein_name"

    # Map reviewed column if present
    if reviewed_col and reviewed_col in df.columns:
        col_map[reviewed_col] = "reviewed"
    elif "reviewed" in df.columns:
        col_map["reviewed"] = "reviewed"
    elif "Status" in df.columns:
        col_map["Status"] = "reviewed"

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    if "accession" not in df.columns or "sequence" not in df.columns:
        raise ValueError("Input TSV must contain accession and sequence columns (after normalization).")

    # Drop exact duplicate accessions (keep first)
    before = df.shape[0]
    df = df.drop_duplicates(subset=["accession"], keep="first")
    logging.info("Dropped duplicates: %d -> %d", before, df.shape[0])

    # Drop fragments
    if drop_fragments and "protein_name" in df.columns:
        mask = df["protein_name"].str.lower().str.contains("fragment", na=False)
        n_frag = mask.sum()
        if n_frag > 0:
            logging.info("Dropping %d fragment entries (protein_name contains 'fragment')", n_frag)
            df = df.loc[~mask].copy()

    # Validate sequences
    valid_mask = df["sequence"].apply(lambda s: is_valid_protein_sequence(s, allow_x=allow_x))
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        logging.info("Dropping %d invalid sequences (stops or non-standard AA).", n_invalid)
        df = df.loc[valid_mask].copy()

    # Optional: normalize organism names if column present
    for colname in df.columns:
        if colname.lower() in ("organism", "organism_name", "organism names"):
            df[colname] = df[colname].str.strip()

    logging.info("Cleaning complete: %d rows remaining", df.shape[0])
    return df


def write_cleaned_fasta(df: pd.DataFrame, accession_col: str = "accession",
                        sequence_col: str = "sequence", out_fasta: str = "uniprot_cleaned.fasta"):
    """
    Write cleaned DataFrame to FASTA file in a simple header format:
      >accession|protein_name|organism_name
      SEQUENCE
    """
    records = []
    for _, row in df.iterrows():
        acc = row.get(accession_col)
        seq = row.get(sequence_col)
        if not isinstance(seq, str) or not seq:
            continue
        # Build a short header
        header_parts = [str(acc)]
        if "protein_name" in row and row["protein_name"]:
            header_parts.append(str(row["protein_name"]))
        if "organism_name" in row and row["organism_name"]:
            header_parts.append(str(row["organism_name"]))
        header = " | ".join(header_parts)
        rec = SeqRecord(seq=Seq(seq if isinstance(seq, str) else str(seq)),
                        id=str(acc),
                        description=header)
        records.append(rec)

    logging.info("Writing %d FASTA records to %s", len(records), out_fasta)
    with open(out_fasta, "w") as fh:
        SeqIO.write(records, fh, "fasta")


# ---------------- Example main flow ----------------
def main():
    outdir = "uniprot_data"
    logging.info("Current working directory: %s", os.path.join(os.getcwd(), outdir))
    os.makedirs(outdir, exist_ok=True)

    ### Option A: Bulk download (recommended if you need full UniProtKB releases)
    # Example downloads (reviewed Swiss-Prot FASTA)
    local_gz = os.path.join(outdir, "uniprot_sprot.fasta.gz")
    local_fasta = os.path.join(outdir, "uniprot_sprot.fasta")
    if not os.path.exists(local_gz):
        download_file(EXAMPLE_SPROT_FASTA, local_gz)
    if not os.path.exists(local_fasta):
        maybe_gunzip(local_gz, local_fasta)

    # Parse FASTA (Biopython) into a simple TSV-like DataFrame for cleaning
    logging.info("Parsing FASTA into DataFrame (may take a while for large files)...")
    records = []
    for rec in SeqIO.parse(local_fasta, "fasta"):
        # If headers are UniProt default, accession is first token in rec.id
        acc = rec.id.split("|")[-1] if "|" in rec.id else rec.id
        prot_name = rec.description
        seq = str(rec.seq)
        records.append({"accession": acc, "protein_name": prot_name, "sequence": seq})
    df_fasta = pd.DataFrame.from_records(records)
    logging.info("Parsed %d sequences from FASTA", len(df_fasta))

    df_cleaned = clean_uniprot_tsv_from_df(df_fasta := df_fasta)

    # save cleaned outputs
    out_tsv = os.path.join(outdir, "uniprot_sprot_cleaned.tsv")
    df_cleaned.to_csv(out_tsv, sep="\t", index=False)
    write_cleaned_fasta(df_cleaned, out_fasta=os.path.join(outdir, "uniprot_sprot_cleaned.fasta"))
    logging.info("Bulk-download cleaning finished; outputs in %s", outdir)

    ### Option B: REST API download for a subset / custom fields
    # If you just want human reviewed proteins:
    if False:
        query = "organism_id:9606 AND reviewed:true"   # example: human reviewed entries
        out_tsv = os.path.join(outdir, "uniprot_human_reviewed.tsv")
        if not os.path.exists(out_tsv):
            download_uniprot_via_rest(query=query,
                                    fields=["accession", "id", "protein_name", "organism_name", "sequence", "reviewed"],
                                    out_tsv=out_tsv,
                                    batch_size=500)
        df_rest = clean_uniprot_tsv(out_tsv,
                                    accession_col="Entry",
                                    protein_name_col="Protein names",
                                    sequence_col="Sequence",
                                    reviewed_col="Reviewed",
                                    drop_fragments=True,
                                    allow_x=False)
        df_rest.to_csv(os.path.join(outdir, "uniprot_human_reviewed_cleaned.tsv"), sep="\t", index=False)
        write_cleaned_fasta(df_rest, out_fasta=os.path.join(outdir, "uniprot_human_reviewed_cleaned.fasta"))
        logging.info("REST-download cleaning finished; outputs in %s", outdir)


# small helper to allow cleaning DataFrame directly
def clean_uniprot_tsv_from_df(df: pd.DataFrame) -> pd.DataFrame:
    tmp = os.path.join(".", ".__tmp_uniprot_in.tsv")
    df.to_csv(tmp, sep="\t", index=False)
    cleaned = clean_uniprot_tsv(tmp, accession_col="accession", protein_name_col="protein_name", sequence_col="sequence")
    os.remove(tmp)
    return cleaned


if __name__ == "__main__":
    main()
