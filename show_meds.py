#!/usr/bin/env python3
"""
match_recommendations.py

Given:
 - models/medicines_df.joblib (pandas DataFrame with at least 'id' and 'name' columns)
 - resp.json (API response with {"results": [ {"id":..., "name":..., ...}, ... ]})

This script finds the matching rows in medicines_df that correspond to the recommendation
results and prints a clean table. It supports:
 - id-based matching (robust to types)
 - case-insensitive name matching
 - optional fuzzy name matching (uses difflib)
 - writing matches to CSV/JSON for further inspection

Usage:
    python match_recommendations.py --meds models/medicines_df.joblib --resp resp.json --out matches.csv
"""
from __future__ import annotations
import argparse
import joblib
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import difflib
import sys


def load_meds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Medicines file not found: {path}")
    obj = joblib.load(path)
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict) and "df" in obj and isinstance(obj["df"], pd.DataFrame):
        df = obj["df"]
    else:
        # try to coerce to DataFrame if possible
        df = pd.DataFrame(obj)
    # normalize expected columns
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # ensure id and name columns exist
    if "id" not in df.columns or "name" not in df.columns:
        raise ValueError("medicines_df must contain 'id' and 'name' columns (case-insensitive).")
    return df


def load_resp(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Response file not found: {path}")
    with path.open("r", encoding="utf8") as fh:
        return json.load(fh)


def match_by_id(meds_df: pd.DataFrame, ids: List[int]) -> pd.DataFrame:
    # ensure ids are ints for matching
    meds_df_id = meds_df.copy()
    meds_df_id["id"] = pd.to_numeric(meds_df_id["id"], errors="coerce").astype("Int64")
    ids_int = []
    for i in ids:
        try:
            ids_int.append(int(i))
        except Exception:
            # skip non-intable ids
            continue
    if not ids_int:
        return pd.DataFrame([], columns=meds_df.columns)
    matched = meds_df_id[meds_df_id["id"].isin(ids_int)]
    return matched


def match_by_name(meds_df: pd.DataFrame, names: List[str], case_insensitive: bool = True) -> pd.DataFrame:
    meds_df_names = meds_df.copy()
    if case_insensitive:
        meds_df_names["_norm_name"] = meds_df_names["name"].str.lower().str.strip()
        names_norm = [n.lower().strip() for n in names if isinstance(n, str)]
        matched = meds_df_names[meds_df_names["_norm_name"].isin(names_norm)].drop(columns=["_norm_name"])
    else:
        matched = meds_df_names[meds_df_names["name"].isin(names)]
    return matched


def fuzzy_name_matches(meds_df: pd.DataFrame, names: List[str], cutoff: float = 0.75, top_n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each name in names, produce a list of fuzzy matches from meds_df (using difflib).
    Returns dict: {query_name: [ {name, id, score}, ... ] }
    """
    results = {}
    meds_names = meds_df["name"].astype(str).tolist()
    for q in names:
        q_str = str(q)
        # use difflib.get_close_matches for quick fuzzy suggestions
        close = difflib.get_close_matches(q_str, meds_names, n=top_n, cutoff=cutoff)
        matches = []
        for c in close:
            row = meds_df[meds_df["name"].astype(str) == c].iloc[0]
            # approximate score using SequenceMatcher ratio
            score = difflib.SequenceMatcher(None, q_str, c).ratio()
            matches.append({"id": int(row["id"]) if pd.notna(row["id"]) else None, "name": c, "score": float(score)})
        results[q_str] = matches
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Match recommendation response entries to medicines_df rows.")
    parser.add_argument("--meds", type=Path, default=Path("models/medicines_df.joblib"), help="Path to medicines_df joblib")
    parser.add_argument("--resp", type=Path, default=Path("resp.json"), help="Path to response JSON")
    parser.add_argument("--out", type=Path, help="Optional output path (CSV or JSON). Extension decides format.")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy name matching when exact name matches not found.")
    parser.add_argument("--cutoff", type=float, default=0.75, help="Fuzzy matching cutoff (0..1)")
    args = parser.parse_args(argv)

    # load inputs
    try:
        meds_df = load_meds(args.meds)
    except Exception as e:
        print("Error loading medicines:", e, file=sys.stderr)
        sys.exit(2)

    try:
        resp = load_resp(args.resp)
    except Exception as e:
        print("Error loading response JSON:", e, file=sys.stderr)
        sys.exit(2)

    results = resp.get("results", [])
    if not results:
        print("No results found in response JSON.")
        sys.exit(0)

    # extract ids and names
    ids = [r.get("id") for r in results if r.get("id") is not None]
    names = [r.get("name") for r in results if r.get("name")]

    # perform matching
    by_id = match_by_id(meds_df, ids)
    by_name = match_by_name(meds_df, names)

    # combine unique matches (preserve original column order)
    combined = pd.concat([by_id, by_name], ignore_index=True).drop_duplicates().reset_index(drop=True)

    # if nothing matched and fuzzy requested, try fuzzy
    fuzzy_info = {}
    if combined.empty and args.fuzzy:
        fuzzy_info = fuzzy_name_matches(meds_df, names, cutoff=args.cutoff)
        print("No exact matches found; fuzzy suggestions (per query):")
        for q, cand in fuzzy_info.items():
            print(f"  Query: {q}")
            if cand:
                for c in cand:
                    print(f"    - {c['name']} (id={c['id']}, score={c['score']:.3f})")
            else:
                print("    - (no close matches)")

    # Output results
    if not combined.empty:
        print(f"Found {len(combined)} matching rows in medicines_df:\n")
        # print a tidy table
        display_df = combined.copy()
        # ensure id and name columns are present and visible
        cols_to_show = [c for c in ["id", "name", "desc"] if c in display_df.columns]
        print(display_df[cols_to_show].to_string(index=False))
    else:
        print("No exact matches found in medicines_df.")

    # Optionally write output
    if args.out:
        ext = args.out.suffix.lower()
        if ext == ".csv":
            combined.to_csv(args.out, index=False)
            print(f"\nWrote matches to CSV: {args.out}")
        elif ext == ".json":
            combined.to_json(args.out, orient="records", force_ascii=False)
            print(f"\nWrote matches to JSON: {args.out}")
        else:
            print("Output extension unrecognized. Supported: .csv, .json")

    # Return fuzzy suggestions as structured data (for downstream usage)
    return {"matches": combined.to_dict(orient="records"), "fuzzy": fuzzy_info}


if __name__ == "__main__":
    main()
