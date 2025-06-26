# pattern_filters.py
# -----------------
# Functions for filtering and processing detected patterns

import pandas as pd


def drop_duplicates(patterns: list[dict]) -> list[dict]:
  """
  Remove exact duplicates based on (pattern, start, end).
  Keeps the first occurrence.
  """
  seen: set[tuple] = set()
  unique: list[dict] = []
  for p in patterns:
    key = (p["pattern"], p["start_date"], p["end_date"])
    if key not in seen:
      seen.add(key)
      unique.append(p)
  return unique


def suppress_nearby_hits(patterns: list[dict], gap: int = 10) -> list[dict]:
  """
  Remove any pattern whose start is fewer than `gap` bars after the previous
  occurrence *of the same pattern type*.
  Patterns MUST be sorted by start_date before calling this helper.
  """
  if not patterns:
    return patterns
  patterns = sorted(patterns, key=lambda p: p["start_date"])
  kept: list[dict] = [patterns[0]]
  for p in patterns[1:]:
    prev = kept[-1]
    same = p["pattern"] == prev["pattern"]
    delta = (pd.to_datetime(p["start_date"]) - pd.to_datetime(
        prev["end_date"])).seconds / 60
    if not (same and delta < gap):
      kept.append(p)
  return kept


def cluster_and_keep_best(patterns: list[dict], overlap: float = 0.7) -> list[dict]:
  """
  On the daily list, collapse patterns that overlap > `overlap` fraction in
  time; keep only the highest‑score ("value") item of each cluster.
  """
  if not patterns:
    return patterns
  patterns = sorted(patterns, key=lambda p: p["start_date"])
  clusters: list[list[dict]] = []
  for p in patterns:
    placed = False
    for c in clusters:
      last = c[0]  # clusters are homogeneous enough
      # --- compute overlap in days (always non‑negative) ---
      s1, e1 = pd.to_datetime(p["start_date"]), pd.to_datetime(p["end_date"])
      s2, e2 = pd.to_datetime(last["start_date"]), pd.to_datetime(
          last["end_date"])

      # raw overlap length (may be negative if no intersection)
      overlap_len = (min(e1, e2) - max(s1, s2)).days + 1

      # clamp to ≥ 0 so we can safely compare with integers
      inter = max(0, overlap_len)

      # length (in days) of the shorter pattern – also ≥ 1
      shorter = min((e1 - s1).days + 1, (e2 - s2).days + 1)
      if shorter and inter / shorter >= overlap:
        c.append(p)
        placed = True
        break
    if not placed:
      clusters.append([p])

  # pick best‑scoring representative
  best = [max(c, key=lambda x: x["value"]) for c in clusters]
  return best


def filter_patterns_by_criteria(patterns: list[dict], min_value: float = 1.2, 
                               status: str = "Confirmed", min_duration_minutes: int = 20) -> list[dict]:
  """
  Filter patterns based on specified criteria.
  
  Args:
      patterns: List of pattern dictionaries
      min_value: Minimum value/score threshold
      status: Required status (e.g., "Confirmed")
      min_duration_minutes: Minimum pattern duration in minutes
      
  Returns:
      Filtered list of patterns
  """
  return [p for p in patterns if
          p.get("value", 0) >= min_value and 
          p.get("status") == status and 
          (pd.to_datetime(p["end_date"]) - pd.to_datetime(p["start_date"])).seconds / 60 >= min_duration_minutes]


def remove_duplicates_by_status(patterns: list[dict], status_to_remove: str = "Duplicate") -> list[dict]:
  """
  Remove patterns with a specific status.
  
  Args:
      patterns: List of pattern dictionaries
      status_to_remove: Status to filter out
      
  Returns:
      Filtered list of patterns
  """
  return [p for p in patterns if p.get("status") != status_to_remove]