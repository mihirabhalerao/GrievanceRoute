"""
Apriori Association Rule Mining for GrievanceRoute.
Implements Apriori from scratch and cross-checks with mlxtend.
Calculates Support, Confidence, and Lift on discretized complaint contexts.
"""

from __future__ import annotations
from itertools import combinations
from typing import List, Set, Dict, Tuple, Any

import pandas as pd
import numpy as np


def prepare_discretized_transactions(df: pd.DataFrame) -> List[List[str]]:
    """Convert complaint records into categorical item transaction sets."""
    transactions: List[List[str]] = []
    for _, row in df.iterrows():
        items: List[str] = []
        if "complaint_type" in row:
            items.append(f"Type={row['complaint_type']}")
        if "zone" in row:
            items.append(f"Zone={row['zone']}")
        if "weather" in row and pd.notnull(row["weather"]):
            items.append(f"Weather={row['weather']}")
        if "monsoon" in row and bool(row["monsoon"]):
            items.append("Monsoon=True")
        if "low_lying_area" in row and bool(row["low_lying_area"]):
            items.append("LowLying=True")
        if "repeat_complaint" in row and bool(row["repeat_complaint"]):
            items.append("Repeat=True")
        if "urgency_score" in row:
            u = int(row["urgency_score"])
            if u >= 4:
                items.append("Urgency=High(4-5)")
            elif u <= 2:
                items.append("Urgency=Low(1-2)")
            else:
                items.append("Urgency=Medium(3)")
        transactions.append(items)
    return transactions


def apriori_from_scratch(
    transactions: List[List[str]],
    min_support: float = 0.05,
    min_confidence: float = 0.25,
) -> Tuple[Dict[Tuple[str, ...], float], List[Dict[str, Any]]]:
    """
    Pure Python Apriori implementation for frequent itemset generation and association rules.
    Calculates support, confidence, and lift.
    """
    n_trans = len(transactions)
    if n_trans == 0:
        return {}, []

    # Count single items C1
    item_counts: Dict[str, int] = {}
    for t in transactions:
        for item in set(t):
            item_counts[item] = item_counts.get(item, 0) + 1

    # Filter L1
    frequent_itemsets: Dict[Tuple[str, ...], float] = {}
    current_L: Set[Tuple[str, ...]] = set()

    for item, count in item_counts.items():
        sup = count / n_trans
        if sup >= min_support:
            frequent_itemsets[(item,)] = sup
            current_L.add((item,))

    k = 2
    trans_sets = [set(t) for t in transactions]

    while current_L and k <= 3:  # up to 3-item combinations for clarity
        candidates: Set[Tuple[str, ...]] = set()
        l_list = list(current_L)
        for i in range(len(l_list)):
            for j in range(i + 1, len(l_list)):
                union_set = set(l_list[i]).union(set(l_list[j]))
                if len(union_set) == k:
                    candidates.add(tuple(sorted(union_set)))

        # Count support for candidates
        next_L: Set[Tuple[str, ...]] = set()
        for cand in candidates:
            cand_set = set(cand)
            count = sum(1 for ts in trans_sets if cand_set.issubset(ts))
            sup = count / n_trans
            if sup >= min_support:
                frequent_itemsets[cand] = sup
                next_L.add(cand)

        current_L = next_L
        k += 1

    # Generate Association Rules: A -> B
    rules: List[Dict[str, Any]] = []
    for itemset, sup_itemset in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                sup_antecedent = frequent_itemsets.get(tuple(sorted(antecedent)), 0.0)
                sup_consequent = frequent_itemsets.get(tuple(sorted(consequent)), 0.0)

                if sup_antecedent > 0:
                    conf = sup_itemset / sup_antecedent
                    if conf >= min_confidence:
                        lift = conf / sup_consequent if sup_consequent > 0 else 0.0
                        rules.append({
                            "antecedents": list(antecedent),
                            "consequents": list(consequent),
                            "support": round(sup_itemset, 4),
                            "confidence": round(conf, 4),
                            "lift": round(lift, 4),
                        })

    rules.sort(key=lambda x: x["lift"], reverse=True)
    return frequent_itemsets, rules


def run_mlxtend_apriori(
    transactions: List[List[str]],
    min_support: float = 0.05,
    min_confidence: float = 0.25,
) -> pd.DataFrame:
    """Run mlxtend Apriori and generate association rules DataFrame."""
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    result_df = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    result_df["antecedents"] = result_df["antecedents"].apply(lambda s: list(s))
    result_df["consequents"] = result_df["consequents"].apply(lambda s: list(s))
    result_df = result_df.sort_values(by="lift", ascending=False).reset_index(drop=True)
    return result_df
