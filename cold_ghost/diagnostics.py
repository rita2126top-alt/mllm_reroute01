"""Internal-validation diagnostics; no benchmark scores or invented results."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def state_errors(current, target):
    current, target = current.float(), target.to(current.device, torch.float32)
    return ((current - target).square().mean(-1), 1.0 - F.cosine_similarity(current, target, dim=-1, eps=1e-6))


def reactivation_events(trace, decisions, teacher):
    result = {}
    for layer in sorted(decisions):
        if layer - 1 not in trace:
            continue
        item = trace[layer]
        returned = item.active_ids[~torch.isin(item.active_ids, trace[layer - 1].active_ids)]
        if not returned.numel():
            continue
        mse, cosine = state_errors(item.visual_input.index_select(1, returned),
                                   teacher[layer].visual_input.to(item.visual_input.device).index_select(1, returned))
        for index, token in enumerate(returned.tolist()):
            result[(layer, token)] = {"age": int(item.age[token]), "mse": float(mse[0, index]),
                                      "cosine_error": float(cosine[0, index])}
    return result


def grouped_events(events):
    groups = {}
    for event in events:
        groups.setdefault(str(event["age"]), []).append(event)
    return {age: {"count": len(items), "mse": sum(item["mse"] for item in items) / len(items),
                   "cosine_error": sum(item["cosine_error"] for item in items) / len(items)}
            for age, items in sorted(groups.items(), key=lambda pair: int(pair[0]))}


def average_precision(scores, labels):
    """Tie-group AP (all tied candidates share a threshold), or None if no positives."""
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: -pair[0])
    true_count = count = index = 0
    ap = 0.0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positive = sum(label for _, label in ordered[index:end])
        true_count += group_positive
        count += end - index
        ap += (group_positive / positives) * (true_count / count)
        index = end
    return ap


def candidate_diagnostics(trace, decisions, teacher):
    scores, labels, error_groups = [], [], {"next_reactivated": [], "next_not_reactivated": []}
    selected_count = true_selected = positive_count = 0
    for layer, item in sorted(trace.items()):
        if item.next_decision is None or not item.skip_ids.numel():
            continue
        target = decisions[item.next_decision].selected_mask[0].index_select(0, item.skip_ids).bool()
        ghost = torch.isin(item.skip_ids, item.ghost_ids)
        selected_count += int(ghost.sum())
        positive_count += int(target.sum())
        true_selected += int((ghost & target).sum())
        rank_scores = item.react_logits[0] if item.react_logits is not None else item.decision_scores[0].index_select(0, item.skip_ids)
        scores.extend(rank_scores.float().cpu().tolist())
        labels.extend(target.int().cpu().tolist())
        current = item.visual_input.index_select(1, item.skip_ids)
        dense = teacher[layer].visual_input.to(current.device).index_select(1, item.skip_ids)
        mse, cosine = state_errors(current, dense)
        for positive, group in ((True, "next_reactivated"), (False, "next_not_reactivated")):
            mask = target == positive
            for m, c in zip(mse[0, mask].tolist(), cosine[0, mask].tolist()):
                error_groups[group].append({"mse": m, "cosine_error": c})
    return {"scores": scores, "labels": labels, "errors": error_groups,
            "ghost_count": selected_count, "positive_count": positive_count, "true_selected": true_selected}


def summarize_candidates(reports):
    scores = [score for report in reports for score in report["scores"]]
    labels = [label for report in reports for label in report["labels"]]
    selected = sum(report["ghost_count"] for report in reports)
    positive = sum(report["positive_count"] for report in reports)
    correct = sum(report["true_selected"] for report in reports)
    result = {"candidate_count": len(labels), "positive_count": positive, "ghost_count": selected,
               "precision": correct / selected if selected else None,
               "recall": correct / positive if positive else None,
               "average_precision": average_precision(scores, labels)}
    for group in ("next_reactivated", "next_not_reactivated"):
        items = [item for report in reports for item in report["errors"][group]]
        result[group] = {"count": len(items), **{metric: sum(item[metric] for item in items) / len(items) if items else None
                                                for metric in ("mse", "cosine_error")}}
    return result


def residual_rank_diagnostics(trace, teacher):
    """Exact rank-8/rank-32 subspaces of dense residuals at actual Active IDs."""
    rows = []
    for layer, item in sorted(trace.items()):
        if item.next_decision is None or not item.skip_ids.numel() or not item.active_ids.numel():
            continue
        dense = teacher[layer].residual.to(item.visual_input.device, torch.float32)[0]
        active = dense.index_select(0, item.active_ids)
        skipped = dense.index_select(0, item.skip_ids)
        _, _, basis = torch.linalg.svd(active, full_matrices=False)
        for requested_rank in (8, 32):
            rank = min(requested_rank, basis.shape[0])
            selected_basis = basis[:rank]
            reconstruction = (skipped @ selected_basis.T) @ selected_basis
            rows.append({"layer": layer, "rank": requested_rank, "effective_rank": rank,
                         "count": skipped.shape[0], "mse_sum": float((skipped - reconstruction).square().mean(-1).sum())})
    return rows


def summarize_rank(rows):
    result = {}
    for rank in (8, 32):
        selected = [row for row in rows if row["rank"] == rank]
        count = sum(row["count"] for row in selected)
        result[str(rank)] = {"count": count, "mse": sum(row["mse_sum"] for row in selected) / count if count else None,
                              "effective_rank_min": min((row["effective_rank"] for row in selected), default=None)}
    return result
