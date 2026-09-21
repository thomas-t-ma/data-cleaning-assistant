from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .actions import CleaningAction
from .transforms import preview_action


def overlapping_previews(
    tables: dict[str, pd.DataFrame],
    actions: list[CleaningAction],
) -> dict[str, list[tuple[str, int]]]:
    """Report overlapping independent previews, without executing any action.

    Only same-table, same-column cell edits are compared. The count is the
    number of original row positions touched by both proposed actions. It
    intentionally does not count row-deletion interactions.
    """
    targets: dict[tuple[str, str], list[CleaningAction]] = defaultdict(list)
    for action in actions:
        if action.column is not None and action.table in tables:
            targets[(action.table, action.column)].append(action)

    result: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (table, _column), candidates in targets.items():
        if len(candidates) < 2:
            continue
        row_sets = {
            action.id: {
                change.row_index
                for change in preview_action(
                    tables, action, sample_limit=len(tables[table])
                ).sample_changes
            }
            for action in candidates
        }
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                intersection = len(row_sets[left.id] & row_sets[right.id])
                if intersection:
                    result[left.id].append((right.id, intersection))
                    result[right.id].append((left.id, intersection))
    return dict(result)
