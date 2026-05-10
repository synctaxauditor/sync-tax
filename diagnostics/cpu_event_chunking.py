from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Event:
    name: str
    ts: float          # start timestamp
    dur: float         # duration
    meta: Dict[str, Any] = field(default_factory=dict)

    # tree structure
    parent: Optional["Event"] = None
    children: List["Event"] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.ts + self.dur

def build_nesting_tree(events) -> List[Event]:
    """
    Returns list of roots. Assumes events are properly nested (intervals either disjoint or contained),
    not partially overlapping.
    """
    # Sort by start asc; tie-break by end desc (longer first) to make nesting deterministic.
    events_new = []
    for e in events:
        if len(e) == 3:
            events_new.append(Event(name=e[2], ts=e[0], dur=e[1]))
        elif len(e) == 4:
            events_new.append(Event(name=e[2], ts=e[0], dur=e[1], meta={"correlation": e[3]}))
        else:
            raise ValueError(f"Incorrect number of fields for event {e}")

    events_sorted = sorted(events_new, key=lambda e: (e.ts, -e.end))

    stack: List[Event] = []
    roots: List[Event] = []

    for e in events_sorted:
        if e.dur < 0:
            raise ValueError(f"Negative duration for event {e}")

        # Pop anything that ended before (or exactly at) this start.
        while stack and e.ts >= stack[-1].end:
            stack.pop()

        # Now either stack is empty (root) or e must be contained in stack[-1].
        if stack:
            parent = stack[-1]
            # If you want to enforce "subset only", assert containment:
            if e.end > parent.end + 1e-3:  # tiny epsilon for float timestamps
                raise ValueError(
                    f"Partial overlap / non-nested intervals: child "
                    f"({e.ts}-{e.end}) exceeds parent ({parent.ts}-{parent.end})"
                )
            e.parent = parent
            parent.children.append(e)
        else:
            roots.append(e)

        stack.append(e)

    # Children should be in chronological order for chunking
    def sort_children(node: Event):
        node.children.sort(key=lambda c: (c.ts, c.end))
        for c in node.children:
            sort_children(c)

    for r in roots:
        sort_children(r)

    return roots

def chunk_event(node: Event) -> List[Event]:
    """
    Emit chunk-events for the 'exclusive' slices of node that are not covered by any child.
    Recursively also chunks children (so you get B1, C, B2, etc.).
    """
    out = []

    cursor = node.ts
    chunk_idx = 0

    for child in node.children:
        # gap before child -> chunk of parent
        if cursor < child.ts:
            if "correlation" in node.meta:
                out.append((cursor, child.ts - cursor, node.name + f"_{chunk_idx}", node.meta["correlation"]))
            else:
                out.append((cursor, child.ts - cursor, node.name + f"_{chunk_idx}"))
            chunk_idx += 1

        out.extend(chunk_event(child))

        cursor = max(cursor, child.end)

    # trailing gap after last child
    if cursor < node.end:
        if "correlation" in node.meta:
            out.append((cursor, node.end - cursor, node.name + f"_{chunk_idx}", node.meta["correlation"]))
        else:
            out.append((cursor, node.end - cursor, node.name + f"_{chunk_idx}"))

    return out

def chunk_all(events):
    roots = build_nesting_tree(events)
    chunked = []
    for r in roots:
        chunked.extend(chunk_event(r))
        # Add wait events between roots
        if r != roots[-1]:
            wait_start = r.end
            wait_end = roots[roots.index(r) + 1].ts
            if wait_end > wait_start:
                chunked.append((wait_start, wait_end - wait_start, "wait_between_roots"))
    # Return chronologically (like a trace)
    return sorted(chunked, key=lambda e: e[0] + e[1])
