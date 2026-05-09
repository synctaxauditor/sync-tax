from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Iterable, Callable, Tuple

@dataclass
class Node:
    """
    Node record. Use Graph.add_node(...) to create/store nodes.
    """
    id: int
    name: str
    timestamp: float
    duration: float
    is_gpu: bool
    stream: int = 0


class DiGraph:
    """
    Simple directed graph:
      - nodes: Dict[node_id, Node]
      - out_edges: Dict[src_id, List[dst_id]]
      - in_edges:  Dict[dst_id, List[src_id]]   (handy for reverse traversals)
    """
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.out_edges: Dict[int, List[int]] = {}
        self.in_edges: Dict[int, List[int]] = {}

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node id {node.id} already exists.")
        self.nodes[node.id] = node
        self.out_edges[node.id] = []
        self.in_edges[node.id] = []

    def add_edge(self, src_id: int, dst_id: int) -> None:
        if src_id not in self.nodes:
            raise KeyError(f"Unknown src_id {src_id}. Add the node first.")
        if dst_id not in self.nodes:
            raise KeyError(f"Unknown dst_id {dst_id}. Add the node first.")
        self.out_edges[src_id].append(dst_id)
        self.in_edges[dst_id].append(src_id)

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def neighbors_out(self, node_id: int) -> List[int]:
        return self.out_edges[node_id]

    def neighbors_in(self, node_id: int) -> List[int]:
        return self.in_edges[node_id]

    def __len__(self) -> int:
        return len(self.nodes)
