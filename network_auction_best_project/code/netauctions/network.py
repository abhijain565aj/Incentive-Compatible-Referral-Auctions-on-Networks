from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Iterable, Set, Tuple


@dataclass(frozen=True)
class RootedTree:
    """Rooted referral tree. Node 0 is the seller/root."""
    parent: Dict[int, Optional[int]]

    @property
    def nodes(self) -> List[int]:
        return sorted(self.parent.keys())

    @property
    def buyers(self) -> List[int]:
        return [u for u in self.nodes if u != 0]

    def children(self, u: int) -> List[int]:
        return sorted(v for v, p in self.parent.items() if p == u)

    def first_level(self) -> List[int]:
        return self.children(0)

    def depth(self, u: int) -> int:
        d = 0
        while u != 0:
            p = self.parent.get(u)
            if p is None:
                raise ValueError(f"node {u} is not connected to root")
            u = p
            d += 1
        return d

    def path(self, u: int, include_root: bool = False) -> List[int]:
        path = [u]
        while u != 0:
            p = self.parent.get(u)
            if p is None:
                raise ValueError(f"node {u} is not connected to root")
            u = p
            path.append(u)
        path = list(reversed(path))
        return path if include_root else [x for x in path if x != 0]

    def subtree(self, u: int) -> List[int]:
        out: List[int] = []
        q: deque[int] = deque([u])
        while q:
            x = q.popleft()
            out.append(x)
            q.extend(self.children(x))
        return sorted(out)

    def outside_subtree(self, u: int) -> List[int]:
        s = set(self.subtree(u))
        return [x for x in self.buyers if x not in s]

    def branch_of(self, u: int) -> int:
        if u == 0:
            return 0
        return self.path(u, include_root=True)[1]

    def branch_subtrees(self) -> Dict[int, List[int]]:
        return {c: self.subtree(c) for c in self.first_level()}

    def is_valid(self) -> bool:
        if 0 not in self.parent or self.parent[0] is not None:
            return False
        for u in self.buyers:
            seen: Set[int] = set()
            x = u
            while x != 0:
                if x in seen or x not in self.parent:
                    return False
                seen.add(x)
                x = self.parent[x]  # type: ignore[index]
                if x is None:
                    return False
        return True

    def insert_sybil_chain(self, edge: Tuple[int, int], count: int, start_id: int) -> "RootedTree":
        """Replace edge u->v by u->z1->...->zk->v."""
        if count <= 0:
            return self
        u, v = edge
        if self.parent.get(v) != u:
            raise ValueError("edge must be a parent-child edge")
        parent = dict(self.parent)
        prev = u
        for z in range(start_id, start_id + count):
            parent[z] = prev
            prev = z
        parent[v] = prev
        return RootedTree(parent)
