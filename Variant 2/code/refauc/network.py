from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Iterable, Optional


@dataclass(frozen=True)
class RootedTree:
    """Rooted directed referral tree. Node 0 is the seller/root."""
    parent: Dict[int, Optional[int]]

    @property
    def nodes(self) -> List[int]:
        return sorted(self.parent.keys())

    @property
    def buyers(self) -> List[int]:
        return [u for u in self.nodes if u != 0]

    def children(self, u: int) -> List[int]:
        return sorted([v for v, p in self.parent.items() if p == u])

    def first_level(self) -> List[int]:
        return self.children(0)

    def depth(self, u: int) -> int:
        d = 0
        while u != 0:
            p = self.parent[u]
            if p is None:
                raise ValueError(f"node {u} is disconnected from root")
            u = p
            d += 1
        return d

    def path_to_root(self, u: int) -> List[int]:
        path = [u]
        while u != 0:
            p = self.parent[u]
            if p is None:
                raise ValueError(f"node {u} is disconnected from root")
            u = p
            path.append(u)
        return list(reversed(path))

    def subtree(self, u: int) -> List[int]:
        out: List[int] = []
        q = deque([u])
        while q:
            x = q.popleft()
            out.append(x)
            q.extend(self.children(x))
        return sorted(out)

    def branch_of(self, u: int) -> int:
        if u == 0:
            return 0
        path = self.path_to_root(u)
        return path[1]

    def branch_subtrees(self) -> Dict[int, List[int]]:
        return {c: self.subtree(c) for c in self.first_level()}

    def induced_after_invitation_cut(self, blocked: Iterable[int]) -> "RootedTree":
        """Remove blocked nodes and descendants; useful for diffusion deviations."""
        blocked_set = set(blocked)
        removed = set()
        for b in blocked_set:
            removed.update(self.subtree(b))
        parent = {u: p for u, p in self.parent.items() if u not in removed}
        return RootedTree(parent=parent)

    def is_valid(self) -> bool:
        if 0 not in self.parent or self.parent[0] is not None:
            return False
        for u in self.buyers:
            seen = set()
            x = u
            while x != 0:
                if x in seen or x not in self.parent:
                    return False
                seen.add(x)
                x = self.parent[x]
                if x is None:
                    return False
        return True
