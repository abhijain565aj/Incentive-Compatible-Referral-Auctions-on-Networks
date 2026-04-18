
class Bidder:
    """Small comparable wrapper used by the adapted GIDM implementation."""

    def __init__(self, name, bid=None):
        self.name = name
        self.bid = bid

    def __lt__(self, other):
        return self.bid < other.bid

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"({self.name}, {self.bid})"

    def __hash__(self):
        return hash(self.name)
