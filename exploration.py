class EpsilonGreedy:
    """Epsilon greedy exploration"""
    def __init__(self):
        self.dec = 0.001
        self.min = 0.01
        self.value = 1

    def decrease(self):
        self.value = self.value - self.dec if self.value > self.min else self.min
