class EpsilonGreedy:
    """Epsilon greedy exploration"""
    def __init__(self, eps_steps, eps_min=100000):
        self.min = eps_min
        self.value = 1
        self.dec = (self.value - self.min) / eps_steps

    def decrease(self):
        self.value = self.value - self.dec if self.value > self.min else self.min
