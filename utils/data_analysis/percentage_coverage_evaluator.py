import numpy as np

class PercentageCoverageEvaluator:
    """
    evaluates percentage coverage for series of percentage given in increasing order
    """
    def __init__(self, x, y):
        """
        @param x x-labels
        @param y y-lables (should be in percentage (<100)) in increasing order
        """
        assert(len(x) == len(y))
        self.x = x
        self.y = y

        self.last_idx = 0
        self.last_percent = 0

    def evaluate_percent_coverage(self, percent):
        """
        @param percent (successive calls should give percent in increasing order
        @return x value that achieves given percentage of y
        """
        # if percent isn't monotonically increasing, restart search
        if percent < self.last_percent:
            self.last_idx = 0

        if percent >= self.y[-1]:
            # idx = -1
            return -1
        else:
            idx = self.last_idx + np.argmax(self.y[self.last_idx:] >= percent)

        self.last_idx = idx
        self.last_percent = self.y[idx]
        return self.x[idx]

if __name__ == '__main__':
    N = 10
    x = np.arange(0, 100, 5)/100.0
    y = np.arange(0, 100, 5)
    eval = PercentageCoverageEvaluator(x, y)
    percentages = [10, 100, 70, 90]
    for percent in percentages:
        print('percent:', eval.evaluate_percent_coverage(percent))



