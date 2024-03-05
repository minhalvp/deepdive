# Use Plotext to create a terminal Dashboard to show realtime training info e.g. loss
import plotext as plt


class Dash:
    def __init__(self, num_metrics: int) -> None:
        self.num_metrics = num_metrics
        self.metrics = [[] for _ in range(num_metrics)]
        self.t = 0

    def update(self, *args) -> None:
        for i, arg in enumerate(args):
            self.metrics[i].append(arg)
        self.t += 1
        self.plot()

    def plot(self) -> None:
        plt.clf()
        for i in range(self.num_metrics):
            plt.plot(self.metrics[i], label=f"metric {i}")
        plt.title(f"Training Metrics")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()
