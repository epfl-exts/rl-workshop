import datetime
import os
import pathlib
from abc import abstractmethod, ABC
from typing import Any

import numpy as np


class Logger(ABC):
    def __init__(self, ) -> None:
        super().__init__()
        self.start = datetime.datetime.now()

    @abstractmethod
    def log_dict(self, global_step: int, values: dict) -> None:
        pass


class TensorBoardLogger(Logger):
    def __init__(self, path: str, name: str) -> None:
        super().__init__()
        import tensorflow as tf
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(os.path.join(path, name))

    def log_histogram(self, global_step: int, name: str, values: Any, bins=1000) -> None:
        import tensorflow as tf
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        bin_edges = bin_edges[1:]
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
        self.log_tensoboard_summary(global_step, summary)

    def log_dict(self, global_step: int, values: dict) -> None:
        import tensorflow as tf
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value) for name, value in values.items()])
        self.log_tensoboard_summary(global_step, summary)

    def log_tensoboard_summary(self, global_step: int, summary) -> None:
        self.summary_writer.add_summary(summary, global_step)


class NoLogger(Logger):

    def log_dict(self, global_step: int, values: dict) -> None:
        pass
