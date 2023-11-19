from typing import List, Tuple
from flwr.common import Metrics


def get_weighted_average_fn(
    metric_names=["loss", "concordance_index"]
):
    def weighted_average(
        metrics: List[Tuple[int, Metrics]]
    ) -> Metrics:
        aggregate_metrics = {}
        examples = [
            num_examples for num_examples, _ in metrics
        ]
        normalization = sum(examples)

        for metric_name in metric_names:
            # Multiply accuracy of each client by number of examples used
            weighted_metric = [
                num_examples * m[metric_name]
                for num_examples, m in metrics
            ]
            # Aggregate and return custom metric (weighted average)
            aggregate_metrics[metric_name] = (
                sum(weighted_metric) / normalization
            )

        return aggregate_metrics

    return weighted_average
