from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Tuple, Union

import numpy as np


@dataclass
class AggregateColumnOperation:
    """An aggreaget operation defines the operation that is applied to the combination metrics that are pooled from different simulations. Usually, a combination operation is applied to axis 0, which means across combinations. In our case, this entails a loss of a dimension."""

    name: str
    operation: Callable[..., Union[np.ndarray, np.float64]]


class AggregateColumnOperations:
    MIN: AggregateColumnOperation = AggregateColumnOperation(
        name="min",
        operation=lambda data: np.min(data, axis=0),
    )
    MAX: AggregateColumnOperation = AggregateColumnOperation(
        name="max",
        operation=lambda data: np.max(data, axis=0),
    )
    MEAN: AggregateColumnOperation = AggregateColumnOperation(
        name="mean",
        operation=lambda data: np.mean(data, axis=0),
    )
    SLOPE_MEAN: AggregateColumnOperation = AggregateColumnOperation(
        name="slope_mean",
        operation=lambda data: np.mean(data, axis=0),
    )
    MEDIAN: AggregateColumnOperation = AggregateColumnOperation(
        name="median",
        operation=lambda data: np.median(data, axis=0),
    )
    SLOPE_MEDIAN: AggregateColumnOperation = AggregateColumnOperation(
        name="slope_median",
        operation=lambda data: np.median(data, axis=0),
    )
    Q1: AggregateColumnOperation = AggregateColumnOperation(
        name="q1",
        operation=lambda data: np.percentile(data, 25, axis=0),
    )
    Q3: AggregateColumnOperation = AggregateColumnOperation(
        name="q3",
        operation=lambda data: np.percentile(data, 75, axis=0),
    )
    DELTA: AggregateColumnOperation = AggregateColumnOperation(
        name="delta",
        operation=lambda min_data, max_data: max_data.mean(axis=0)
        - min_data.mean(axis=0),
    )


@dataclass
class ColumnMapping:
    """Defines what combination columns should have which operations applied to them. For example, for the column 'mean', we might want to know the minimum mean, maximum mean, etc. For the column 'median', we might want to know Q1 across means, Q3 across means. These can all be defined individually here. The operations are known combination operations."""

    name: str
    required_columns: List[str]
    operations: List[
        Union[AggregateColumnOperation, Tuple[AggregateColumnOperation, List[str]]]
    ]


class ColumnMappings:
    MEAN: ColumnMapping = ColumnMapping(
        name="mean",
        required_columns=["mean"],
        operations=[
            AggregateColumnOperations.MIN,
            AggregateColumnOperations.MAX,
            AggregateColumnOperations.MEAN,
            AggregateColumnOperations.MEDIAN,
            AggregateColumnOperations.Q1,
            AggregateColumnOperations.Q3,
            (AggregateColumnOperations.DELTA, ["min", "max"]),
        ],
    )
    MEDIAN: ColumnMapping = ColumnMapping(
        name="median",
        required_columns=["median"],
        operations=[
            AggregateColumnOperations.MIN,
            AggregateColumnOperations.MAX,
            AggregateColumnOperations.MEAN,
            AggregateColumnOperations.MEDIAN,
            AggregateColumnOperations.Q1,
            AggregateColumnOperations.Q3,
            (AggregateColumnOperations.DELTA, ["min", "max"]),
        ],
    )
    SLOPE: ColumnMapping = ColumnMapping(
        name="slope",
        required_columns=["slope"],
        operations=[
            AggregateColumnOperations.MIN,
            AggregateColumnOperations.MAX,
            AggregateColumnOperations.MEAN,
            AggregateColumnOperations.MEDIAN,
            AggregateColumnOperations.Q1,
            AggregateColumnOperations.Q3,
        ],
    )
    CONSENSUS: ColumnMapping = ColumnMapping(
        name="consensus",
        required_columns=["raw"],
        operations=[
            AggregateColumnOperation(
                name="raw",
                operation=lambda data: np.float64(data),
            )
        ],
    )


@dataclass
class AggregateColumnConfig:
    """Defines a column and its associated colum mappings. For example, for entropy we might want to get info about its median and slope, but not for consensus."""

    data_column: str
    column_mappings: List[ColumnMapping] = field(
        default_factory=lambda: [
            ColumnMappings.MEAN,
            ColumnMappings.MEDIAN,
            ColumnMappings.SLOPE,
        ]
    )


aggregate_column_configs = {
    "entropy": AggregateColumnConfig(data_column="ctx_entropy_mean"),
    "base_rate_entropy": AggregateColumnConfig(
        data_column="ctx_base_rate_entropy_mean"
    ),
    "activation": AggregateColumnConfig(data_column="ctx_activation_mean"),
    "consensus": AggregateColumnConfig(
        data_column="consensus_reached", column_mappings=[ColumnMappings.CONSENSUS]
    ),
}


def make_aggregate_output_name(
    aggregate_column_prefix: str, column_name: str, operation_name: str
) -> str:
    """Build the aggregate output column name

    Args:
        aggregate_column_prefix (str): Prefix for the column
        aggregate_column_prefix (str): Name of the column that we will apply the action to
        operation_name (str): Action that was applied to the colcolumnunm (min, max, mean etc.)

    Returns:
        str: Combined prefix for the column
    """

    return f"{aggregate_column_prefix}_{column_name}_{operation_name}"


def get_aggregate_column_config(aggregate_column_name: str) -> AggregateColumnConfig:
    """Retrieve the configuration for an aggregate column

    Args:
        aggregate_column_name (str): Name of the aggregate column

    Raises:
        ValueError: Raised if the name of the aggregate column does not reference an existing config

    Returns:
        AggregateColumnConfig: Configuration associated with the specified aggregate column name
    """

    if not aggregate_column_name in aggregate_column_configs:
        raise ValueError(f"'{aggregate_column_name}' is not a valid aggregate column")

    return aggregate_column_configs[aggregate_column_name]


def get_aggregate_metrics(
    aggregated_data: Dict[str, Dict[str, List[Any]]],
    aggregate_column_names: List[str],
) -> Dict[str, Union[np.ndarray, np.float64]]:
    """Get aggregate metrics (min, max, mean) for specified properties of a parameter combination

    Args:
        aggregated_data (Dict[str, Dict[str, List[Any]]]): Aggregated per-combination data
        aggregate_column_names (List[str]): List of properties to get get data for

    Returns:
        Dict[str, Union[np.ndarray, np.float64]]: Dict with properties as keys and floats or numpy arrays as values
    """

    aggregate_metrics = {}

    for aggregate_column_name in aggregate_column_names:
        config = get_aggregate_column_config(aggregate_column_name)
        data_column = config.data_column

        # What deeper columns need to be "actioned" on?
        for column_mapping in config.column_mappings:
            base_columns = column_mapping.required_columns
            for operation_tuple in column_mapping.operations:
                required_columns = base_columns
                if isinstance(operation_tuple, AggregateColumnOperation):
                    operation = operation_tuple
                elif isinstance(operation_tuple, tuple):
                    operation, required_columns = operation_tuple
                else:
                    raise TypeError("Unrecognised action structure")

                # This will be the name of the column in the output dataframe
                output_column_name = make_aggregate_output_name(
                    aggregate_column_name, column_mapping.name, operation.name
                )

                # Automatically retrieve the columns that this specific action operation requires
                required_data = [
                    np.array(aggregated_data[config.data_column][column])
                    for column in required_columns
                ]
                aggregate_result = operation.operation(*required_data)

                aggregate_metrics[output_column_name] = aggregate_result

    return aggregate_metrics
