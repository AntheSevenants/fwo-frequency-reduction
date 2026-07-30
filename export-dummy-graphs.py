import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict

import visualisation.distributions
import export.files


@dataclass
class GraphConfig:
    plot_func: Callable
    args: Dict[str, Any]


graphs = {
    "priors_graph": GraphConfig(
        plot_func=visualisation.distributions.plot_priors_graph,
        args={
            "zipf_params": [1, 0.6, 0.4, 0.2, 0.01],
            "title": "Zipfian distribution per parameter setting (with samping)",
            "legend_title": r"Zipfian parameter $\beta$",
            "legend_values": ["1", "0.6", "0.4", "0.2", "0.01"],
        },
    )
}


parser = argparse.ArgumentParser(description="export dummy graphs")
parser.add_argument(
    "export_dir", type=str, help="Directory where figure will be stored"
)
parser.add_argument("graph_name", type=str, help="Name of the graph")
parser.add_argument(
    "--disable_title",
    action="store_true",
    help="Remove title from the graph",
    default=False,
)
args = parser.parse_args()

if args.graph_name not in graphs:
    raise ValueError("Unknown graph")

output_graphs = {
    args.graph_name: graphs[args.graph_name].plot_func(
        **{**graphs[args.graph_name].args, "disable_title": args.disable_title}
    )
}

export.files.export_files(output_graphs, "dummy", args.export_dir)
