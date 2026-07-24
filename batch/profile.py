from typing import Any, Dict


class BatchProfile:
    def __init__(
        self, parent_params: Dict[str, Any], child_params: Dict[str, Dict[str, Any]]
    ):
        self._parent_params = parent_params
        self._child_params = child_params
        self.param_sets = {}

        self.__post_init__()

    def __post_init__(self):
        # Is there are no child parameters, we create a dummy one
        if len(self._child_params) == 0:
            self._child_params["main"] = {}

        # Now populate all parameter sets by combining parent and child params
        for child_param_key in self._child_params:
            self.param_sets[child_param_key] = {
                **self._parent_params,
                **self._child_params[child_param_key],
            }
