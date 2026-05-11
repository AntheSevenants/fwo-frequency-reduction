import model.enums
import model.reduction

# TODO
params = {
    "regular": {
        "num_agents": 10,
        "reduction": [
            model.reduction.SoftThresholding(value_floor=5, strength=1),
            model.reduction.SoftThresholding(value_floor=5, strength=5),
        ],
        "reduction_prob": [0.1, 0.5, 1],
        "vector_bounds": [[70, 100]],
    }
}
