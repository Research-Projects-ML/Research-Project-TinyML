import tensorflow_model_optimization as tfmot


def apply_pruning(model, epochs):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=epochs * 1000
        )
    }

    return prune_low_magnitude(model, **pruning_params)