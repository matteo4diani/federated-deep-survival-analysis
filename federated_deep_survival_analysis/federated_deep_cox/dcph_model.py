from auton_survival.models.cph.dcph_utilities import (
    test_step,
)
from auton_survival import DeepCoxPH
from auton_survival import metrics
import numpy as np

from federated_deep_survival_analysis.federated_deep_cox.dcph_dataset import (
    SurvivalDataset,
)


def train(
    model: DeepCoxPH, trainloader: SurvivalDataset, config
):
    lr = config["lr"]
    patience = config["patience"]
    epochs = config["local_epochs"]
    vsize = config["validation_size"]
    batch_size = config["batch_size"]
    weight_decay = config["weight_decay"]

    model.fit(
        trainloader.features,
        trainloader.outcomes.time.values,
        trainloader.outcomes.event.values,
        iters=epochs,
        learning_rate=lr,
        patience=patience,
        vsize=vsize,
        batch_size=batch_size,
        breslow=False,
        weight_decay=weight_decay,
    )


def test(model: DeepCoxPH, testloader):
    features, times, events = prepare_data_from_loader(
        testloader
    )

    test_x, test_t, test_e = prepare_test_data(
        model, features, times, events
    )

    loss = test_step(
        model.torch_module, test_x, test_t, test_e
    )

    concordance_index = metrics.concordance_index_censored(
        events.astype(bool),
        times,
        model.predict_time_independent_risk(
            features
        ).squeeze(),
    )

    return loss, concordance_index


def model_to_parameters(model: DeepCoxPH):
    from flwr.common.parameter import ndarrays_to_parameters

    ndarrays = [
        val.cpu().numpy()
        for _, val in model.torch_module.state_dict().items()
    ]

    parameters = ndarrays_to_parameters(ndarrays=ndarrays)

    return parameters


def prepare_data_from_loader(loader: SurvivalDataset):
    return (
        loader.features,
        loader.outcomes.time.values,
        loader.outcomes.event.values,
    )


def prepare_test_data(
    model: DeepCoxPH, features, times, events
):
    from auton_survival.models.dsm.dsm_utilities import (
        _reshape_tensor_with_nans,
    )

    x, t_, e_, _, _, _ = model._preprocess_training_data(
        features,
        times,
        events,
        0.0,
        (features, times, events),
        0,
    )

    t = _reshape_tensor_with_nans(t_)
    e = _reshape_tensor_with_nans(e_)

    return x, t, e
