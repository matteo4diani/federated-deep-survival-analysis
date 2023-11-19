from auton_survival.models.cph.dcph_utilities import (
    test_step,
)
from auton_survival import DeepCoxPH
from auton_survival import metrics

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
    momentum = config["momentum"]

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
        momentum=momentum,
    )


def test(model: DeepCoxPH, testloader: SurvivalDataset):
    features, times, events = _prepare_data_from_loader(
        testloader
    )

    test_x, test_t, test_e = _prepare_test_data(
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


def _prepare_data_from_loader(loader: SurvivalDataset):
    return (
        loader.features,
        loader.outcomes.time.values,
        loader.outcomes.event.values,
    )


def _prepare_test_data(
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


def model_to_parameters(model_fn=None):
    from flwr.common.parameter import ndarrays_to_parameters

    model = model_fn()
    ndarrays = [
        val.cpu().numpy()
        for _, val in model.torch_module.state_dict().items()
    ]

    parameters = ndarrays_to_parameters(ndarrays=ndarrays)

    return parameters


def get_model_fn(
    layers=None,
    random_seed=None,
    early_init=None,
    optimizer=None,
    activation=None,
    bias=None,
    input_dim=None,
):
    def model_fn():
        return DeepCoxPH(
            layers,
            random_seed=random_seed,
            early_init=early_init,
            optimizer=optimizer,
            activation=activation,
            bias=bias,
            input_dim=input_dim,
        )

    return model_fn
