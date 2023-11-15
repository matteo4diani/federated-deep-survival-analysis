from auton_survival.models.cph.dcph_utilities import test_step
from auton_survival import DeepCoxPH
from auton_survival import metrics

from federated_deep_survival_analysis.federated_deep_cox.dcph_dataset import SurvivalDataset


def train(model: DeepCoxPH, trainloader: SurvivalDataset, config):
    lr = config["lr"]
    patience = config["patience"]
    epochs = config["local_epochs"]
    vsize = config["validation_size"]
    batch_size = config["batch_size"]
    

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
    )


def test(model: DeepCoxPH, testloader):
    x, t, e = prepare_data_from_loader(testloader)

    loss = test_step(model, x, t, e)
    concordance_index = metrics.concordance_index_censored(
        e,
        t,
        model.predict_time_independent_risk(x).squeeze(),
    )

    return loss, concordance_index


def model_to_parameters(model: DeepCoxPH):
    from flwr.common.parameter import ndarrays_to_parameters

    ndarrays = [
        val.cpu().numpy() for _, val in model.torch_module.state_dict().items()
    ]

    parameters = ndarrays_to_parameters(ndarrays=ndarrays)

    return parameters


def prepare_data_from_loader(testloader):
    features, times, events = testloader

    return features, times, events
