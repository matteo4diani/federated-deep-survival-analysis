from auton_survival.models.cph.dcph_utilities import test_step
from auton_survival import DeepCoxPH
from auton_survival import metrics


def train(model, trainloader, epochs):
    pass


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
    return None, None, None
