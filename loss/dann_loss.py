import numpy as np
import torch
import dann_config


# call loss_DANN instead of this function
def _loss_DANN(
        class_predictions_logits,
        logprobs_target,
        instances_labels,
        is_target,
        domain_loss_weight,
        prediction_loss_weight,
        unk_value=dann_config.UNK_VALUE,
):
    """
    :param class_predictions_logits: Tensor, shape = (batch_size, n_classes).
        Raw (NO logsoftmax).
    :param logprobs_target: Tensor, shape = (batch_size,):
        logprobs that domain is target.
    :param instances_labels: np.Array, shape = (batch_size,)
    :param is_target: np.Array, shape = (batch_size,)
    :param domain_loss_weight: weight of domain loss
    :param prediction_loss_weight: weight of prediction loss
    :param unk_value: value that means that true label is unknown
    """
    instances_labels = torch.Tensor(instances_labels).long()
    is_target = torch.Tensor(is_target).float()

    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value)
    prediction_loss = crossentropy(class_predictions_logits, instances_labels)
    binary_crossentropy = torch.nn.BCEWithLogitsLoss()
    domain_loss = binary_crossentropy(logprobs_target, is_target)
    loss = domain_loss_weight * domain_loss \
           + prediction_loss_weight * prediction_loss
    return loss


def test_loss_DANN_():
    cpl = torch.Tensor(
        np.array([
            [1.0, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]))
    lpt = torch.Tensor(np.array([-5.1, 5, -6, 8]))
    ils = np.array([1, 2, -100, 2], dtype='int')
    ist = np.array([0, 1, 0, 1], dtype='int')
    assert abs(0.7448 - _loss_DANN(cpl, lpt, ils, ist, 1, 1)) < 1e-4
    print("OK test_loss_DANN_")
    return True


def calc_domain_loss_weight(current_iteration,
                            total_iterations,
                            gamma=dann_config.LOSS_GAMMA):
    progress = current_iteration / total_iterations
    lambda_p = 2 / (1 + np.exp(-gamma * progress))
    return lambda_p


def calc_prediction_loss_weight(current_iteration, total_iterations):
    return 1


def loss_DANN(model,
              batch,
              epoch,
              n_epochs,
              target_domain_idx=dann_config.TARGET_DOMAIN_IDX):
    """
    :param model: model.forward(images) should return dict with keys
        'class' : logits  of classes (raw, not logsoftmax)
        'domain': logprobs  for domains (not logits, must sum to 1)
    :param batch: dict with keys 'images', 'true_classes', 'domains'.
    if true_class is unknown, then class should be dann_config.UNK_VALUE
    :param epoch: current number of iteration
    :param n_epochs: total number of iterations
    :param target_domain_idx: what domain number is target
    :return: loss torch.Tensor
    """
    # Approximate interface
    model_output = model.forward(batch['images'])
    class_predictions_logits = model_output['class']
    logprobs_target = model_output['domain'][target_domain_idx]
    instances_labels = batch['true_classes']
    is_target = (batch['domains'] == target_domain_idx)
    domain_loss_weight = calc_domain_loss_weight(epoch,
                                                 n_epochs)
    prediction_loss_weight = calc_prediction_loss_weight(epoch,
                                                         n_epochs)
    return _loss_DANN(class_predictions_logits,
                      logprobs_target,
                      instances_labels,
                      is_target,
                      domain_loss_weight=domain_loss_weight,
                      prediction_loss_weight=prediction_loss_weight)
