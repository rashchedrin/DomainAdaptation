import numpy as np
import torch
import config.dann_config as dann_config


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


# call loss_DANN instead of this function
def _loss_DANN_splitted(
        class_logits_on_source,
        class_logits_on_target,
        logprobs_target_on_source,
        logprobs_target_on_target,
        true_labels_on_source,
        true_labels_on_target,
        domain_loss_weight,
        prediction_loss_weight,
        unk_value=dann_config.UNK_VALUE,
):
    """
    :param class_logits_on_source: Tensor, shape = (batch_size, n_classes).
    :param class_logits_on_target: Tensor, shape = (batch_size, n_classes).
    :param logprobs_target_on_source: Tensor, shape = (batch_size,):
    :param logprobs_target_on_target: Tensor, shape = (batch_size,):
    :param true_labels_on_source: np.Array, shape = (batch_size,)
    :param true_labels_on_target: np.Array, shape = (batch_size,)
    :param domain_loss_weight: weight of domain loss
    :param prediction_loss_weight: weight of prediction loss
    :param unk_value: value that means that true class label is unknown
    """
    # TARGET_DOMAIN_IDX is 1
    source_len = len(class_logits_on_source)
    target_len = len(class_logits_on_target)
    true_labels_on_source = torch.Tensor(true_labels_on_source).long()
    true_labels_on_target = torch.Tensor(true_labels_on_target).long()
    is_target_on_source = torch.zeros(source_len).float()
    is_target_on_target = torch.ones(target_len).float()

    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value, reduction='sum')
    prediction_loss_on_source = crossentropy(class_logits_on_source, true_labels_on_source)
    prediction_loss_on_target = crossentropy(class_logits_on_target, true_labels_on_target)
    n_known = (true_labels_on_source != unk_value).sum() + \
              (true_labels_on_target != unk_value).sum()
    prediction_loss = (prediction_loss_on_source + prediction_loss_on_target) / n_known

    binary_crossentropy = torch.nn.BCEWithLogitsLoss(reduction='sum')
    domain_loss_on_source = binary_crossentropy(logprobs_target_on_source, is_target_on_source)
    domain_loss_on_target = binary_crossentropy(logprobs_target_on_target, is_target_on_target)
    domain_loss = (domain_loss_on_source + domain_loss_on_target) / (source_len + target_len)
    loss = domain_loss_weight * domain_loss \
           + prediction_loss_weight * prediction_loss
    return loss, {
        "domain_loss_on_source": domain_loss_on_source,
        "domain_loss_on_target": domain_loss_on_target,
        "domain_loss": domain_loss,
        "prediction_loss_on_source": prediction_loss_on_source,
        "prediction_loss_on_target": prediction_loss_on_target,
        "prediction_loss": prediction_loss
    }


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
              n_epochs):
    """
    :param model: model.forward(images) should return dict with keys
        'class' : Tensor, shape = (batch_size, n_classes)  logits  of classes (raw, not logsoftmax)
        'domain': Tensor, shape = (batch_size,) logprob for domain
    :param batch: dict with keys
        'src_images':
        'trg_images':
        'src_classes':np.Array, shape = (batch_size,)
        'trg_classes':np.Array, shape = (batch_size,)
    if true_class is unknown, then class should be dann_config.UNK_VALUE
    :param epoch: current number of iteration
    :param n_epochs: total number of iterations
    :return:
        loss: torch.Tensor,
        losses dict:{
            "domain_loss_on_source"
            "domain_loss_on_target"
            "domain_loss"
            "prediction_loss_on_source"
            "prediction_loss_on_target"
            "prediction_loss"
        }
    """
    model_output = model.forward(batch['src_images'])
    class_logits_on_source = model_output['class']
    logprobs_target_on_source = model_output['domain']

    model_output = model.forward(batch['trg_images'])
    class_logits_on_target = model_output['class']
    logprobs_target_on_target = model_output['domain']

    domain_loss_weight = calc_domain_loss_weight(epoch, n_epochs)
    prediction_loss_weight = calc_prediction_loss_weight(epoch, n_epochs)
    return _loss_DANN_splitted(
        class_logits_on_source,
        class_logits_on_target,
        logprobs_target_on_source,
        logprobs_target_on_target,
        true_labels_on_source=batch['src_classes'],
        true_labels_on_target=batch['trg_classes'],
        domain_loss_weight=domain_loss_weight,
        prediction_loss_weight=prediction_loss_weight)
