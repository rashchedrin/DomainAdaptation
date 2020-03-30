import torch
import dann_config
import numpy as np

def loss_DANN_(
    class_predictions_logits, 
    logprobs_target,
    instances_labels, 
    is_target,
    unk_value = dann_config.unk_value, 
    weight_domain_loss = dann_config.weight_domain_loss, 
    weight_prediction_loss = dann_config.weight_prediction_loss):
    """
    param class_predictions_logits: Tensor, shape = (batch_size, n_classes). Raw (NO logsoftmax).
    param logprobs_target: Tensor, shape = (batch_size,): logprob predictions that domain is target.
    param instances_labels: np.Array, shape = (batch_size,)
    param is_target: np.Array, shape = (batch_size,)
    param unk_value: value that means that true label is unknown
    """
    
    instances_labels =  torch.Tensor(instances_labels).long()
    is_target = torch.Tensor(is_target).float()
    
    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value)
    prediction_loss = crossentropy(class_predictions_logits, instances_labels)
    binary_crossentropy = torch.nn.BCEWithLogitsLoss()
    domain_loss = binary_crossentropy(logprobs_target, is_target)
    loss = weight_domain_loss * domain_loss \
         + weight_prediction_loss * prediction_loss
    return loss

def test_loss_DANN_():
    cpl = torch.Tensor(
            np.array([
                [1.0,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12]
            ]))
    lpt = torch.Tensor(np.array([-5.1,5,-6, 8]))
    ils = np.array([1,2,-100,2], dtype='int')
    ist = np.array([0,1,0,1], dtype='int')
    assert(abs(0.7448 - loss_DANN_(cpl, lpt, ils, ist)) < 1e-4)
    
def loss_DANN(model, batch, target_domain_idx=dann_config.target_domain_idx):
    # Approximate interface
    class_predictions_logits = model.classes_logits(batch)
    logprobs_target = model.logprob_domain_is_target(batch)
    instances_labels = batch.true_classes 
    is_target = (batch.domain == target_domain_idx)
    return loss_DANN_(class_predictions_logits,
                     logprobs_target,
                     instances_labels,
                     is_target)

