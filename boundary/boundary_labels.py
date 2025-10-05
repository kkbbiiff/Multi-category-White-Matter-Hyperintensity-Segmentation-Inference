import numpy as np
import torch
import torch.nn.functional as F


def generate_boundary_labels(semantic_targets, num_classes):
    targets = semantic_targets.float()

    boundary_labels = torch.zeros((targets.size(0), num_classes * (num_classes - 1), targets.size(1), targets.size(2)),
                                  device=targets.device)

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                class_i_mask = (targets == i).float()
                class_j_mask = (targets == j).float()

                boundary_ij = class_i_mask * class_j_mask

                boundary_index = i * (num_classes - 1) + (j if i > j else j - 1)
                boundary_labels[:, boundary_index] = boundary_ij

    return boundary_labels
