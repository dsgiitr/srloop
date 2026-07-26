"""
Intra-Loop Self Distillation (ILSD) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ILSDLoss(nn.Module):
    """
    Intra-Loop Self Distillation Loss for the ELT model.
    Combines the standard diffusion noise-prediction loss on the final loop
    with a distillation loss where the final loop acts as a teacher for intermediate loops.
    """
    def __init__(self, distillation_weight=1.0, student_loops=None):
        """
        :param distillation_weight: Weight of the distillation loss term.
        :param student_loops: A list of loop indices to apply distillation to.
                              If None, applies to all loops 0 to L-2.
        """
        super().__init__()
        self.distillation_weight = distillation_weight
        self.student_loops = student_loops

    def forward(self, loop_predictions, target_noise):
        """
        :param loop_predictions: List of predictions [out_1, out_2, ..., out_L] from the model
        :param target_noise: The true noise added to the latents/images
        :return: (total_loss, dict_of_loss_components)
        """
        num_loops = len(loop_predictions)
        
        # 1. Standard Diffusion Loss on the final loop (Teacher)
        final_prediction = loop_predictions[-1]
        
        # Depending on whether the model learns sigma (variance), we might need to split the output
        if final_prediction.shape[1] == target_noise.shape[1] * 2:
            # Model predicts both noise and variance (e.g. DiT)
            final_pred_noise, _ = torch.split(final_prediction, target_noise.shape[1], dim=1)
        else:
            final_pred_noise = final_prediction

        task_loss = F.mse_loss(final_pred_noise, target_noise, reduction='mean')

        # 2. Intra-Loop Self Distillation (ILSD)
        # Teacher target is the detached final prediction
        teacher_target = final_pred_noise.detach()
        
        distill_loss = 0.0
        
        # Determine which intermediate loops to penalize
        if self.student_loops is None:
            # By default, use all loops except the last one
            student_indices = range(num_loops - 1)
        else:
            student_indices = self.student_loops

        valid_students = 0
        for i in student_indices:
            if i < num_loops - 1:
                student_pred = loop_predictions[i]
                if student_pred.shape[1] == target_noise.shape[1] * 2:
                    student_pred_noise, _ = torch.split(student_pred, target_noise.shape[1], dim=1)
                else:
                    student_pred_noise = student_pred
                
                distill_loss += F.mse_loss(student_pred_noise, teacher_target, reduction='mean')
                valid_students += 1

        if valid_students > 0:
            distill_loss = distill_loss / valid_students

        total_loss = task_loss + self.distillation_weight * distill_loss
        
        return total_loss, {
            "task_loss": task_loss.item(),
            "distill_loss": distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
            "total_loss": total_loss.item()
        }
