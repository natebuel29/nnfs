import numpy as np
from loss import Loss
class Loss_CategoricalCrossentropy(Loss):

    #Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        #clip data to prevent division by 0
        #clip both sides to not drag mean towards any value
        y_pred_clip = np.clip(y_pred,1e-7,1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clip[range(samples),y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clip*y_true,
                axis=1
            )
        
        #Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        
        return negative_log_likelihoods