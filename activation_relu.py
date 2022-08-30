import numpy as np

class Activation_ReLU:

    #forward pass
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)