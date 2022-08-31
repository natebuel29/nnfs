import numpy as np

class Loss:

    #Calculate the data and regularization losses 
    #given model output and ground truth values
    def calculate(self,output,y):

        #calculate sample loses
        sample_losses = self.forward(output,y)

        #calculate the mean loss
        data_loss = np.mean(sample_losses)

        return data_loss