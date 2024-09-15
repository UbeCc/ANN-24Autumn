import numpy as np

class HingeLoss:
    def __init__(self, margin=1.0, power=1):
        """
        Initializes the HingeLoss class.

        Parameters:
        - margin (float): The margin parameter in hinge loss calculation.
        - power (int): The power to which the hinge loss terms are raised.
        """
        self.margin = margin
        self.power = power

    def compute(self, x, y):
        """
        Compute the hinge loss for the provided scores and labels.

        Parameters:
        - x (numpy.ndarray): The scores for each class, shape should be [n_samples, n_classes].
        - y (numpy.ndarray): The actual labels, shape should be [n_samples] and contains indices of the correct classes.

        Returns:
        - float: The computed hinge loss.
        """
        n_samples = x.shape[0]
        
        # Get the correct class scores
        correct_class_scores = x[np.arange(n_samples), y].reshape(-1, 1)
        print(correct_class_scores.shape, "X", x.shape)
        # Calculate the loss matrix
        loss_matrix = np.maximum(0, self.margin - correct_class_scores + x)
        print(loss_matrix.shape)
        # Ensure we do not count the correct class in the loss
        loss_matrix[np.arange(n_samples), y] = 0
        
        # Raise the loss to the power if specified
        if self.power != 1:
            loss_matrix = np.power(loss_matrix, self.power)
        
        # Calculate the mean loss across all samples
        mean_loss = np.sum(loss_matrix) / n_samples
        
        return mean_loss

# Example usage
if __name__ == "__main__":
    x = np.array([[0.1, 0.2], [1.0, 0.2], [0.4, 0.5]])
    y = np.array([0, 1, 0])
    hinge_loss_calculator = HingeLoss(margin=1.0, power=1)
    loss = hinge_loss_calculator.compute(x, y)
    print(f"Hinge Loss: {loss}")