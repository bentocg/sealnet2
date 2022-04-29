import numpy as np
from scipy.stats import loguniform
import uuid


def main(n_experiments: int = 3000):
    """
    Create a textfile with hyperparameter combinations to be tested. Training job reads one line
    at a time, removing it after reading so multiple training experiments can be run simultaneously.
    """

    experiments = []

    # Sample a combinations of hyperparameters
    for _ in range(n_experiments):

        uniform_weights = str(np.random.randint(0, 2))
        count_alpha = str(np.random.uniform(0, 0.9))
        neg_to_pos_ratio = str(np.random.uniform(0, 1.0))
        aug = np.random.choice(["simple", "complex"])
        patience = str(np.random.randint(2, 6))
        loss_mask = np.random.choice(
            [
                "Dice",
                "SoftDice",
                "Focal",
                "Mixed",
            ]
        )
        learning_rate = str(loguniform.rvs(1e-5, 1e-2))
        model_architecture = np.random.choice(["Unet", "TransUnet"])
        dropout_regression = str(np.random.uniform(0, 0.35))
        experiments.append(
            " ".join(
                [
                    uniform_weights,
                    count_alpha,
                    neg_to_pos_ratio,
                    aug,
                    loss_mask,
                    learning_rate,
                    patience,
                    model_architecture,
                    dropout_regression
                ]
            ) + "\n"
        )

    # Write experiments to file
    with open("experiments_to_process.txt", "w") as file:
        for line in experiments:
            file.write(line)


if __name__ == "__main__":
    main()
