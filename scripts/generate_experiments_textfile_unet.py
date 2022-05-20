import numpy as np
from scipy.stats import loguniform


def main(n_experiments: int = 3000):
    """
    Create a textfile with hyperparameter combinations to be tested. Training job reads one line
    at a time, removing it after reading so multiple training experiments can be run simultaneously.
    """

    experiments = []

    # Sample a combinations of hyperparameters
    for _ in range(n_experiments):

        uniform_weights = str(np.random.randint(0, 2))
        count_alpha = str(np.random.uniform(0, 0.5))
        neg_to_pos_ratio = str(np.random.uniform(0.5, 1.2))
        aug = np.random.choice(["simple", "complex"])
        patience = str(np.random.randint(4, 7))
        loss_mask = np.random.choice(
            [
                "Dice",
                "SoftDice",
                "Mixed",
            ]
        )
        learning_rate = str(loguniform.rvs(1e-4, 2e-3))
        model_architecture = np.random.choice(
            [
                "UnetEfficientNet-b2",
                "UnetEfficientNet-b1",
                "UnetEfficientNet-b0",
            ]
        )
        dropout_regression = str(np.random.uniform(0, 0.35))
        tta = "1"
        experiments.append(
            " ".join(
                [
                    uniform_weights,
                    neg_to_pos_ratio,
                    count_alpha,
                    aug,
                    loss_mask,
                    learning_rate,
                    patience,
                    model_architecture,
                    dropout_regression,
                    tta
                ]
            )
            + "\n"
        )

    # Write experiments to file
    with open("experiments_to_process_unet.txt", "w") as file:
        for line in experiments:
            file.write(line)


if __name__ == "__main__":
    main()
