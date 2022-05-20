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

        aug = np.random.choice(["simple", "complex"])
        patience = str(np.random.randint(4, 7))
        learning_rate = str(loguniform.rvs(1e-4, 2e-3))
        model_architecture = np.random.choice(
            [
                "maskrcnn_resnet50_fpn",
                "fasterrcnn_resnet50_fpn",
            ]
        )
        box_iou_thresh = str(np.random.uniform(0.4, 0.8))
        experiments.append(
            " ".join(
                [
                    aug,
                    learning_rate,
                    patience,
                    model_architecture,
                    box_iou_thresh
                ]
            )
            + "\n"
        )

    # Write experiments to file
    with open("experiments_to_process_maskrcnn.txt", "w") as file:
        for line in experiments:
            file.write(line)


if __name__ == "__main__":
    main()
