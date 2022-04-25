import numpy as np
from scipy.stats import loguniform
import uuid


def main():
    experiments = []
    for _ in range(3000):
        experiment_id = str(uuid.uuid4())
        uniform_weights = str(np.random.randint(0, 2))
        count_alpha = str(np.random.uniform(0, 0.9))
        neg_to_pos_ratio = str(np.random.uniform(0, 1.0))
        aug = np.random.choice(["simple", "complex"])
        patience = str(np.random.uniform(2, 6))
        loss_mask = np.random.choice(
            [
                "Dice",
                "SoftDice",
                "Focal",
                "Mixed",
            ]
        )
        learning_rate = str(loguniform.rvs(1e-5, 1e-3))
        experiments.append(
            " ".join(
                [
                    experiment_id,
                    uniform_weights,
                    count_alpha,
                    neg_to_pos_ratio,
                    aug,
                    loss_mask,
                    learning_rate,
                    patience
                ]
            ) + "\n"
        )

    with open("experiments_to_process.txt", "w") as file:
        for line in experiments:
            file.write(line)


if __name__ == "__main__":
    main()
