import numpy as np

dataset = [
    {"Age": 22, "Income": 30000, "Buy Product": "No"},
    {"Age": 25, "Income": 50000, "Buy Product": "Yes"},
    {"Age": 45, "Income": 80000, "Buy Product": "Yes"},
    {"Age": 35, "Income": 40000, "Buy Product": "No"}
]


def satisfies_condition(example, condition):
    return condition(example)


def ebl(dataset):
    explanation = None
    for example in dataset:
        if example["Buy Product"] == "Yes":
            age = example["Age"]
            income = example["Income"]
            explanation = lambda x: (
                25 <= x["Age"] <= 45
            ) and (
                x["Income"] > 40000
            )
            break
    return explanation


def test_explanation(dataset, explanation):
    predictions = []
    correct_predictions = 0
    false_predictions = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for example in dataset:
        predicted = "Yes" if satisfies_condition(example, explanation) else "No"
        predictions.append(predicted)
        if predicted == example["Buy Product"]:
            correct_predictions += 1
            if predicted == "Yes":
                true_positives += 1
            else:
                true_negatives += 1
        else:
            false_predictions += 1
            if predicted == "Yes":
                false_positives += 1
            else:
                false_negatives += 1
    accuracy = correct_predictions / len(dataset) * 100

    return (
        accuracy,
        predictions,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives
    )


def run():
    explanation = ebl(dataset)
    print(
        "Learned Rule: If the person's age is between 25 and 45 and income is above 40,000, they will buy the product."
    )
    accuracy, predictions, true_positives, false_positives, true_negatives, false_negatives = test_explanation(
        dataset,
        explanation
    )
    print(f"Explanation-based learning accuracy: {accuracy}%")
    print("\nPredictions vs Actuals:")
    for i, example in enumerate(dataset):
        print(
            f"Example {i + 1}: Predicted = {predictions[i]}, Actual = {example['Buy Product']}"
        )

    print(f"\nAccuracy: {accuracy}%")
    print(f"\nTrue Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")

run()
