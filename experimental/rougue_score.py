from rouge import Rouge


def calculate_rouge_scores(hypotheses, references):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return scores


if __name__ == "__main__":
    # Example usage
    hypothesis = "the cat was found under the bed"
    reference = "the cat was under the bed"

    scores = calculate_rouge_scores(hypothesis, reference)
    print(scores)
