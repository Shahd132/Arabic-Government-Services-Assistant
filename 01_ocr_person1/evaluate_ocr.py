"""
evaluate_ocr.py
Measures how good your OCR is using CER (Character Error Rate) and
WER (Word Error Rate) against ground-truth text.

Needs: pip install jiwer

Usage:
    from evaluate_ocr import evaluate
    scores = evaluate(predicted_text, ground_truth_text)
    print(scores)  # {'cer': 0.05, 'wer': 0.12}
"""

import jiwer


def evaluate(predicted: str, ground_truth: str) -> dict:
    """
    Lower is better for both metrics. 0.0 = perfect match.
    cer: Character Error Rate
    wer: Word Error Rate
    """
    cer = jiwer.cer(ground_truth, predicted)
    wer = jiwer.wer(ground_truth, predicted)
    return {"cer": cer, "wer": wer}


def evaluate_dataset(pairs: list) -> dict:
    """
    pairs: list of (predicted, ground_truth) tuples, e.g. one per test image.
    Returns average CER/WER across the whole set -- this is the number
    you report for your project.
    """
    predictions = [p for p, _ in pairs]
    references = [g for _, g in pairs]

    cer = jiwer.cer(references, predictions)
    wer = jiwer.wer(references, predictions)
    return {"cer": cer, "wer": wer, "num_samples": len(pairs)}


if __name__ == "__main__":
    # quick sanity check with a fake example
    pred = "اهلا بالعالم"
    truth = "اهلا يا عالم"
    print(evaluate(pred, truth))
