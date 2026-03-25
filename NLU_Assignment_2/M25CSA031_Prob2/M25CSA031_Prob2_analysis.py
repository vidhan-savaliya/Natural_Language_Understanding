"""
Problem 2 : Qualitative Analysis

Analysis of the realism of the generated names and identifying a common failures.

it shows generated samples, checking for commo failure and computing asimple realism heuristic score.


"""

import os
import re


def load_names(filepath):
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found!")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def analyze_model_output(generated, model_name):
    """
    Runs a quality checking and prints analysis.
    """
    print(f"\n{'='*55}")
    print(f"  QUALITATIVE ANALYSIS: {model_name}")
    print(f"{'='*55}")

    if not generated:
        print("  No generated names found.")
        return

    total = len(generated)

    # Show 10 sampled names
    print("\n  Representative Samples:")
    for i, name in enumerate(generated[:10]):
        print(f"    {i+1}. {name}")

    # checking for common failures
    too_short = [n for n in generated if len(n) < 3]
    too_long = [n for n in generated if len(n) > 25]
    has_weird_chars = [n for n in generated if not re.match(r'^[A-Za-z ]+$', n)]
    empty_names = [n for n in generated if len(n) == 0]
    has_space = [n for n in generated if ' ' in n]
    proper_case = [n for n in generated if n and n[0].isupper()]

    print(f"\n  Failure Mode Analysis ({total} names):")
    print(f"    Too short (<3 chars):     {len(too_short)} ({len(too_short)/total*100:.1f}%)")
    if too_short[:3]:
        print(f"      Examples: {too_short[:3]}")
    print(f"    Too long (>25 chars):     {len(too_long)} ({len(too_long)/total*100:.1f}%)")
    print(f"    Non-alphabetic chars:     {len(has_weird_chars)} ({len(has_weird_chars)/total*100:.1f}%)")
    print(f"    Empty names:              {len(empty_names)}")
    print(f"    Has space (first+last):   {len(has_space)} ({len(has_space)/total*100:.1f}%)")
    print(f"    Proper capitalization:    {len(proper_case)} ({len(proper_case)/total*100:.1f}%)")

    avg_len = sum(len(n) for n in generated) / total
    print(f"    Average name length:      {avg_len:.1f} characters")

    # Simple realism heuristic
    score = 0
    if len(has_space) / total > 0.5: score += 1
    if len(proper_case) / total > 0.7: score += 1
    if len(too_short) / total < 0.1: score += 1
    if len(has_weird_chars) / total < 0.05: score += 1

    labels = ["Poor", "Below Average", "Average", "Good", "Excellent"]
    print(f"\n    Realism Score: {score}/4 ({labels[score]})")


if __name__ == "__main__":
    training = load_names("TrainingNames.txt")
    print(f"Training data: {len(training)} names")

    model_outputs = {
        "Vanilla RNN":     "generated_vanilla_rnn.txt",
        "BLSTM":           "generated_blstm.txt",
        "RNN + Attention": "generated_rnn_attention.txt",
    }

    for model_name, filepath in model_outputs.items():
        generated = load_names(filepath)
        analyze_model_output(generated, model_name)

    