# test_templates.py
from collections import Counter, defaultdict
from vision import quick_extract
from ALL_GROUND_TRUTH import GROUND_TRUTH

TEMPLATES_PATH = "templates/letter_templates_v1.npz"


def print_grid(grid):
    for row in grid:
        print("".join(row))


def compare_one(
    filename,
    mode="template",
    confusion_counter=None,
    misclassified_log=None,
    verbose=True,
):
    """
    Compare prediction vs ground truth for a single screenshot.

    Args:
        filename: image filename (e.g. "IMG_1190.PNG")
        mode: "template" or "hybrid"
        confusion_counter: Counter to track (true, pred) mismatches
        misclassified_log: dict-like mapping (true, pred) -> list of (filename, r, c)
        verbose: if True, print grids and accuracy

    Returns:
        (correct_cells, total_cells)
    """
    image_path = f"data/screenshots/{filename}"
    gt = GROUND_TRUTH[filename]  # list-of-lists of chars

    pred_grid, theme, _ = quick_extract(
        image_path,
        recognizer=mode,
        templates_path=TEMPLATES_PATH,
    )

    rows = len(gt)
    cols = len(gt[0])

    correct = 0
    total = rows * cols

    for r in range(rows):
        for c in range(cols):
            true_letter = gt[r][c]
            pred_letter = pred_grid[r][c]

            if pred_letter == true_letter:
                correct += 1
            else:
                # Track confusion if a Counter was provided
                if confusion_counter is not None:
                    confusion_counter[(true_letter, pred_letter)] += 1

                # Track exactly where each misclassification happened
                if misclassified_log is not None:
                    misclassified_log[(true_letter, pred_letter)].append(
                        (filename, r, c)
                    )

    if verbose:
        print(f"\n=== {filename} ===")
        print("GROUND TRUTH:")
        print_grid(gt)
        print("\nPREDICTED:")
        print_grid(pred_grid)
        print(f"\nCell accuracy: {correct}/{total} = {correct/total:.3f}")

    return correct, total


if __name__ == "__main__":
    # Mode to test: "template" or "hybrid"
    MODE = "template"

    total_correct = 0
    total_cells = 0
    confusions = Counter()
    misclassified_examples = defaultdict(list)  # (true, pred) -> list of (fname, r, c)

    # Loop over *all* puzzles we have ground truth for
    for fname in GROUND_TRUTH.keys():
        c, t = compare_one(
            fname,
            mode=MODE,
            confusion_counter=confusions,
            misclassified_log=misclassified_examples,
            verbose=False,    # set True if you want to see all grids
        )
        total_correct += c
        total_cells += t

    overall = total_correct / total_cells if total_cells else 0.0

    print("\n=======================================")
    print(f"Overall accuracy ({MODE}): {total_correct}/{total_cells} = {overall:.4f}")
    print("=======================================\n")

    # ---- Confusion summary ----
    if confusions:
        print("Top misclassifications (true → predicted):\n")
        for (true_letter, pred_letter), count in confusions.most_common():
            print(f"  {true_letter} → {pred_letter}: {count} time(s)")

            # Show exactly which screenshots and cells these came from
            examples = misclassified_examples.get((true_letter, pred_letter), [])
            for fname, r, c in examples:
                print(f"       - {fname} at row {r}, col {c}")
    else:
        print("No misclassifications recorded 🎉")
