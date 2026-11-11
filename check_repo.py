"""Final check of the repo (Member 4, D8).

It checks that every file and folder the task sheet asks for is there.

Run:  uv run check_repo.py
"""
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

EXPECTED_MEMBERS = 4
MODELS_PER_MEMBER = 10
TEMPLATE_NAME = "NameHyperparameter_experiment_results.md"   # the empty template, not a real table
NOT_MEMBER_FOLDERS = ["plots", "Test", "__pycache__"]

results = []   # one True/False for every check


def check(condition, text):
    """Print OK or PROBLEM for one check and remember the result."""
    if condition:
        print("[OK]      " + text)
    else:
        print("[PROBLEM] " + text)
    results.append(condition)


def count_empty_behaviour_cells(table_path):
    """Count the table rows whose last column (Observed behaviour) is empty."""
    empty = 0
    with open(table_path, encoding="utf-8") as file:
        for line in file:
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0].isdigit() and cells[-1] == "":   # a result row with nothing written at the end
                empty += 1
    return empty


# 1. Files and folders at the root of the repo
print("--- Root of the repo ---")
for name in ["train.py", "play.py", "README.md", "pyproject.toml", "Task allocation sheet.pdf", "dqn_model.pth"]:
    check(os.path.isfile(os.path.join(ROOT, name)), name)
plots = glob.glob(os.path.join(ROOT, "plots", "*.png"))
check(len(plots) > 0, f"plots/ has {len(plots)} picture(s)")

stray_tables = [os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*Hyperparameter_experiment_results.md"))
                if os.path.basename(p) != TEMPLATE_NAME]
check(len(stray_tables) == 0, "no results table left at the root (it belongs in the member's folder)"
      + (f" -> found: {', '.join(stray_tables)}" if stray_tables else ""))

# 2. One folder per member
print("\n--- Member folders ---")
members = [name for name in sorted(os.listdir(ROOT))
          if os.path.isdir(os.path.join(ROOT, name))
          and not name.startswith(".")
          and name not in NOT_MEMBER_FOLDERS]
check(len(members) == EXPECTED_MEMBERS, f"{len(members)} member folder(s) found, expected {EXPECTED_MEMBERS}: {members}")

total_models = 0
for member in members:
    folder = os.path.join(ROOT, member)
    models = glob.glob(os.path.join(folder, "*_model.pth"))
    tables = glob.glob(os.path.join(folder, "*Hyperparameter_experiment_results.md"))
    total_models += len(models)

    check(len(models) == MODELS_PER_MEMBER, f"{member}: {len(models)} model(s), expected {MODELS_PER_MEMBER}")
    check(len(tables) == 1, f"{member}: {len(tables)} results table(s), expected 1")
    if len(tables) == 1:
        empty = count_empty_behaviour_cells(tables[0])
        check(empty == 0, f"{member}: every 'Observed behaviour' cell is filled" + (f" ({empty} empty)" if empty else ""))

print(f"\nTotal models: {total_models} (expected {EXPECTED_MEMBERS * MODELS_PER_MEMBER})")

# 3. Summary
print()
if all(results):
    print("ALL CHECKS PASSED")
else:
    print(f"{results.count(False)} PROBLEM(S) TO FIX")
