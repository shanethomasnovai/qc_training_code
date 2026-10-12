import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def read_labels(path, metadata=False):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_filename", "image_grading"}
        if metadata:
            required |= {"patient_id", "project", "project_site"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or [])}")
        result = {}
        for row in reader:
            row = {key: (value or "").strip() for key, value in row.items() if key}
            image_id = Path(row["image_filename"].replace("\\", "/")).stem
            if not image_id or image_id in result:
                raise ValueError(f"{path}: empty or duplicate image ID: {image_id}")
            grade = float(row["image_grading"])
            if grade not in (1, 2, 3, 4, 5):
                raise ValueError(f"{path}: invalid grade for {image_id}: {grade}")
            row["image_grading"] = int(grade)
            if metadata and any(not row[key] for key in ("patient_id", "project", "project_site")):
                raise ValueError(f"{path}: missing patient/trial/site for {image_id}")
            result[image_id] = row
        return result


def choose_validation(rows, fraction, seed):
    patients = sorted({row["patient_id"] for row in rows})
    if len(patients) < 2:
        raise ValueError("At least two non-test patients are needed")
    size = max(1, min(len(patients) - 1, round(len(patients) * fraction)))
    categories = lambda row: (
        f"previousStudent:{row['previousStudent_grade']}",
        f"shane:{row['shane_grade']}",
        f"site:{row['project_site']}",
    )
    totals = Counter(category for row in rows for category in categories(row))
    rng = random.Random(seed)
    best = None
    best_score = float("inf")
    for _ in range(1000):
        selected = set(rng.sample(patients, size))
        val = Counter(category for row in rows if row["patient_id"] in selected for category in categories(row))
        score = sum(abs(val[key] / count - fraction) for key, count in totals.items())
        score += sum(100 for key, count in totals.items() if ":" in key and not key.startswith("site:") and val[key] == count)
        if score < best_score:
            best, best_score = selected, score
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-student", type=Path, required=True)
    parser.add_argument("--shane", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("/Users/shanethomas/Desktop/images"))
    parser.add_argument("--held-out", default="VISUSS")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-extension", default=".png")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared"))
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if not args.image_dir.is_dir():
        raise ValueError(f"Image folder not found: {args.image_dir}")
    previousStudent = read_labels(args.previous_student, metadata=True)
    shane = read_labels(args.shane)
    extension = "." + args.image_extension.lstrip(".").lower()
    images = {}
    for path in args.image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == extension:
            if path.stem in images:
                raise ValueError(f"Duplicate image file identifier: {path.stem}")
            images[path.stem] = path.resolve()
    test_patients = {
        row["patient_id"] for row in previousStudent.values()
        if row["project"].casefold() == args.held_out.casefold()
    }
    if not test_patients:
        projects = sorted({row["project"] for row in previousStudent.values()})
        raise ValueError(f"No trial exactly named {args.held_out!r}. Available project values: {projects}")
    rows = []
    missing_images = []
    excluded_patient_overlap = []
    for image_id in sorted(previousStudent.keys() & shane.keys()):
        first, second = previousStudent[image_id], shane[image_id]
        is_test = first["project"].casefold() == args.held_out.casefold()
        if not is_test and first["patient_id"] in test_patients:
            excluded_patient_overlap.append(image_id)
            continue
        if image_id not in images:
            missing_images.append(image_id)
            continue
        rows.append({
            "image_id": image_id,
            "image_path": str(images[image_id]),
            "patient_id": first["patient_id"],
            "project": first["project"],
            "project_site": first["project_site"],
            "previousStudent_grade": first["image_grading"],
            "shane_grade": second["image_grading"],
            "split": "test" if is_test else "pending",
        })
    if missing_images:
        raise ValueError(f"{len(missing_images)} matched images have no {extension} file. Examples: {missing_images[:10]}")
    if not any(row["split"] == "test" for row in rows):
        raise ValueError("No matched images from the held-out trial; cannot compare the two runs on this trial")
    pool = [row for row in rows if row["split"] != "test"]
    validation = choose_validation(pool, args.val_fraction, args.seed)
    for row in pool:
        row["split"] = "validation" if row["patient_id"] in validation else "train"
    groups = {split: {row["patient_id"] for row in rows if row["split"] == split} for split in ("train", "validation", "test")}
    assert not groups["train"] & groups["validation"]
    assert not (groups["train"] | groups["validation"]) & groups["test"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.csv"
    if manifest.exists():
        raise FileExistsError(f"{manifest} already exists. Reuse it, or choose another output directory")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "held_out_project": args.held_out,
        "seed": args.seed,
        "validation_patient_fraction_requested": args.val_fraction,
        "previousStudent_only_ids": sorted(previousStudent.keys() - shane.keys()),
        "shane_only_ids": sorted(shane.keys() - previousStudent.keys()),
        "excluded_non_test_images_from_test_patients": excluded_patient_overlap,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        counts = {grader: {grade: sum(row[f"{grader}_grade"] == grade for row in subset) for grade in range(1, 6)} for grader in ("previousStudent", "shane")}
        report["splits"][split] = {"images": len(subset), "patients": len(groups[split]), "grade_counts": counts, "sites": dict(Counter(row["project_site"] for row in subset))}
        print(f"{split}: {len(subset)} images, {len(groups[split])} patients; grades {counts}")
    for grader in ("previousStudent", "shane"):
        absent = [grade for grade, count in report["splits"]["train"]["grade_counts"][grader].items() if count == 0]
        if absent:
            print(f"WARNING: {grader} has no training examples for grades {absent}. Review the split report before training.")
    (args.output_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved shared split: {manifest}")


if __name__ == "__main__":
    main()
