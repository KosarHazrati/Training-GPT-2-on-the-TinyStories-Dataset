from pathlib import Path
from datasets import load_dataset


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    num_examples = 100
    ds = ds.select(range(num_examples))

    prompts_path = data_dir / "prompts.txt"
    refs_path = data_dir / "references.txt"

    with prompts_path.open("w", encoding="utf-8") as f_p, refs_path.open(
        "w", encoding="utf-8"
    ) as f_r:
        for ex in ds:
            full = ex["text"].strip()

            parts = full.split(".")
            first = parts[0].strip()
            if not first:
                first = full[:80]
            if not first.endswith("."):
                first += "."

            prompt = first
            reference = full.replace("\n", " ").strip()

            f_p.write(prompt + "\n")
            f_r.write(reference + "\n")

    print(f"Wrote {num_examples} prompts to {prompts_path}")
    print(f"Wrote {num_examples} references to {refs_path}")


if __name__ == "__main__":
    main()