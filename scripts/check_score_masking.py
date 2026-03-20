import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anticipation.vocab import CONTROL_OFFSET, REST, TIME_MASK, DUR_MASK, NOTE_MASK
from train import ALTERNATING_START, TokenizedDataset


def main():
    parser = argparse.ArgumentParser(
        description="Verify that hybrid training masks score tokens only."
    )
    parser.add_argument("dataset_file", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_prob", type=float, default=0.3)
    args = parser.parse_args()

    dataset = TokenizedDataset(
        args.dataset_file,
        onset_jitter_std=0.0,
        dur_jitter_range=0.0,
        mask_prob=args.mask_prob,
        transpose_range_semitones=0,
        tempo_scale_range=0.0,
        is_training=True,
    )

    checked_samples = min(args.samples, len(dataset))
    total_masked_triplets = 0

    for idx in range(checked_samples):
        torch.manual_seed(args.seed + idx)
        item = dataset[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]
        attention_mask = item["attention_mask"]

        if not torch.equal(attention_mask, torch.ones_like(attention_mask)):
            raise AssertionError(f"attention mask changed at sample {idx}")

        changed_positions = (input_ids != labels).nonzero(as_tuple=False).flatten().tolist()
        if len(changed_positions) % 3 != 0:
            raise AssertionError(f"non-triplet masking detected at sample {idx}: {changed_positions}")

        for start in range(0, len(changed_positions), 3):
            triplet_positions = changed_positions[start:start + 3]
            base = triplet_positions[0]
            if triplet_positions != [base, base + 1, base + 2]:
                raise AssertionError(f"non-contiguous masked triplet at sample {idx}: {triplet_positions}")
            if base < ALTERNATING_START:
                raise AssertionError(f"masked control/prefix token at sample {idx}: {triplet_positions}")

            label_triplet = labels[base:base + 3].tolist()
            input_triplet = input_ids[base:base + 3].tolist()

            if not (
                label_triplet[0] < CONTROL_OFFSET
                and label_triplet[1] < CONTROL_OFFSET
                and label_triplet[2] < CONTROL_OFFSET
                and label_triplet[2] != REST
            ):
                raise AssertionError(
                    f"masked triplet is not a score-note triplet at sample {idx}: {label_triplet}"
                )

            if input_triplet != [TIME_MASK, DUR_MASK, NOTE_MASK]:
                raise AssertionError(
                    f"masked tokens do not use score mask ids at sample {idx}: {input_triplet}"
                )

            total_masked_triplets += 1

    print(
        f"Checked {checked_samples} sample(s); verified {total_masked_triplets} masked score triplet(s)."
    )


if __name__ == "__main__":
    main()
