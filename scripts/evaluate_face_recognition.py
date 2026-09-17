#!/usr/bin/env python3
"""Empirical validation of the SFace face recognition implementation."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.face_recognition import MobileFaceNetRecognizer
from src.identity_manager import IdentityManager


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def load_face_images(dataset_path):
    import cv2
    images_by_person = {}
    for entry in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, entry)
        if not os.path.isdir(person_dir):
            continue
        person_images = []
        for fname in sorted(os.listdir(person_dir)):
            fpath = os.path.join(person_dir, fname)
            if not os.path.isfile(fpath):
                continue
            img = cv2.imread(fpath)
            if img is None:
                continue
            person_images.append(img)
        if person_images:
            images_by_person[entry] = person_images
    return images_by_person


def generate_embeddings(recognizer, images):
    return [recognizer.get_embedding(img) for img in images]


def compute_pairwise_similarities(embeddings_by_person):
    import itertools
    genuine_sims, impostor_sims = [], []
    person_ids = sorted(embeddings_by_person.keys())
    for pid in person_ids:
        embs = embeddings_by_person[pid]
        for i, j in itertools.combinations(range(len(embs)), 2):
            if embs[i] is not None and embs[j] is not None:
                genuine_sims.append(cosine_similarity(embs[i], embs[j]))
    for pid_a, pid_b in itertools.combinations(person_ids, 2):
        for ea in embeddings_by_person[pid_a]:
            for eb in embeddings_by_person[pid_b]:
                if ea is not None and eb is not None:
                    impostor_sims.append(cosine_similarity(ea, eb))
    return genuine_sims, impostor_sims


def evaluate_thresholds(genuine_sims, impostor_sims, thresholds):
    results = []
    n_genuine, n_impostor = len(genuine_sims), len(impostor_sims)
    for t in thresholds:
        gar = sum(1 for s in genuine_sims if s >= t) / n_genuine if n_genuine else 0.0
        false_rejects = sum(1 for s in genuine_sims if s < t)
        iar = sum(1 for s in impostor_sims if s >= t) / n_impostor if n_impostor else 0.0
        false_accepts = sum(1 for s in impostor_sims if s >= t)
        results.append(dict(threshold=t, gar=gar, iar=iar,
                            false_accepts=false_accepts, false_rejects=false_rejects))
    return results


def print_report(genuine_sims, impostor_sims, threshold_results,
                 n_identities, n_images, current_threshold):
    """Print a concise validation report."""
    print("=" * 70)
    print("FACE RECOGNITION VALIDATION REPORT")
    print("=" * 70)
    print(f"Identities: {n_identities}, Images: {n_images}")
    print(f"Genuine pairs: {len(genuine_sims)}, Impostor pairs: {len(impostor_sims)}")
    print()

    g = np.array(genuine_sims)
    print("Genuine (Same-Person) Similarities:")
    print(f"  count:  {len(g)}")
    print(f"  min:    {g.min():.4f}")
    print(f"  max:    {g.max():.4f}")
    print(f"  mean:   {g.mean():.4f}")
    print(f"  median: {np.median(g):.4f}")
    print(f"  std:    {g.std():.4f}")
    print()

    i_arr = np.array(impostor_sims)
    print("Impostor (Different-Person) Similarities:")
    print(f"  count:  {len(i_arr)}")
    print(f"  min:    {i_arr.min():.4f}")
    print(f"  max:    {i_arr.max():.4f}")
    print(f"  mean:   {i_arr.mean():.4f}")
    print(f"  median: {np.median(i_arr):.4f}")
    print(f"  std:    {i_arr.std():.4f}")
    print()

    gap = g.min() - i_arr.max()
    print(f"Separation (genuine_min - impostor_max): {gap:.4f}")
    print()

    print("Threshold Evaluation:")
    hdr = f"{'Threshold':>10} {'GAR':>8} {'IAR':>8} {'FalseAcc':>10} {'FalseRej':>10}"
    print(hdr)
    print("-" * 50)
    for r in threshold_results:
        marker = " <-- current" if abs(r["threshold"] - current_threshold) < 0.001 else ""
        print(f"{r['threshold']:>10.2f} {r['gar']:>8.4f} {r['iar']:>8.4f} "
              f"{r['false_accepts']:>10} {r['false_rejects']:>10}{marker}")
    print()

    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    current = next((r for r in threshold_results
                    if abs(r["threshold"] - current_threshold) < 0.001), None)
    if current:
        if current["false_accepts"] > 0:
            print(f"Current threshold ({current_threshold}): UNSAFE")
            print(f"  -> {current['false_accepts']} false accepts detected")
        else:
            print(f"Current threshold ({current_threshold}): No false accepts on this dataset")

    conservative = next((r for r in reversed(threshold_results)
                         if r["false_accepts"] == 0), None)
    if conservative:
        print(f"Conservative ({conservative['threshold']:.2f}): "
              f"GAR={conservative['gar']:.4f}, 0 false accepts")

    best = min(threshold_results, key=lambda r: (r["false_accepts"], -r["gar"]))
    if best["false_accepts"] == 0:
        print(f"Recommended: >= {best['threshold']:.2f} for zero false accepts")
    else:
        print(f"Best: threshold={best['threshold']:.2f}, false_accepts={best['false_accepts']}")


def test_identity_manager(recognizer, embeddings_by_person, threshold):
    """Test the real IdentityManager with genuine and impostor queries."""
    print()
    print("=" * 70)
    print("IDENTITY MANAGER REAL-IMAGE TEST")
    print("=" * 70)

    config = dict(matching_threshold=threshold, recognition_interval=30,
                  retry_interval=15, max_per_frame=1,
                  min_face_size=40, min_brightness=20, max_brightness=235)
    id_mgr = IdentityManager(config)
    person_ids = sorted(embeddings_by_person.keys())

    for pid in person_ids:
        first_emb = embeddings_by_person[pid][0]
        if first_emb is not None:
            id_mgr.register_identity(pid, first_emb)

    gen_ok, gen_total = 0, 0
    for pid in person_ids:
        second_emb = embeddings_by_person[pid][1]
        if second_emb is not None:
            gen_total += 1
            match = id_mgr.match_embedding(second_emb)
            if match == pid:
                gen_ok += 1

    imp_ok, imp_total, false_acc = 0, 0, 0
    for idx_a, pid_a in enumerate(person_ids):
        for idx_b, pid_b in enumerate(person_ids):
            if idx_a == idx_b:
                continue
            emb_b = embeddings_by_person[pid_b][0]
            if emb_b is None:
                continue
            imp_total += 1
            match = id_mgr.match_embedding(emb_b)
            if match is None or match != pid_a:
                imp_ok += 1
            else:
                false_acc += 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate SFace face recognition with real face images")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--model", type=str,
                        default="models/face_recognition_sface_2021dec.onnx",
                        help="Path to SFace ONNX model")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    parser.add_argument("--current-threshold", type=float, default=0.4)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    rec_config = {"model_path": args.model}
    recognizer = MobileFaceNetRecognizer(rec_config)
    if not recognizer.is_available():
        print("ERROR: Model not available.")
        sys.exit(1)
    print("Model loaded successfully.\n")

    print(f"Loading dataset: {args.dataset}")
    images_by_person = load_face_images(args.dataset)
    if not images_by_person:
        print("ERROR: No images found.")
        sys.exit(1)
    n_images = sum(len(v) for v in images_by_person.values())
    print(f"Loaded {n_images} images for {len(images_by_person)} identities.\n")

    print("Generating embeddings...")
    embeddings_by_person = {}
    failures = 0
    for pid, imgs in images_by_person.items():
        embs = generate_embeddings(recognizer, imgs)
        embeddings_by_person[pid] = embs
        failures += sum(1 for e in embs if e is None)
    total_embs = sum(len(v) for v in embeddings_by_person.values())
    print(f"Generated {total_embs - failures}/{total_embs} embeddings "
          f"({failures} failures)\n")

    print("Computing pairwise similarities...")
    genuine_sims, impostor_sims = compute_pairwise_similarities(embeddings_by_person)
    print(f"Genuine pairs: {len(genuine_sims)}, Impostor pairs: {len(impostor_sims)}\n")

    threshold_results = evaluate_thresholds(genuine_sims, impostor_sims, args.thresholds)
    print_report(genuine_sims, impostor_sims, threshold_results,
                 len(images_by_person), n_images, args.current_threshold)
    test_identity_manager(recognizer, embeddings_by_person, args.current_threshold)


if __name__ == "__main__":
    main()
