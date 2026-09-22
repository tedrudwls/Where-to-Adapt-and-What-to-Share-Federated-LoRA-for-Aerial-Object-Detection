# AOD-4 v6 dataset and client splits

## Source and scope

The experiments used the published AOD-4 v6 COCO export with four target categories: airplane, bird, drone and helicopter. Obtain the original data from [Mendeley Data, version 1](https://doi.org/10.17632/cd5z895tr2.1); this repository does not contain the images. The verified historical input contained 15,761 train, 4,514 validation and 2,241 test images. The accompanying COCO annotation counts were 22,058 / 6,369 / 3,171. Category ID `0` (`airplane-helicopter-drone-bird`) was declared as metadata but had zero annotations; the four actual classes were retained. Background images were retained.

Each image has an official split membership. The `official_aod4_v6` policy **preserves** that membership; it does not silently remove overlapping-source images. Client partitions are created separately inside train, validation and test, with the source group assigned atomically to one client within each split and ownership aligned across splits where possible. The Dirichlet seed changes those client assignments, not only the mini-batch order. Thus `(training seed, partition seed)=(42,42),(43,43),(44,44)` represents three distinct paired partition/training replicates.

## Preparing the tree

The code expects the COCO directories to be named `train`, `val`, `test`, each containing `_annotations.coco.json` and the referenced images. Example:

```text
/your/data/AOD4/Images/
  train/_annotations.coco.json
  val/_annotations.coco.json
  test/_annotations.coco.json
```

If a downloaded archive uses `valid` rather than `val`, resolve that directory name before split preparation. Verify the extracted archive and its annotations against the source before running training. The prepared YOLO-format labels and symlinks are an **intermediate input representation for Ultralytics RT-DETR**, not a YOLO detector. The original COCO metadata and pixel dimensions are checked during preparation and preflight.

```bash
export PROJECT_DIR="$(pwd)"
export DATA_ROOT=/absolute/path/to/AOD4/Images
export SPLIT_DIR="$PROJECT_DIR/data/splits"
python3 scripts/prepare_split.py \
  --data_root "$DATA_ROOT" --output_dir "$SPLIT_DIR" \
  --source_split_policy official_aod4_v6 \
  --partition dirichlet --alpha 0.4 --num_clients 3 --seed 42
```

Repeat for seeds 43 and 44; see [Reproducibility](REPRODUCIBILITY.md). A newly generated manifest embeds the actual data root, annotation/image inventory digests and generated YAML/tree digests. **Do not edit a historical JSON or dataset YAML to replace an absolute path**: doing so breaks integrity checks. Regenerate into a fresh output directory on a different host, then compare policy, counts, class distributions, and other invariant metadata. The compact [historical split summary](../provenance/splits/summary.json) and full-manifest SHA-256 inventory are provenance records; the complete historical manifest bundle is not yet a public artifact.

## Documented source overlap

The unmodified official export contains **281** Roboflow source-key/hash-connected components that span at least two official splits, with **two** exact cross-split image SHA-256 collision signals. A component is a set of records linked by source key or exact hash; 281 is **not** a count of duplicates or images to delete. Similar source-key names may represent related variants or video frames, but that possibility does not establish source-disjoint evaluation. The official-split test AP remains a valid description of performance on that published split; it is **not** evidence of generalization to unseen source sequences. No source-clean common-test detection AP is claimed in this repository.

The source inventory and split-manifest hashes were verified after recovery against the historical experiment input: all **22,516** image SHA-256 values and all three COCO annotation files matched. The original dataset itself is not a project-created release asset.
