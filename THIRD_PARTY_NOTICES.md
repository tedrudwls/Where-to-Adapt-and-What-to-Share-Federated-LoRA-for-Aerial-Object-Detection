# License and third-party notices

## Project-authored material

Unless a file says otherwise, the source code and project-authored documentation in this repository are licensed under the **GNU Affero General Public License v3.0 only (`AGPL-3.0-only`)**. See [`LICENSE`](LICENSE).

Validation-selected checkpoints released by the authors for this project are also distributed under `AGPL-3.0-only`, subject to all applicable upstream rights and notices described below. The license does not grant rights in third-party datasets, source images, trademarks, or materials that the authors do not own.

## Ultralytics and RT-DETR-L

This project uses the Ultralytics software stack and an Ultralytics RT-DETR-L pretrained checkpoint. Ultralytics states that its open-source software and trained models are provided under the GNU Affero General Public License v3.0 unless a separate commercial license applies. Ultralytics and other applicable upstream rightsholders retain their respective rights in the upstream software and pretrained model materials.

The pretrained `rtdetr-l.pt` file is not currently stored in the ordinary Git repository. Its expected SHA-256 is recorded for provenance in [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md). Any redistribution or use of that pretrained file remains subject to the applicable Ultralytics terms:

- <https://www.ultralytics.com/license>
- <https://www.ultralytics.com/legal/agpl-3-0-software-license>

## AOD-4 dataset and qualitative source images

The raw AOD-4 dataset is not redistributed by this repository. It is available from Soni et al., Mendeley Data, Version 1, under **CC BY 4.0**:

- Dataset record: <https://doi.org/10.17632/cd5z895tr2.1>
- License: <https://creativecommons.org/licenses/by/4.0/>

The two unseen-helicopter qualitative figures under `assets/` contain AOD-4 source images. Those underlying images remain under CC BY 4.0; the project-added bounding boxes, labels, and composite layout do not relicense the source images.

## Other dependencies

Third-party Python packages and system dependencies retain their own licenses and copyright notices. Listing a dependency in `requirements.txt` does not relicense it under this project's license. Users and redistributors are responsible for complying with all applicable dependency and model terms.
