# Data Setup

This repository does **not** ship any datasets or test images.

## Dataset Sources (Train / Val / Test)

Use the challenge package as the primary source for the full workflow (train/val/test):

- [Challenge package (Dropbox)](https://www.dropbox.com/scl/fo/f88w72e55xuy5ofjup77b/AGVFqg57pq_AmZwzbg5_RnQ?rlkey=lyph59zpbdkohlqx0i0zg9suc&dl=0)

Reference source for the original DIV2K dataset:

- [Official DIV2K site](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Mapping used by this repository:

- train: 500 pairs -> `data/Mini-DIV2K/Train/{HR,LR_x4}`
- val: 80 pairs -> `data/Mini-DIV2K/Val/{HR,LR_x4}`
- test LR: 80 images (`0001.png` ... `0080.png`) -> `data/test/LR`

## Local Layout

Unzip it so the folder structure matches the paths used by the configs:

```text
Project 2/
  data/
    Mini-DIV2K/
      Train/
        HR/        # 500 PNGs
        LR_x4/     # 500 PNGs
      Val/
        HR/        # 80 PNGs
        LR_x4/     # 80 PNGs
```

The inference script reads from `data/test/LR` by default and writes outputs to `Generated upscaled images from testset/`.
