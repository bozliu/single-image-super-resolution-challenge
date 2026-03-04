# Data Setup

This repository does **not** ship any datasets or test images.

## Download (Mini-DIV2K)

Download the Mini-DIV2K dataset from the public link provided by the course handout:

- [Mini-DIV2K download (Dropbox)](https://www.dropbox.com/sh/e7opsbgu5ww1qe3/AAAvrcVCykCR2-G--e2H1WCxa?dl=0)

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

## Test Set (LR)

Place the 80 LR test images (named `0001.png` ... `0080.png`) here:

```text
Project 2/
  data/
    test/
      LR/          # 80 PNGs
```

The inference script reads from `data/test/LR` by default and writes outputs to `Generated upscaled images from testset/`.
