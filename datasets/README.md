For training and testing, here is a recommended directory layout for dehazing (Dehaze), deraining (Derain), and motion deblurring (Motion Deblur) datasets. Please download the datasets and place them here accordingly.

```shell
|-- datasets
    # Dehazing (Dehaze)
    |-- Dehaze
        # Example: RESIDE
        |-- RESIDE
            |-- ITS        # train (Indoor Training Set)
                |-- hazy   # inputs (hazy images)
                |-- gt     # targets (clear images)
            |-- SOTS       # test (Synthetic Objective Testing Set)
                |-- indoor
                    |-- hazy
                    |-- gt
                |-- outdoor
                    |-- hazy
                    |-- gt

    # Deraining (Derain)
    |-- Derain
        |-- Rain100L
            |-- rainy   # inputs (rainy images)
            |-- gt      # targets (clean images)
        |-- Rain100H
            |-- rainy
            |-- gt
        |-- Rain1400   # optional extended dataset
            |-- rainy
            |-- gt

    # Motion Deblurring (Motion Deblur)
    |-- Deblur
        |-- GoPro
            |-- train
                |-- blur   # inputs (blurry images)
                |-- sharp  # targets (sharp images)
            |-- test
                |-- blur
                |-- sharp
        # Optional: REDS-Blur/REDS-like datasets
        |-- REDS
            |-- blur
            |-- sharp
```

Notes:
- The structure above is a recommendation. Always follow the dataset paths configured in your YAML files (e.g., `options/train/*.yml`, `options/test/*.yml`), typically under the keys like `dataroot_*`.
- Organize different tasks in separate folders (e.g., `Dehaze/RESIDE`, `Derain/Rain100*`, `Deblur/GoPro`) to avoid confusion.
- For custom datasets, ensure one-to-one filename alignment between inputs and targets, and specify suffixes and pairing rules correctly in your configs.

