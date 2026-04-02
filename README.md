# Adversarial Evasion Attacks on Computer Vision using SHAP Values

This repository provides a minimal, reproducible implementation of the SHAP-based adversarial evasion attack described in:

> **Adversarial Evasion Attacks on Computer Vision using SHAP Values**  
> Frank Mollard, Marcus Becker, Florian Röhrbein (2024)  
> DOI: [10.13140/RG.2.2.28762.40647](https://www.researchgate.net/doi/10.13140/RG.2.2.28762.40647)

---

## Overview

Adversarial examples expose a fundamental vulnerability in deep learning models: small, carefully crafted perturbations can cause a classifier to fail completely — while the modified image looks identical to a human.

This work proposes a **white-box evasion attack** that uses **SHAP values** (Shapley Additive Explanations) instead of raw input gradients. The key insight is that pixels with large positive or negative SHAP attributions are the ones most responsible for the model's prediction. By nudging those pixels toward a neutral attribution range, the attack disrupts the model's decision without visibly altering the image.

A distinctive empirical finding: the relationship between a pixel's value and its SHAP attribution forms a **butterfly pattern** — contributions can be positive, negative, or neutral, and are not monotonically related to pixel brightness. The attack explicitly exploits this symmetry.

---

## Repository Structure

```
shap-attacks/
├── README.md
├── requirements.txt
├── shap_attack.py          # Core attack function (shapAttack)
├── model.py                # CNN classifier definition and training utilities
├── data_utils.py           # Image loading and preprocessing helpers
└── notebook/
    └── shap-attacks.ipynb  # Full walkthrough (Kaggle-compatible)
```

---

## Core Attack Function

The central contribution is the `shapAttack` function in `shap_attack.py`:

```python
import numpy as np

def shapAttack(x, shapVal, strength=60, t_shap=0.001, t_x=0.0, mid_clip=0.1, up=True):
    """
    SHAP-guided adversarial perturbation.

    Parameters
    ----------
    x        : np.ndarray  — original image, pixel values in [0, 1]
    shapVal  : np.ndarray  — SHAP values for the image (same shape as x)
    strength : float       — scales the magnitude of the perturbation
    t_shap   : float       — SHAP threshold; recommended: std(shap_values) * 2
    t_x      : float       — pixel-value margin around 0.5 (neutral zone)
    mid_clip : float       — clipping margin to prevent overshooting
    up       : bool        — if True, also attack low-valued pixels with
                             negative attributions (bidirectional attack)

    Returns
    -------
    np.ndarray  — perturbation vector (add to x, then clip to [0, 1])
    """
    t_shap = t_shap * strength / 10
    attack_vector = shapVal * strength

    # High pixel values with strong positive attribution → push toward neutral
    n_upper = np.where((attack_vector > t_shap) & (x > 0.5 + t_x), -attack_vector, 0)
    right = np.where((n_upper != 0) & (x + n_upper < 0.5 - mid_clip),
                     x - 0.5 - mid_clip, n_upper)

    if up:
        # Low pixel values with strong negative attribution → push toward neutral
        n_lower = np.where((attack_vector < -t_shap) & (x < 0.5 - t_x), -attack_vector, 0)
        left = np.where((n_lower != 0) & (x + n_lower > 0.5 + mid_clip),
                        x - 0.5 + mid_clip, n_lower)
    else:
        left = 0

    return right + left
```

Apply the attack:

```python
perturbation = shapAttack(image, shap_values.T, t_shap=np.std(shap_values) * 2)
adversarial  = np.clip(image + perturbation, 0, 1)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

The notebook uses the [Animals and Humans dataset](https://www.kaggle.com/datasets/frankmollard/animals-and-humans) from Kaggle. Download and set the path:

```python
data_path = "/path/to/animals-and-humans"
```

### 3. Run the notebook

```bash
jupyter notebook notebook/shap-attacks.ipynb
```

Or run the full pipeline from the command line:

```bash
python model.py        # train the CNN classifier
python shap_attack.py  # run the attack on 100 test images
```

---

## Requirements

```
tensorflow>=2.10
shap
numpy
pandas
matplotlib
Pillow
scikit-learn
jupyter
```

---

## Results

On a binary cat/dog classifier (CNN, ~3 conv layers, trained with cross-entropy):

- Attacking **100 cat images** with SHAP-guided perturbations causes the majority to be misclassified as dogs.
- The perturbed images are **visually indistinguishable** from the originals to human observers.
- The butterfly scatter pattern (SHAP value vs. pixel value) confirms that only a subset of pixels carry strong attributions — the attack is highly targeted.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{https://doi.org/10.13140/rg.2.2.28762.40647,
  doi       = {10.13140/RG.2.2.28762.40647},
  url       = {https://www.researchgate.net/doi/10.13140/RG.2.2.28762.40647},
  author    = {Mollard, Frank and Becker, Marcus and Röhrbein, Florian},
  language  = {en},
  title     = {Adversarial Evasion Attacks on Computer Vision using SHAP Values},
  publisher = {Unpublished},
  year      = {2024}
}
```

An arXiv preprint is also available:

```bibtex
@misc{mollard2026adversarial,
  title         = {Adversarial Evasion Attacks on Computer Vision using SHAP Values},
  author        = {Frank Mollard and Marcus Becker and Florian Roehrbein},
  year          = {2026},
  eprint        = {2601.10587},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2601.10587}
}
```

---

## License

MIT
