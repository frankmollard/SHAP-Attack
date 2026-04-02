"""
shap_attack.py
--------------
SHAP-guided adversarial evasion attack.

Based on:
  Mollard, Becker, Röhrbein (2024).
  "Adversarial Evasion Attacks on Computer Vision using SHAP Values."
  DOI: 10.13140/RG.2.2.28762.40647
"""

import numpy as np


def shapAttack(x, shapVal, strength=60, t_shap=0.001, t_x=0.0, mid_clip=0.1, up=True):
    """
    Compute a SHAP-guided adversarial perturbation for a single image.

    The attack targets pixels whose SHAP attributions are large in magnitude,
    pushing them toward a neutral range where they no longer support the
    model's original prediction.

    Parameters
    ----------
    x        : np.ndarray  — flattened image, pixel values in [0, 1]
    shapVal  : np.ndarray  — SHAP values for x (same shape)
    strength : float       — overall perturbation strength (scales attack_vector)
    t_shap   : float       — SHAP magnitude threshold below which pixels are
                             left untouched. Recommended: std(shap_values) * 2
    t_x      : float       — pixel-value margin around 0.5 defining the
                             "already neutral" zone; pixels within
                             [0.5 - t_x, 0.5 + t_x] are skipped
    mid_clip : float       — prevents overshooting the neutral midpoint (0.5)
    up       : bool        — if True, apply the attack bidirectionally
                             (low-value pixels with negative attribution
                             are also pushed toward neutral)

    Returns
    -------
    np.ndarray
        Perturbation array (same shape as x).
        Apply with: adversarial = np.clip(x + perturbation, 0, 1)
    """
    t_shap = t_shap * strength / 10
    attack_vector = shapVal * strength

    # --- Right branch: high pixel values with strong positive attribution ---
    # These pixels push the prediction upward; neutralise them by decreasing value.
    n_upper = np.where(
        (attack_vector > t_shap) & (x > 0.5 + t_x),
        -attack_vector,
        0,
    )
    right = np.where(
        (n_upper != 0) & (x + n_upper < 0.5 - mid_clip),
        x - 0.5 - mid_clip,
        n_upper,
    )

    # --- Left branch: low pixel values with strong negative attribution ---
    # These pixels push the prediction downward; neutralise them by increasing value.
    if up:
        n_lower = np.where(
            (attack_vector < -t_shap) & (x < 0.5 - t_x),
            -attack_vector,
            0,
        )
        left = np.where(
            (n_lower != 0) & (x + n_lower > 0.5 + mid_clip),
            x - 0.5 + mid_clip,
            n_lower,
        )
    else:
        left = 0

    return right + left


def run_attack(images, shap_values, classifier, strength=60, verbose=True):
    """
    Attack a batch of images and report misclassification statistics.

    Parameters
    ----------
    images       : np.ndarray, shape (N, H*W*C)   — normalised images [0, 1]
    shap_values  : np.ndarray, shape (N, H*W*C)   — precomputed SHAP values
    classifier   : Keras model                    — trained CNN
    strength     : float                          — attack strength
    verbose      : bool                           — print per-image results

    Returns
    -------
    dict with keys:
        'misclassified'  : int  — number of images whose prediction flipped
        'perturbations'  : list — perturbation arrays for each image
        'adversarials'   : list — clipped adversarial images
    """
    t_shap = np.std(shap_values) * 2
    misclassified = 0
    perturbations = []
    adversarials = []

    for i, (s, x) in enumerate(zip(shap_values, images)):
        at = shapAttack(x, s.T, t_shap=t_shap, strength=strength)
        attacked = np.clip(x + at, 0, 1)

        pred_orig = classifier.predict(x.reshape(1, -1), verbose=0)[0][0]
        pred_atk  = classifier.predict(attacked.reshape(1, -1), verbose=0)[0][0]

        # Original label: cat if pred >= 0.5, dog otherwise
        flipped = (pred_orig >= 0.5) != (pred_atk >= 0.5)
        if flipped:
            misclassified += 1

        perturbations.append(at)
        adversarials.append(attacked)

        if verbose:
            status = "FLIPPED" if flipped else "stable"
            print(f"  [{i:3d}] orig={pred_orig:.3f}  atk={pred_atk:.3f}  {status}")

    total = len(images)
    print(f"\n{misclassified}/{total} images misclassified after attack "
          f"({100 * misclassified / total:.1f}%)")

    return {
        "misclassified": misclassified,
        "perturbations": perturbations,
        "adversarials": adversarials,
    }
