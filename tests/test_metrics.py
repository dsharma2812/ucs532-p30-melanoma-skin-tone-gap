import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import auroc, threshold_at_specificity, sensitivity, false_positive_rate


def test_auroc_hand_computed():
    # y=[0,0,1,1], s=[.1,.4,.35,.8]: pairs (pos,neg): .35>.1 yes, .35>.4 no, .8>.1 yes, .8>.4 yes -> 3/4
    assert abs(auroc([0, 0, 1, 1], [.1, .4, .35, .8]) - 0.75) < 1e-12


def test_auroc_ties_and_perfect():
    assert auroc([0, 1], [.5, .5]) == 0.5
    assert auroc([0, 0, 1, 1], [0, 1, 2, 3]) == 1.0


def test_threshold_sensitivity_fpr():
    y = np.array([0] * 10 + [1] * 4)
    s = np.array(list(range(10)) + [3, 8, 9.5, 10])
    thr = threshold_at_specificity(y, s, 0.9)      # 9 of 10 negatives at or below -> thr = 8
    assert thr == 8
    assert sensitivity(y, s, thr) == 0.5            # 9.5 and 10 exceed 8
    assert false_positive_rate(y, s, thr) == 0.1    # only the negative scoring 9


def test_shuffled_labels_auroc_near_half():
    rng = np.random.default_rng(0)
    s = rng.random(5000); y = rng.integers(0, 2, 5000)
    assert abs(auroc(y, s) - 0.5) < 0.03
