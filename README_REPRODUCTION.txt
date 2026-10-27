CORRECTED NUMERICAL REPRODUCTION CODES

Contents
--------
verification.py
    Recomputes the finite-budget, asymptotic, multistage, multi-density,
    sampling, forward-reduction, inventory, and Moreau-envelope results.

major_verification.py
    Recomputes the Markov, backward-reduction, larger-budget scaling,
    real-data, and independent deterministic-equivalent LP audits.

data/macrodata_realcons.csv
    Required realcons input for the real-data experiment.

Corrections in this version
---------------------------
1. The multi-density W1 ratio denominator uses the exact tree's actual
   conditional-mean cell probabilities. It no longer substitutes
   nearest-representative probabilities.
2. Exact-tree construction timing stops before the W1 denominator evaluation.
3. The duplicate application transport helper in major_verification.py also
   uses the actual supplied probabilities.

Reproduction
------------
Run from the extracted directory, in this order:

  python3 -W error verification.py
  python3 -W error major_verification.py

Do not set COR_USE_CACHE or MAJOR_USE_CACHE for a fresh numerical reproduction.
The scripts require Python 3, NumPy, SciPy, and Matplotlib. They create the
sections/ and figures/ directories and regenerate LaTeX tables and PDF/EPS
figures. Both scripts stop on failed assertions. Deterministic numerical values
and random seeds are fixed; runtime columns depend on the machine.

The scripts require no credentials, downloads, or external data services.
