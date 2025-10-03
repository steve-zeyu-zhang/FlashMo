# HumanML3D conversion helpers

Source: https://github.com/EricGuo5513/HumanML3D

Pinned commit: `9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23`

`quaternion.py` and `skeleton.py` originate from `common/`; `param_util.py`
originates from `paramUtil.py`. `processor.py` contains the `uniform_skeleton`
and `process_file` functions extracted from `motion_representation.ipynb`, with
the notebook's HumanML3D constants and imports. The original MIT license is
included. Quaternion helpers retain their upstream Facebook attribution.

Local compatibility changes: relative imports, `np.finfo(np.float)` becomes
`np.finfo(float)`, and obsolete `np.float` casts in notebook comments become
`np.float64`. Algorithmic behavior, including the original inverse-kinematics
root convention, is retained to match that preprocessing version. The wrapper
rejects nonfinite outputs instead of silently producing evaluation features.

No motion sequences, learned evaluators, body models, or reference skeleton
assets are vendored.
