"""pulse-qc: data-quality gating and beat-detection evaluation for pulsatile waveforms.

Public API: :func:`evaluate`, :func:`compare`, :func:`register_detector`,
:func:`register_rule`, :class:`Result`, and :class:`RuleResult` for custom rules.
"""

from pulse_qc.api import Result, compare, evaluate, register_detector, register_rule
from pulse_qc.qc.rules import RuleResult

__version__ = "0.2.0"

__all__ = [
    "Result",
    "RuleResult",
    "__version__",
    "compare",
    "evaluate",
    "register_detector",
    "register_rule",
]
