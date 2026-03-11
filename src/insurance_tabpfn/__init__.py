"""
insurance-tabpfn: Foundation model wrapper for thin-data insurance pricing segments.

This library wraps TabPFN v2 and TabICLv2 with insurance-specific workflow:
exposure handling, GLM benchmark comparison, PDP-based relativities extraction,
and committee paper generation.

Use case: small books, new products, schemes with <5,000 policies where you
don't have enough data for a stable GLM but still need defensible relativities.

Quick start:
    from insurance_tabpfn import InsuranceTabPFN

    model = InsuranceTabPFN(backend="auto")
    model.fit(X_train, y_train, exposure=exposure_train)
    rates = model.predict(X_test, exposure=exposure_test)
"""

from insurance_tabpfn.model import InsuranceTabPFN
from insurance_tabpfn.benchmark import GLMBenchmark, BenchmarkResult
from insurance_tabpfn.relativities import RelativitiesExtractor
from insurance_tabpfn.report import CommitteeReport
from insurance_tabpfn.validators import validate_inputs, ThinSegmentWarning

__version__ = "0.1.0"

__all__ = [
    "InsuranceTabPFN",
    "GLMBenchmark",
    "BenchmarkResult",
    "RelativitiesExtractor",
    "CommitteeReport",
    "validate_inputs",
    "ThinSegmentWarning",
    "__version__",
]
