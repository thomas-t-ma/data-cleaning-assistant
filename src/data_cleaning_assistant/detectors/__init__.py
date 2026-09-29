from .advanced import detect_fuzzy_categories
from .basic import detect_basic_issues
from .conditional import detect_conditional_consistency
from .multiselect import detect_multiselect_component_variants
from .other_specify import detect_other_specify_consistency

__all__ = [
    "detect_basic_issues",
    "detect_conditional_consistency",
    "detect_fuzzy_categories",
    "detect_multiselect_component_variants",
    "detect_other_specify_consistency",
]
