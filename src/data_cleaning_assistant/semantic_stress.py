"""Fixed synthetic semantic corpus. Ground truth NEVER enters the model prompt.

Each case is evaluated in its own one-column CSV, preventing unrelated pairs
from influencing detector or model decisions. "Defer" means preserve distinct
labels, not that two codes have identical meaning in every clinical context.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SemanticCase:
    id: str
    group: str
    column: str
    canonical: str
    variant: str
    expected: str
    context: str = ""


def semantic_cases() -> list[SemanticCase]:
    # All labels are synthetic; study-specific notes are fictional fixture
    # metadata deliberately supplied as evidence (NOT hidden scoring answers).
    rows = [
        # Obvious word typos: longer full labels make lexical detection realistic.
        ("s01", "typo", "specimen_source", "Peripheral Blood", "Peripheral Blod", "accept", ""),
        ("s02", "typo", "specimen_source", "Bone Marrow Aspirate", "Bone Marrow Aspirte", "accept", ""),
        ("s03", "typo", "response", "Complete Response", "Complete Respons", "accept", ""),
        ("s04", "typo", "response", "Partial Response", "Partial Respons", "accept", ""),
        ("s05", "typo", "visit_label", "Initial Screening Visit", "Initial Screeing Visit", "accept", ""),
        ("s06", "typo", "visit_label", "Followup Appointment", "Followup Appointmnt", "accept", ""),
        ("s07", "typo", "assay_name", "Serum Immunofixation", "Serum Immunofixaton", "accept", ""),
        ("s08", "typo", "assay_name", "Flow Cytometry", "Flow Cytometery", "accept", ""),
        ("s09", "typo", "measurement", "Monoclonal Protein", "Monoclonal Protien", "accept", ""),
        ("s10", "typo", "event", "Treatment Initiation", "Treatment Intiation", "accept", ""),
        ("s11", "typo", "event", "Relapse Evaluation", "Relapse Evaluaton", "accept", ""),
        ("s12", "typo", "event", "Adverse Event", "Adverse Evnt", "accept", ""),
        ("s13", "typo", "measurement", "White Blood Cell Count", "White Blood Cel Count", "accept", ""),
        ("s14", "typo", "measurement", "Red Blood Cell Count", "Red Blood Cel Count", "accept", ""),
        ("s15", "typo", "measurement", "ECOG Performance Status", "ECOG Performnce Status", "accept", ""),
        # Identical-looking codes can be equivalent or distinct depending on
        # *documented* fixture metadata. The model sees only that metadata.
        ("s16", "contextual_padding", "visit_label", "Baseline Visit 1", "Baseline Visit 01",
         "accept", "Fixture dictionary: visit index is stored as an integer; leading zeros are display formatting only."),
        ("s17", "contextual_padding", "visit_label", "Followup Visit 2", "Followup Visit 02",
         "accept", "Fixture dictionary: both labels use the same integer visit index; zero padding has no coding significance."),
        ("s18", "contextual_padding", "cohort_label", "Study Cohort 1", "Study Cohort 01",
         "accept", "Fixture dictionary explicitly maps zero-padded cohort text to the same integer cohort index."),
        ("s19", "formatting", "specimen_source", "Peripheral Blood", "PERIPHERAL BLOOD",
         "accept", "Fixture dictionary states capitalization is not part of the specimen category code."),
        ("s20", "formatting", "specimen_source", "Peripheral Blood", " Peripheral Blood ",
         "accept", "Fixture dictionary states surrounding spaces are export artifacts, not category codes."),
        # Negative controls: strong lexical similarity is not scientific
        # equivalence. Some short/dissimilar pairs may be screened out by the
        # deterministic detector; those are reported as NOT EXPOSED.
        ("n01", "numeric_code", "assay_label", "Synthetic assay control 101",
         "Synthetic assay control 102", "defer", ""),
        ("n02", "dose", "treatment", "Treatment Dose 5 mg",
         "Treatment Dose 50 mg", "defer", ""),
        ("n03", "stage", "stage", "Disease Stage II", "Disease Stage III", "defer", ""),
        ("n04", "marker", "marker", "CD38", "CD138", "defer", ""),
        ("n05", "marker", "marker", "Kappa", "Lambda", "defer", ""),
        ("n06", "missingness", "response", "Unknown", "Not applicable", "defer", ""),
        ("n07", "numeric_code", "assay_label", "Synthetic test lot 17",
         "Synthetic test lot 18", "defer", ""),
        ("n08", "numeric_code", "specimen_code", "Specimen Panel 04",
         "Specimen Panel 05", "defer", ""),
        ("n09", "numeric_code", "visit_label", "Visit Day 1",
         "Visit Day 10", "defer", ""),
        ("n10", "numeric_code", "visit_label", "Visit Day 1",
         "Visit Day 2", "defer", ""),
        ("n11", "dose", "treatment", "Synthetic Dose 10 mg",
         "Synthetic Dose 100 mg", "defer", ""),
        ("n12", "numeric_code", "assay_label", "Synthetic assay control 210",
         "Synthetic assay control 211", "defer", ""),
        ("n13", "numeric_code", "batch_label", "Batch Reference 001",
         "Batch Reference 002", "defer", ""),
        ("n14", "marker", "marker", "CD4", "CD45", "defer", ""),
        ("n15", "outcome", "outcome", "Disease Progression",
         "Disease Regression", "defer", ""),
        ("n16", "timepoint", "timepoint", "Baseline Assessment",
         "Relapse Assessment", "defer", ""),
        ("n17", "numeric_code", "sample_code", "Sample Control A1",
         "Sample Control B1", "defer", ""),
        ("n18", "contextual_padding", "visit_label", "Baseline Visit 1",
         "Baseline Visit 01", "defer",
         "Fixture dictionary: these text codes are separately allocated protocol-arm labels; leading zeros distinguish their source-arm codes."),
        ("n19", "contextual_padding", "cohort_label", "Study Cohort 1",
         "Study Cohort 01", "defer",
         "Fixture dictionary: padded and unpadded strings are distinct recruitment-stratum codes, despite sharing numeric value."),
        ("n20", "contextual_padding", "visit_label", "Followup Visit 2",
         "Followup Visit 02", "defer",
         "Fixture dictionary: padded and unpadded labels are different specimen collection streams."),
    ]
    corpus = [SemanticCase(*row) for row in rows]
    if len(corpus) != 40 or len({case.id for case in corpus}) != 40:
        raise AssertionError("Semantic corpus must have 40 distinct cases")
    if sum(c.expected == "accept" for c in corpus) != 20:
        raise AssertionError("Semantic corpus must contain 20 safe corrections")
    if sum(c.expected == "defer" for c in corpus) != 20:
        raise AssertionError("Semantic corpus must contain 20 unsafe merges")
    return corpus
