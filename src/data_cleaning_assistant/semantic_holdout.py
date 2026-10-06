"""Fresh single-edit scientific-neighbor holdout corpus.

This corpus is intentionally separate from semantic_stress.semantic_cases so prompt
experiments developed against the original forty cases can be evaluated without
changing those fixtures. Every pair is constructed to cross the fuzzy-candidate
surface and to look typo-shaped: exactly one sufficiently long nonnumeric token
differs by one character while surrounding tokens are identical.

Safe cases use an obvious nonword misspelling of a recognizable term. Risky cases
pair two valid scientific/clinical terms that are only one edit apart. Hidden
expected decisions never enter the model prompt.
"""
from __future__ import annotations

from dataclasses import replace

from .semantic_stress import SemanticCase


def semantic_neighbor_holdout_cases() -> list[SemanticCase]:
    safe_terms = [
        ("h01", "procedure", "Bronchoscopy", "Bronchscopy"),
        ("h02", "laboratory_test", "Creatinine", "Creatnine"),
        ("h03", "laboratory_test", "Hemoglobin", "Hemoglobn"),
        ("h04", "measurement", "Platelet", "Platlet"),
        ("h05", "measurement", "Neutrophil", "Neutrophl"),
        ("h06", "measurement", "Lymphocyte", "Lymphcyte"),
        ("h07", "procedure", "Echocardiogram", "Echocardiogrm"),
        ("h08", "procedure", "Colonoscopy", "Colonosopy"),
        ("h09", "assay_name", "Cytogenetics", "Cytogenetcs"),
        ("h10", "assay_name", "Immunophenotyping", "Immunophenotypng"),
        ("h11", "treatment_phase", "Maintenance", "Maintenace"),
        ("h12", "visit_type", "Surveillance", "Surveilance"),
        ("h13", "disease_status", "Metastasis", "Metastsis"),
        ("h14", "disease_status", "Recurrence", "Recurrnce"),
        ("h15", "disease_status", "Progression", "Progresion"),
        ("h16", "specimen_label", "Specimen", "Specmen"),
        ("h17", "event", "Collection", "Collecton"),
        ("h18", "site_type", "Laboratory", "Laboratry"),
        ("h19", "diagnostic_label", "Diagnosis", "Diagnois"),
        ("h20", "procedure", "Transfusion", "Transfuson"),
    ]

    risky_terms = [
        ("r01", "anatomic_term", "Ileum", "Ilium"),
        ("r02", "clinical_term", "Ileum", "Ileus"),
        ("r03", "clinical_term", "Edema", "Enema"),
        ("r04", "descriptor", "Focal", "Fecal"),
        ("r05", "anatomic_term", "Optic", "Otic"),
        ("r06", "anatomic_term", "Stoma", "Stroma"),
        ("r07", "clinical_descriptor", "Latent", "Patent"),
        ("r08", "anatomic_descriptor", "Medial", "Median"),
        ("r09", "anatomic_term", "Molar", "Malar"),
        ("r10", "movement", "Abduction", "Adduction"),
        ("r11", "muscle_function", "Abductor", "Adductor"),
        ("r12", "anatomic_descriptor", "Peroneal", "Perineal"),
        ("r13", "laboratory_term", "Lactate", "Lactase"),
        ("r14", "laboratory_term", "Amylase", "Amylose"),
        ("r15", "clinical_descriptor", "Viral", "Vital"),
        ("r16", "physiology_term", "Afferent", "Efferent"),
        ("r17", "anatomic_descriptor", "Cecal", "Fecal"),
        ("r18", "anatomic_descriptor", "Basal", "Nasal"),
        ("r19", "anatomic_descriptor", "Cardial", "Cardiac"),
        ("r20", "anatomic_descriptor", "Facial", "Fascial"),
    ]

    rows: list[SemanticCase] = []
    for case_id, column, canonical, variant in safe_terms:
        rows.append(
            SemanticCase(
                id=case_id,
                group="holdout_safe_typo",
                column=column,
                canonical=f"Clinical Term {canonical}",
                variant=f"Clinical Term {variant}",
                expected="accept",
            )
        )
    for case_id, column, canonical, variant in risky_terms:
        rows.append(
            SemanticCase(
                id=case_id,
                group="holdout_valid_neighbor",
                column=column,
                canonical=f"Clinical Term {canonical}",
                variant=f"Clinical Term {variant}",
                expected="defer",
            )
        )

    frequencies = ((3, 1), (5, 2), (7, 1), (4, 3))
    rows = [
        replace(
            case,
            canonical_count=frequencies[index % len(frequencies)][0],
            variant_count=frequencies[index % len(frequencies)][1],
        )
        for index, case in enumerate(rows)
    ]

    if len(rows) != 40 or len({case.id for case in rows}) != 40:
        raise AssertionError("Neighbor holdout must contain 40 distinct cases")
    if sum(case.expected == "accept" for case in rows) != 20:
        raise AssertionError("Neighbor holdout must contain 20 safe typo cases")
    if sum(case.expected == "defer" for case in rows) != 20:
        raise AssertionError("Neighbor holdout must contain 20 valid-neighbor deferrals")
    return rows
