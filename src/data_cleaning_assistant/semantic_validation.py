"""Frozen 100-case semantic validation corpus.

This corpus is intentionally separate from the original semantic stress set and the
single-edit neighbor holdout. It is designed for one-shot comparison of already-frozen
architectures. Do not revise prompts or expected labels after inspecting live results.

Ground truth never enters model prompts. Each case is evaluated in an isolated
one-column synthetic CSV through the production profiler and candidate generator.
"""
from __future__ import annotations

from dataclasses import replace

from .semantic_stress import SemanticCase


def frozen_validation_cases() -> list[SemanticCase]:
    rows: list[SemanticCase] = []

    def add(
        case_id: str,
        group: str,
        column: str,
        canonical: str,
        variant: str,
        expected: str,
        context: str = "",
    ) -> None:
        rows.append(
            SemanticCase(
                id=case_id,
                group=group,
                column=column,
                canonical=canonical,
                variant=variant,
                expected=expected,
                context=context,
            )
        )

    # 15 fresh clinical/scientific single-edit misspellings.
    safe_single = [
        ("v001", "procedure", "Diagnostic Procedure Catheterization", "Diagnostic Procedure Catheteriztion"),
        ("v002", "method", "Laboratory Method Histopathology", "Laboratory Method Histopathlogy"),
        ("v003", "finding", "Disease Finding Thrombocytopenia", "Disease Finding Thrombocytopnia"),
        ("v004", "finding", "Clinical Finding Hypertension", "Clinical Finding Hypertenion"),
        ("v005", "imaging_method", "Imaging Method Radiography", "Imaging Method Radiogrphy"),
        ("v006", "imaging_method", "Imaging Method Ultrasonography", "Imaging Method Ultrasonogrphy"),
        ("v007", "analyte", "Laboratory Analyte Bilirubin", "Laboratory Analyte Bilirbin"),
        ("v008", "analyte", "Laboratory Analyte Albumin", "Laboratory Analyte Albumn"),
        ("v009", "analyte", "Laboratory Analyte Potassium", "Laboratory Analyte Potassum"),
        ("v010", "analyte", "Laboratory Analyte Magnesium", "Laboratory Analyte Magnesum"),
        ("v011", "measurement", "Laboratory Measure Hematocrit", "Laboratory Measure Hematcrit"),
        ("v012", "process", "Laboratory Process Coagulation", "Laboratory Process Coagulaton"),
        ("v013", "process", "Laboratory Process Centrifugation", "Laboratory Process Centrifugtion"),
        ("v014", "storage_process", "Storage Process Cryopreservation", "Storage Process Cryopreservtion"),
        ("v015", "pulmonary_test", "Pulmonary Test Spirometry", "Pulmonary Test Spiromety"),
    ]
    for case_id, column, canonical, variant in safe_single:
        add(case_id, "safe_single_edit_typo", column, canonical, variant, "accept")

    # 10 multi-error labels that should still be recognized as one intended term.
    safe_multi = [
        ("v016", "assay_name", "Serum Ferritin Assay", "Serum Feritin Asay"),
        ("v017", "review_type", "Peripheral Smear Review", "Peripherl Smear Revie"),
        ("v018", "analysis_type", "Molecular Mutation Analysis", "Moleculr Mutation Analysi"),
        ("v019", "panel_name", "Clinical Chemistry Panel", "Clinicl Chemistry Pannel"),
        ("v020", "review_type", "Radiology Image Review", "Radiolgy Image Reveiw"),
        ("v021", "processing_status", "Specimen Processing Complete", "Specmen Procesing Complete"),
        ("v022", "report_status", "Pathology Report Finalized", "Patholgy Report Finalizd"),
        ("v023", "test_name", "Respiratory Function Testing", "Respiratory Functon Testng"),
        ("v024", "analysis_type", "Genomic Variant Annotation", "Genomic Varant Anotation"),
        ("v025", "analysis_type", "Protein Expression Analysis", "Protein Expresion Analysi"),
    ]
    for case_id, column, canonical, variant in safe_multi:
        add(case_id, "safe_multi_edit_typo", column, canonical, variant, "accept")

    # 10 non-medical scientific cases to test domain generality.
    safe_nonmedical = [
        ("v026", "astronomy_term", "Astronomy Gravitational Lensing", "Astronomy Gravitatonal Lensing"),
        ("v027", "geology_term", "Geology Sedimentary Rock", "Geology Sedimentry Rock"),
        ("v028", "geophysics_term", "Geophysics Tectonic Plate", "Geophysics Tectonc Plate"),
        ("v029", "botany_term", "Botany Photosynthesis Rate", "Botany Photosynthsis Rate"),
        ("v030", "physics_term", "Physics Quantum Entanglement", "Physics Quantum Entanglemnt"),
        ("v031", "materials_term", "Materials Thermal Conductivity", "Materials Thermal Conductivty"),
        ("v032", "polymer_term", "Polymer Science Viscosity", "Polymer Science Viscosty"),
        ("v033", "chemistry_term", "Chemistry Catalytic Efficiency", "Chemistry Catalytic Effciency"),
        ("v034", "spectroscopy_term", "Spectroscopy Spectral Wavelength", "Spectroscopy Spectral Wavelngth"),
        ("v035", "crystallography_term", "Crystallography Crystallization Temperature", "Crystallography Crystalliztion Temperature"),
    ]
    for case_id, column, canonical, variant in safe_nonmedical:
        add(case_id, "safe_nonmedical_typo", column, canonical, variant, "accept")

    # 5 schema-authorized case/whitespace normalizations.
    formatting = [
        ("v036", "specimen_source", "Peripheral Vein", "PERIPHERAL VEIN",
         "Fixture dictionary: capitalization is presentation-only and does not encode the specimen category."),
        ("v037", "procedure", "Core Biopsy", " Core Biopsy ",
         "Fixture dictionary: surrounding whitespace is an export artifact and has no semantic meaning."),
        ("v038", "analyte", "Serum Calcium", "SERUM CALCIUM",
         "Fixture dictionary: capitalization is not part of the analyte code."),
        ("v039", "sample_type", "Research Sample", " Research Sample ",
         "Fixture dictionary: surrounding spaces are introduced by the export system and are non-semantic."),
        ("v040", "specimen_type", "Frozen Plasma", "FROZEN PLASMA",
         "Fixture dictionary: capitalization is formatting only for this field."),
    ]
    for case_id, column, canonical, variant, context in formatting:
        add(case_id, "safe_schema_formatting", column, canonical, variant, "accept", context)

    # 10 schema-authorized zero-padding equivalences.
    safe_padding = [
        ("v041", "cycle_label", "Cycle Number 3", "Cycle Number 03"),
        ("v042", "replicate_label", "Assay Replicate 4", "Assay Replicate 04"),
        ("v043", "session_label", "Imaging Session 5", "Imaging Session 05"),
        ("v044", "pool_label", "Specimen Pool 6", "Specimen Pool 06"),
        ("v045", "lane_label", "Sequencing Lane 7", "Sequencing Lane 07"),
        ("v046", "batch_label", "Analysis Batch 8", "Analysis Batch 08"),
        ("v047", "wave_label", "Enrollment Wave 9", "Enrollment Wave 09"),
        ("v048", "rack_label", "Storage Rack 10", "Storage Rack 010"),
        ("v049", "round_label", "Processing Round 11", "Processing Round 011"),
        ("v050", "phase_label", "Study Phase 12", "Study Phase 012"),
    ]
    for case_id, column, canonical, variant in safe_padding:
        add(
            case_id,
            "safe_schema_padding",
            column,
            canonical,
            variant,
            "accept",
            "Fixture dictionary: the suffix is stored as an integer; leading zeros are display formatting only and both strings map to the same value.",
        )

    # 20 fresh valid one-edit neighbors. Every pair is semantically distinct.
    risky_neighbors = [
        ("v051", "biochemical_term", "Biochemical Term Insulin", "Biochemical Term Inulin"),
        ("v052", "biochemical_term", "Biochemical Term Lactose", "Biochemical Term Lactase"),
        ("v053", "chemistry_term", "Chemistry Analyte Nitrate", "Chemistry Analyte Nitrite"),
        ("v054", "chemistry_term", "Chemistry Analyte Sulfate", "Chemistry Analyte Sulfite"),
        ("v055", "enzyme_name", "Enzyme Name Ligase", "Enzyme Name Lipase"),
        ("v056", "molecular_term", "Molecular Term Cytosine", "Molecular Term Cytokine"),
        ("v057", "genetics_term", "Genetics Term Exon", "Genetics Term Axon"),
        ("v058", "genetics_term", "Genetics Term Codon", "Genetics Term Colon"),
        ("v059", "science_term", "Science Term Neuron", "Science Term Neutron"),
        ("v060", "clinical_descriptor", "Clinical Descriptor Axonal", "Clinical Descriptor Agonal"),
        ("v061", "anatomic_descriptor", "Anatomic Descriptor Radial", "Anatomic Descriptor Racial"),
        ("v062", "geometry_descriptor", "Geometry Descriptor Planar", "Geometry Descriptor Plantar"),
        ("v063", "anatomic_descriptor", "Anatomic Descriptor Mesial", "Anatomic Descriptor Medial"),
        ("v064", "developmental_term", "Developmental Term Bronchial", "Developmental Term Branchial"),
        ("v065", "biology_descriptor", "Biology Descriptor Dermal", "Biology Descriptor Germal"),
        ("v066", "network_descriptor", "Network Descriptor Nodal", "Network Descriptor Modal"),
        ("v067", "laboratory_term", "Laboratory Term Stain", "Laboratory Term Strain"),
        ("v068", "discipline_term", "Discipline Term Oncology", "Discipline Term Ontology"),
        ("v069", "clinical_term", "Clinical Term Anemia", "Clinical Term Anomia"),
        ("v070", "anatomic_descriptor", "Anatomic Descriptor Oral", "Anatomic Descriptor Oval"),
    ]
    for case_id, column, canonical, variant in risky_neighbors:
        add(case_id, "unsafe_valid_neighbor", column, canonical, variant, "defer")

    # 10 ambiguous/meaningful abbreviations. Similarity does not establish aliasing.
    abbreviations = [
        ("v071", "code", "Clinical Record Abbreviation CR", "Clinical Record Abbreviation PR"),
        ("v072", "code", "Clinical Record Abbreviation SD", "Clinical Record Abbreviation PD"),
        ("v073", "code", "Clinical Record Abbreviation CT", "Clinical Record Abbreviation MR"),
        ("v074", "code", "Clinical Record Abbreviation BM", "Clinical Record Abbreviation PB"),
        ("v075", "code", "Clinical Record Abbreviation ALT", "Clinical Record Abbreviation AST"),
        ("v076", "code", "Clinical Record Abbreviation BUN", "Clinical Record Abbreviation BNP"),
        ("v077", "code", "Clinical Record Abbreviation MRI", "Clinical Record Abbreviation MRA"),
        ("v078", "code", "Clinical Record Abbreviation FISH", "Clinical Record Abbreviation ISH"),
        ("v079", "code", "Clinical Record Abbreviation PET", "Clinical Record Abbreviation PEF"),
        ("v080", "code", "Clinical Record Abbreviation AC", "Clinical Record Abbreviation TC"),
    ]
    for case_id, column, canonical, variant in abbreviations:
        add(case_id, "unsafe_abbreviation", column, canonical, variant, "defer")

    # 10 numeric identifiers/doses/timepoints where a small lexical change is meaningful.
    numeric = [
        ("v081", "visit_label", "Clinical Trial Protocol Cycle Day 7", "Clinical Trial Protocol Cycle Day 14"),
        ("v082", "dose", "Medication Dose 2 mg", "Medication Dose 20 mg"),
        ("v083", "aliquot", "Specimen Aliquot 11", "Specimen Aliquot 12"),
        ("v084", "plate", "Laboratory Plate 07", "Laboratory Plate 08"),
        ("v085", "arm", "Treatment Arm 2", "Treatment Arm 3"),
        ("v086", "timepoint", "Followup Week 4", "Followup Week 5"),
        ("v087", "tube", "Collection Tube 101", "Collection Tube 102"),
        ("v088", "batch", "Sequencing Batch 210", "Sequencing Batch 220"),
        ("v089", "panel", "Diagnostic Panel 6", "Diagnostic Panel 9"),
        ("v090", "version", "Protocol Version 1.1", "Protocol Version 1.2"),
    ]
    for case_id, column, canonical, variant in numeric:
        add(case_id, "unsafe_numeric_identifier", column, canonical, variant, "defer")

    # 5 explicit schema overrides where padding DOES encode a distinct category.
    distinct_padding = [
        ("v091", "run_label", "Sequencing Run 13", "Sequencing Run 013"),
        ("v092", "cohort_label", "Recruitment Cohort 14", "Recruitment Cohort 014"),
        ("v093", "stream_label", "Specimen Stream 15", "Specimen Stream 015"),
        ("v094", "block_label", "Imaging Block 16", "Imaging Block 016"),
        ("v095", "stratum_label", "Protocol Stratum 17", "Protocol Stratum 017"),
    ]
    for case_id, column, canonical, variant in distinct_padding:
        add(
            case_id,
            "unsafe_schema_padding",
            column,
            canonical,
            variant,
            "defer",
            "Fixture dictionary: padded and unpadded forms are separately assigned source codes and must remain distinct despite equal numeric value.",
        )

    # 5 semantically meaningful status/missingness distinctions.
    missingness = [
        ("v096", "status", "Clinical Study Result Status Code Not tested", "Clinical Study Result Status Code Not detected"),
        ("v097", "status", "Clinical Study Evaluation Status Code Not evaluable", "Clinical Study Evaluation Status Code Not available"),
        ("v098", "status", "Clinical Study Assessment Status Code Not assessed", "Clinical Study Assessment Status Code Not assigned"),
        ("v099", "status", "Clinical Study Specimen Status Code Not collected", "Clinical Study Specimen Status Code Not corrected"),
        ("v100", "status", "Clinical Study Result Status Code Undetermined", "Clinical Study Result Status Code Undetected"),
    ]
    for case_id, column, canonical, variant in missingness:
        add(case_id, "unsafe_status_semantics", column, canonical, variant, "defer")

    # Counterbalance prevalence so the model cannot equate "more common" with "correct".
    frequencies = ((3, 1), (5, 2), (7, 1), (4, 3), (8, 2))
    rows = [
        replace(
            case,
            canonical_count=frequencies[index % len(frequencies)][0],
            variant_count=frequencies[index % len(frequencies)][1],
        )
        for index, case in enumerate(rows)
    ]

    if len(rows) != 100 or len({case.id for case in rows}) != 100:
        raise AssertionError("Frozen validation corpus must contain 100 distinct cases")
    if sum(case.expected == "accept" for case in rows) != 50:
        raise AssertionError("Frozen validation corpus must contain 50 safe corrections")
    if sum(case.expected == "defer" for case in rows) != 50:
        raise AssertionError("Frozen validation corpus must contain 50 unsafe merges")
    return rows
