"""Fresh frozen corpus for adaptive calibrated-review consensus.

This corpus was created only after the v0.7.7 frozen validation report was saved.
It must not be revised after the first live consensus result is inspected.
"""
from __future__ import annotations

from dataclasses import replace

from .semantic_stress import SemanticCase


def adaptive_consensus_cases() -> list[SemanticCase]:
    rows: list[SemanticCase] = []

    def add(case_id, group, column, canonical, variant, expected, context=""):
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

    safe_single = [
        ("c001","procedure","Clinical Procedure Electrocardiography","Clinical Procedure Electrocardiogrphy"),
        ("c002","discipline","Laboratory Discipline Hematology","Laboratory Discipline Hematolgy"),
        ("c003","method","Laboratory Method Immunochemistry","Laboratory Method Immunochemstry"),
        ("c004","method","Laboratory Method Spectrophotometry","Laboratory Method Spectrophotmetry"),
        ("c005","analyte","Laboratory Analyte Phosphorus","Laboratory Analyte Phosphrus"),
        ("c006","analyte","Laboratory Analyte Triglyceride","Laboratory Analyte Triglycride"),
        ("c007","process","Biological Process Ventilation","Biological Process Ventilaton"),
        ("c008","process","Biological Process Perfusion","Biological Process Perfuson"),
        ("c009","procedure","Cardiac Procedure Echocardiography","Cardiac Procedure Echocardiogrphy"),
        ("c010","treatment","Pulmonary Treatment Bronchodilator","Pulmonary Treatment Bronchodilatr"),
        ("c011","discipline","Clinical Discipline Nephrology","Clinical Discipline Nephrolgy"),
        ("c012","process","Disease Process Pathogenesis","Disease Process Pathgenesis"),
        ("c013","process","Cellular Process Metabolism","Cellular Process Metablism"),
        ("c014","process","Cellular Process Apoptosis","Cellular Process Apotosis"),
        ("c015","process","Cellular Process Proliferation","Cellular Process Proliferaton"),
        ("c016","process","Cellular Process Differentiation","Cellular Process Differentation"),
        ("c017","procedure","Clinical Procedure Catheter Ablation","Clinical Procedure Catheter Ablaton"),
        ("c018","method","Laboratory Method Chromatography","Laboratory Method Chromatogrphy"),
        ("c019","method","Laboratory Method Electrophoresis","Laboratory Method Electrophoresi"),
        ("c020","measurement","Clinical Measurement Oxygenation","Clinical Measurement Oxygentation"),
    ]
    for x in safe_single:
        add(x[0],"safe_single_edit_typo",*x[1:],"accept")

    safe_multi = [
        ("c021","report","Clinical Pathology Interpretation","Clinicl Patholgy Interpretation"),
        ("c022","assay","Molecular Sequencing Assay","Moleculr Sequencng Assay"),
        ("c023","analysis","Genetic Association Analysis","Genetc Association Analysi"),
        ("c024","procedure","Diagnostic Imaging Procedure","Diagnostc Imaging Procdure"),
        ("c025","processing","Biospecimen Processing Workflow","Biospecmen Procesing Workflow"),
        ("c026","analysis","Transcriptomic Expression Analysis","Transcriptomc Expresion Analysis"),
        ("c027","analysis","Proteomic Quantification Analysis","Proteomc Quantificaton Analysis"),
        ("c028","assessment","Cardiovascular Risk Assessment","Cardiovacular Risk Assesment"),
        ("c029","procedure","Pulmonary Function Evaluation","Pulmonry Function Evaluaton"),
        ("c030","review","Histologic Specimen Review","Histologc Specmen Review"),
    ]
    for x in safe_multi:
        add(x[0],"safe_multi_edit_typo",*x[1:],"accept")

    safe_nonmedical = [
        ("c031","physics_term","Physics Electromagnetism Measurement","Physics Electromagnetsm Measurement"),
        ("c032","physics_term","Physics Thermodynamics Experiment","Physics Thermodynamcs Experiment"),
        ("c033","chemistry_term","Chemistry Stoichiometry Calculation","Chemistry Stoichiomtry Calculation"),
        ("c034","geology_term","Geology Stratigraphy Survey","Geology Stratigrphy Survey"),
        ("c035","geology_term","Geology Paleontology Dataset","Geology Paleontolgy Dataset"),
        ("c036","atmosphere_term","Atmospheric Meteorology Observation","Atmospheric Meteorolgy Observation"),
        ("c037","materials_term","Materials Magnetization Curve","Materials Magnetiztion Curve"),
        ("c038","polymer_term","Polymer Polymerization Reaction","Polymer Polymeriztion Reaction"),
        ("c039","mineral_term","Mineralogy Mineralization Process","Mineralogy Mineraliztion Process"),
        ("c040","crystal_term","Crystallography Diffraction Pattern","Crystallogrphy Diffraction Pattern"),
    ]
    for x in safe_nonmedical:
        add(x[0],"safe_nonmedical_typo",*x[1:],"accept")

    safe_formatting = [
        ("c041","sample_type","Archived Tissue Sample","ARCHIVED TISSUE SAMPLE"),
        ("c042","method","Flow Sorting Method"," Flow Sorting Method "),
        ("c043","source","Venous Blood Source","VENOUS BLOOD SOURCE"),
        ("c044","material","Fixed Cell Material"," Fixed Cell Material "),
        ("c045","status","Completed Analysis Status","COMPLETED ANALYSIS STATUS"),
    ]
    for case_id,column,canonical,variant in safe_formatting:
        add(case_id,"safe_schema_formatting",column,canonical,variant,"accept",
            "Fixture dictionary: capitalization and surrounding whitespace are presentation artifacts and do not encode category identity.")

    safe_padding = [
        ("c046","replicate","Experimental Replicate 21","Experimental Replicate 021"),
        ("c047","session","Microscopy Session 22","Microscopy Session 022"),
        ("c048","batch","Extraction Batch 23","Extraction Batch 023"),
        ("c049","plate","Assay Plate 24","Assay Plate 024"),
        ("c050","phase","Processing Phase 25","Processing Phase 025"),
    ]
    for case_id,column,canonical,variant in safe_padding:
        add(case_id,"safe_schema_padding",column,canonical,variant,"accept",
            "Fixture dictionary: the numeric suffix is an integer; leading zeros are display formatting only and both labels are explicitly equivalent.")

    unsafe_neighbors = [
        ("c051","biochemical_term","Biochemical Term Cysteine","Biochemical Term Cystine"),
        ("c052","chemical_term","Chemical Term Ketone","Chemical Term Ketene"),
        ("c053","anatomic_term","Anatomic Term Septal","Anatomic Term Sepal"),
        ("c054","anatomic_term","Anatomic Term Ileal","Anatomic Term Ideal"),
        ("c055","anatomic_term","Anatomic Term Portal","Anatomic Term Mortal"),
        ("c056","clinical_term","Clinical Term Tonic","Clinical Term Toxic"),
        ("c057","pathology_term","Pathology Term Lesion","Pathology Term Legion"),
        ("c058","anatomic_term","Anatomic Term Cortex","Anatomic Term Vortex"),
        ("c059","gastro_term","Gastrointestinal Term Gastric","Gastrointestinal Term Gastrin"),
        ("c060","anatomic_term","Anatomic Term Humeral","Anatomic Term Numeral"),
        ("c061","chemical_term","Chemical Term Purine","Chemical Term Urine"),
        ("c062","chemical_term","Chemical Term Valine","Chemical Term Saline"),
        ("c063","genetics_term","Genetics Term Allele","Genetics Term Allege"),
        ("c064","biology_term","Biology Term Clone","Biology Term Clove"),
        ("c065","laboratory_term","Laboratory Term Buffer","Laboratory Term Buffet"),
        ("c066","biology_term","Biology Term Colony","Biology Term Colon"),
        ("c067","laboratory_term","Laboratory Term Assay","Laboratory Term Essay"),
        ("c068","biology_term","Biology Term Plasma","Biology Term Plaza"),
        ("c069","oncology_term","Oncology Term Tumor","Oncology Term Tutor"),
        ("c070","oncology_term","Oncology Term Cancer","Oncology Term Canter"),
        ("c071","clinical_term","Clinical Term Serum","Clinical Term Scrum"),
        ("c072","anatomic_term","Anatomic Term Renal","Anatomic Term Penal"),
        ("c073","clinical_term","Clinical Term Venous","Clinical Term Venus"),
        ("c074","anatomic_term","Anatomic Term Colon","Anatomic Term Color"),
        ("c075","anatomic_term","Anatomic Term Medial","Anatomic Term Media"),
    ]
    for x in unsafe_neighbors:
        add(x[0],"unsafe_valid_neighbor",*x[1:],"defer")

    unsafe_status = [
        ("c076","status","Clinical Result Status Not performed","Clinical Result Status Not confirmed"),
        ("c077","status","Clinical Result Status Not measured","Clinical Result Status Not reported"),
        ("c078","status","Clinical Result Status Unavailable","Clinical Result Status Uninterpretable"),
        ("c079","status","Clinical Result Status Pending","Clinical Result Status Positive"),
        ("c080","status","Clinical Result Status Unknown","Clinical Result Status Unverified"),
        ("c081","status","Clinical Result Status Negative","Clinical Result Status Negated"),
        ("c082","status","Clinical Result Status Incomplete","Clinical Result Status Inconclusive"),
        ("c083","status","Clinical Result Status Rejected","Clinical Result Status Repeated"),
        ("c084","status","Clinical Result Status Confirmed","Clinical Result Status Conformed"),
        ("c085","status","Clinical Result Status Recorded","Clinical Result Status Recovered"),
    ]
    for x in unsafe_status:
        add(x[0],"unsafe_status_semantics",*x[1:],"defer")

    unsafe_abbrev = [
        ("c086","code","Clinical Abbreviation ANC","Clinical Abbreviation ALC"),
        ("c087","code","Clinical Abbreviation RBC","Clinical Abbreviation WBC"),
        ("c088","code","Clinical Abbreviation CSF","Clinical Abbreviation CBF"),
        ("c089","code","Clinical Abbreviation HR","Clinical Abbreviation RR"),
        ("c090","code","Clinical Abbreviation BP","Clinical Abbreviation BT"),
    ]
    for x in unsafe_abbrev:
        add(x[0],"unsafe_abbreviation",*x[1:],"defer")

    unsafe_numeric = [
        ("c091","visit","Protocol Visit Day 21","Protocol Visit Day 28"),
        ("c092","dose","Treatment Dose 4 mg","Treatment Dose 40 mg"),
        ("c093","tube","Specimen Tube 301","Specimen Tube 302"),
        ("c094","version","Analysis Version 2.3","Analysis Version 2.4"),
        ("c095","cycle","Treatment Cycle 6","Treatment Cycle 8"),
    ]
    for x in unsafe_numeric:
        add(x[0],"unsafe_numeric_identifier",*x[1:],"defer")

    unsafe_padding = [
        ("c096","source_code","Source Code 31","Source Code 031"),
        ("c097","arm_code","Protocol Arm 32","Protocol Arm 032"),
        ("c098","stream_code","Collection Stream 33","Collection Stream 033"),
        ("c099","site_code","Analysis Site 34","Analysis Site 034"),
        ("c100","rack_code","Storage Rack 35","Storage Rack 035"),
    ]
    for case_id,column,canonical,variant in unsafe_padding:
        add(case_id,"unsafe_schema_padding",column,canonical,variant,"defer",
            "Fixture dictionary: padded and unpadded forms are separately assigned source codes and must remain distinct despite equal numeric value.")

    frequencies=((3,1),(5,2),(7,1),(4,3),(8,2))
    rows=[
        replace(
            case,
            canonical_count=frequencies[i % len(frequencies)][0],
            variant_count=frequencies[i % len(frequencies)][1],
        )
        for i,case in enumerate(rows)
    ]

    if len(rows)!=100 or len({c.id for c in rows})!=100:
        raise AssertionError("Adaptive consensus corpus must contain 100 distinct cases")
    if sum(c.expected=="accept" for c in rows)!=50:
        raise AssertionError("Adaptive consensus corpus must contain 50 safe cases")
    if sum(c.expected=="defer" for c in rows)!=50:
        raise AssertionError("Adaptive consensus corpus must contain 50 unsafe cases")
    return rows
