from pathlib import Path

from openpyxl import Workbook


OUTPUT = Path(__file__).with_name("messy_research_example.xlsx")

wb = Workbook()
ws = wb.active
ws.title = "Visits"
ws.append(["PatientID", "Sex", "Age", "VisitDate", "Site", "Hemoglobin", "StudyArm"])
rows = [
    ["P001", "Female", "34", "2026-01-02", "Atlanta", "11.2", "A"],
    ["P002", "female", "41", "1/5/2026", "Atlanta", "12.1", "A"],
    ["P003", " Female ", "N/A", "January 8, 2026", "Atlanta", "10.8", "B"],
    ["P004", "Male", "52", "bad-date", "Atlanta", "42.0", "B"],
    ["P005", "MALE", "49", "2026-01-12", "Atlanta", "unknown", "B"],
    ["P005", "MALE", "49", "2026-01-12", "Atlanta", "unknown", "B"],
]
for row in rows:
    ws.append(row)

subjects = wb.create_sheet("Subjects")
subjects.append(["PatientID", "DOB", "Status"])
subjects.append(["P001", "1991-04-02", "Active"])
subjects.append(["P002", "1984-03-11", "active"])
subjects.append(["P003", "NA", "Active"])

wb.save(OUTPUT)
print(f"Wrote {OUTPUT}")
