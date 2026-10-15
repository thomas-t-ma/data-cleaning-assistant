from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO

from .actions import ActionType


@dataclass(frozen=True, slots=True)
class StressActionExpectation:
    type: ActionType
    column: str

    @property
    def key(self) -> tuple[str, str]:
        return self.type.value, self.column


@dataclass(frozen=True, slots=True)
class StressCorpusCase:
    id: str
    domain: str
    row_count: int
    dirty_csv: str
    golden_csv: str
    expected_actions: tuple[StressActionExpectation, ...]
    protected_columns: tuple[str, ...] = ()


def _csv_text(header: list[str], rows: list[list[object]]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _month_first_date_variant(value: date, index: int) -> str:
    if index % 4 == 0:
        return f"{value.month}/{value.day}/{value.year}"
    if index % 4 == 1:
        return value.isoformat()
    if index % 4 == 2:
        return value.strftime("%b %d %Y")
    return f"{value.month:02d}/{value.day:02d}/{value.year}"


def _month_first_timestamp_variant(value: datetime, index: int) -> str:
    if index % 4 == 0:
        return f"{value.month}/{value.day}/{value.year} {value:%H:%M}"
    if index % 4 == 1:
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if index % 4 == 2:
        return value.strftime("%b %d %Y %H:%M")
    return f"{value.month:02d}/{value.day:02d}/{value.year} {value:%H:%M}"


def _workforce_case(rows: int = 500) -> StressCorpusCase:
    header = [
        "Employee_ID",
        "Join_Date",
        "Department",
        "Salary",
        "Phone",
        "Notes",
    ]
    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    departments = ["Engineering", "Finance", "Operations", "Support"]
    start = date(2021, 1, 13)

    for i in range(rows):
        employee_id = f"EMP{i + 1:05d}"
        join_date = start + timedelta(days=(i * 17) % 1200)
        department = departments[i % len(departments)]
        salary = 48000 + (i % 11) * 3500
        phone = f"-1{(650000000 + i * 7919) % 900000000:09d}"
        notes = [
            "Standard employee record",
            "Eligible for annual review",
            "No additional notes",
        ][i % 3]

        dirty_department = f" {department} " if i % 9 == 0 else department
        dirty_salary: object = "N/A" if i % 17 == 0 else salary

        dirty.append(
            [
                employee_id,
                _month_first_date_variant(join_date, i),
                dirty_department,
                dirty_salary,
                phone,
                notes,
            ]
        )
        golden.append(
            [
                employee_id,
                join_date.isoformat(),
                department,
                "" if i % 17 == 0 else salary,
                phone,
                notes,
            ]
        )

    return StressCorpusCase(
        id="workforce_500",
        domain="HR/workforce",
        row_count=rows,
        dirty_csv=_csv_text(header, dirty),
        golden_csv=_csv_text(header, golden),
        expected_actions=(
            StressActionExpectation(ActionType.STANDARDIZE_DATES, "Join_Date"),
            StressActionExpectation(ActionType.TRIM_WHITESPACE, "Department"),
            StressActionExpectation(ActionType.NORMALIZE_MISSING, "Salary"),
        ),
        protected_columns=("Employee_ID", "Phone", "Notes"),
    )


def _commerce_case(rows: int = 450) -> StressCorpusCase:
    header = [
        "Order_ID",
        "Order_Timestamp",
        "SKU",
        "Status",
        "Unit_Price",
        "Customer_ID",
    ]
    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    statuses = ["New", "Packed", "Shipped", "Delivered"]
    start = datetime(2026, 1, 13, 8, 0)

    for i in range(rows):
        order_id = f"ORD{i + 1:06d}"
        moment = start + timedelta(hours=i * 5, minutes=(i * 7) % 60)
        sku = f"{(i % 97) + 1:05d}"
        status = statuses[i % len(statuses)]
        price = 25 + (i % 40) * 25
        customer_id = f"CUST{(i % 137) + 1:05d}"

        dirty_status = f" {status} " if i % 11 == 0 else status
        if i % 8 == 0:
            dirty_price: object = f"${price:,}"
        elif i % 13 == 0:
            dirty_price = f"{price:,}"
        else:
            dirty_price = price

        dirty.append(
            [
                order_id,
                _month_first_timestamp_variant(moment, i),
                sku,
                dirty_status,
                dirty_price,
                customer_id,
            ]
        )
        golden.append(
            [
                order_id,
                moment.strftime("%Y-%m-%d %H:%M:%S"),
                sku,
                status,
                price,
                customer_id,
            ]
        )

    return StressCorpusCase(
        id="commerce_450",
        domain="sales/orders",
        row_count=rows,
        dirty_csv=_csv_text(header, dirty),
        golden_csv=_csv_text(header, golden),
        expected_actions=(
            StressActionExpectation(
                ActionType.STANDARDIZE_DATES,
                "Order_Timestamp",
            ),
            StressActionExpectation(ActionType.TRIM_WHITESPACE, "Status"),
            StressActionExpectation(ActionType.COERCE_NUMERIC, "Unit_Price"),
        ),
        protected_columns=("Order_ID", "SKU", "Customer_ID"),
    )


def _survey_case(rows: int = 400) -> StressCorpusCase:
    header = [
        "Respondent_ID",
        "Submitted_At",
        "Satisfaction",
        "Comment",
        "Followup_Email",
    ]
    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    satisfaction_values = ["Satisfied", "Neutral", "Dissatisfied"]
    comments = [
        "Great service",
        "Would use again",
        "The process was clear and straightforward",
        "No additional comments",
        "Support answered my question",
    ]
    start = datetime(2026, 4, 13, 9, 0)

    for i in range(rows):
        respondent_id = f"R{i + 1:05d}"
        moment = start + timedelta(minutes=i * 23)
        satisfaction = satisfaction_values[i % len(satisfaction_values)]
        comment = comments[i % len(comments)]
        email = (
            f"user{i + 1}@example.com"
            if i % 19
            else f"invalid-email-{i + 1}"
        )
        dirty_satisfaction = (
            f" {satisfaction} "
            if i % 10 == 0
            else satisfaction
        )

        dirty.append(
            [
                respondent_id,
                _month_first_timestamp_variant(moment, i),
                dirty_satisfaction,
                comment,
                email,
            ]
        )
        golden.append(
            [
                respondent_id,
                moment.strftime("%Y-%m-%d %H:%M:%S"),
                satisfaction,
                comment,
                email,
            ]
        )

    return StressCorpusCase(
        id="survey_400",
        domain="survey/free text",
        row_count=rows,
        dirty_csv=_csv_text(header, dirty),
        golden_csv=_csv_text(header, golden),
        expected_actions=(
            StressActionExpectation(
                ActionType.STANDARDIZE_DATES,
                "Submitted_At",
            ),
            StressActionExpectation(
                ActionType.TRIM_WHITESPACE,
                "Satisfaction",
            ),
        ),
        protected_columns=(
            "Respondent_ID",
            "Comment",
            "Followup_Email",
        ),
    )


def _finance_case(rows: int = 350) -> StressCorpusCase:
    header = [
        "Transaction_ID",
        "Posting_Date",
        "Amount",
        "Memo",
        "Reference",
    ]
    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    memos = [
        "Card purchase",
        "Monthly deposit",
        "Transfer in",
        "Utility payment",
    ]
    start = date(2026, 5, 13)

    for i in range(rows):
        transaction_id = f"TXN{i + 1:06d}"
        posting_date = start + timedelta(days=i % 140)
        amount = 50 + (i % 30) * 25
        if i % 17 == 0:
            canonical_amount = -amount
            dirty_amount: object = f"({amount:,})"
        elif i % 7 == 0:
            canonical_amount = amount
            dirty_amount = f"${amount:,}"
        elif i % 11 == 0:
            canonical_amount = amount
            dirty_amount = f"{amount:,}"
        else:
            canonical_amount = amount
            dirty_amount = amount

        dirty.append(
            [
                transaction_id,
                _month_first_date_variant(posting_date, i),
                dirty_amount,
                memos[i % len(memos)],
                f"REF-{i + 1:06d}",
            ]
        )
        golden.append(
            [
                transaction_id,
                posting_date.isoformat(),
                canonical_amount,
                memos[i % len(memos)],
                f"REF-{i + 1:06d}",
            ]
        )

    return StressCorpusCase(
        id="finance_350",
        domain="finance",
        row_count=rows,
        dirty_csv=_csv_text(header, dirty),
        golden_csv=_csv_text(header, golden),
        expected_actions=(
            StressActionExpectation(
                ActionType.STANDARDIZE_DATES,
                "Posting_Date",
            ),
            StressActionExpectation(ActionType.COERCE_NUMERIC, "Amount"),
        ),
        protected_columns=("Transaction_ID", "Memo", "Reference"),
    )


def _logistics_case(rows: int = 300) -> StressCorpusCase:
    header = [
        "Shipment_ID",
        "Ship_Date",
        "Carrier",
        "Incident_Zip",
        "Status",
        "Weight_Kg",
    ]
    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    carriers = ["UPS", "FedEx", "DHL", "USPS"]
    statuses = ["Queued", "In Transit", "Delivered"]
    zip_values = ["01234", "02108", "10001", "30303", "94105"]
    start = date(2026, 7, 13)

    for i in range(rows):
        shipment_id = f"SHIP{i + 1:06d}"
        ship_date = start + timedelta(days=i % 120)
        carrier = carriers[i % len(carriers)]
        postal = zip_values[i % len(zip_values)]
        status = statuses[i % len(statuses)]
        weight = 10 + (i % 25) * 5

        dirty_status = f" {status} " if i % 12 == 0 else status

        dirty.append(
            [
                shipment_id,
                _month_first_date_variant(ship_date, i),
                carrier,
                postal,
                dirty_status,
                weight,
            ]
        )
        golden.append(
            [
                shipment_id,
                ship_date.isoformat(),
                carrier,
                postal,
                status,
                weight,
            ]
        )

    return StressCorpusCase(
        id="logistics_300",
        domain="logistics",
        row_count=rows,
        dirty_csv=_csv_text(header, dirty),
        golden_csv=_csv_text(header, golden),
        expected_actions=(
            StressActionExpectation(
                ActionType.STANDARDIZE_DATES,
                "Ship_Date",
            ),
            StressActionExpectation(ActionType.TRIM_WHITESPACE, "Status"),
        ),
        protected_columns=("Shipment_ID", "Incident_Zip"),
    )


def quantitative_stress_cases() -> list[StressCorpusCase]:
    return [
        _workforce_case(),
        _commerce_case(),
        _survey_case(),
        _finance_case(),
        _logistics_case(),
    ]
