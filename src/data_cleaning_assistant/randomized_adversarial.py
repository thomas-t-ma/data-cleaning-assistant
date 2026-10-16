from __future__ import annotations

import csv
import random
from datetime import date, datetime, timedelta
from io import StringIO

from .actions import ActionType
from .quantitative_stress import StressActionExpectation, StressCorpusCase


DEFAULT_ADVERSARIAL_SEEDS = (101, 211, 307, 401, 509)


def _csv_text(header: list[str], rows: list[list[object]]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _shuffle_pair(
    dirty: list[list[object]],
    golden: list[list[object]],
    rng: random.Random,
) -> tuple[list[list[object]], list[list[object]]]:
    order = list(range(len(dirty)))
    rng.shuffle(order)
    return [dirty[index] for index in order], [golden[index] for index in order]


def _dirty_date(value: date, index: int, rate: float, rng: random.Random) -> str:
    # The first row is deliberately directional month-first evidence.
    if index == 0:
        return f"{value.month}/{value.day}/{value.year}"
    if rng.random() >= rate:
        return value.isoformat()
    return rng.choice(
        [
            f"{value.month}/{value.day}/{value.year}",
            f"{value.month:02d}/{value.day:02d}/{value.year}",
            value.strftime("%b %d %Y"),
        ]
    )


def _dirty_timestamp(
    value: datetime,
    index: int,
    rate: float,
    rng: random.Random,
) -> str:
    if index == 0:
        return f"{value.month}/{value.day}/{value.year} {value:%H:%M}"
    if rng.random() >= rate:
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return rng.choice(
        [
            f"{value.month}/{value.day}/{value.year} {value:%H:%M}",
            f"{value.month:02d}/{value.day:02d}/{value.year} {value:%H:%M}",
            value.strftime("%b %d %Y %H:%M"),
        ]
    )


def _ensure_probability(rng: random.Random, low: float, high: float) -> float:
    return rng.uniform(low, high)


def _workforce_case(seed: int, rows: int) -> StressCorpusCase:
    rng = random.Random(seed * 1009 + 11)
    header = [
        "Employee_ID",
        "Join_Date",
        "Department",
        "Salary",
        "Phone",
        "Notes",
    ]
    departments = ["Engineering", "Finance", "Operations", "Support"]
    notes = [
        "Standard employee record",
        "Eligible for annual review",
        "No additional notes",
        "Follow up next quarter",
    ]
    date_rate = _ensure_probability(rng, 0.35, 0.85)
    whitespace_rate = _ensure_probability(rng, 0.03, 0.22)
    missing_rate = _ensure_probability(rng, 0.02, 0.10)
    start = date(2023, 1, 13)

    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    for index in range(rows):
        employee_id = f"EMP{seed:03d}{index + 1:05d}"
        join_date = start + timedelta(days=(index * 19) % 900)
        department = departments[index % len(departments)]
        salary = 46000 + (index % 13) * 2750
        phone = f"-1{(610000000 + index * 3571 + seed) % 900000000:09d}"
        note = notes[index % len(notes)]

        dirty_department = (
            f" {department} "
            if index == 0 or rng.random() < whitespace_rate
            else department
        )
        is_missing = index == 1 or rng.random() < missing_rate
        dirty_salary: object = rng.choice(["N/A", "NA", "NULL"]) if is_missing else salary

        dirty.append(
            [
                employee_id,
                _dirty_date(join_date, index, date_rate, rng),
                dirty_department,
                dirty_salary,
                phone,
                note,
            ]
        )
        golden.append(
            [
                employee_id,
                join_date.isoformat(),
                department,
                "" if is_missing else salary,
                phone,
                note,
            ]
        )

    dirty, golden = _shuffle_pair(dirty, golden, rng)
    return StressCorpusCase(
        id=f"adv_workforce_{seed}",
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


def _commerce_case(seed: int, rows: int) -> StressCorpusCase:
    rng = random.Random(seed * 1013 + 23)
    header = [
        "Order_ID",
        "Order_Timestamp",
        "SKU",
        "Status",
        "Unit_Price",
        "Customer_ID",
    ]
    statuses = ["New", "Packed", "Shipped", "Delivered"]
    timestamp_rate = _ensure_probability(rng, 0.30, 0.85)
    whitespace_rate = _ensure_probability(rng, 0.03, 0.20)
    currency_rate = _ensure_probability(rng, 0.05, 0.24)
    start = datetime(2026, 1, 13, 8, 0)

    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    for index in range(rows):
        order_id = f"ORD{seed:03d}{index + 1:05d}"
        moment = start + timedelta(hours=index * 7, minutes=(index * 11) % 60)
        sku = f"{(index * 17 + seed) % 99999:05d}"
        status = statuses[index % len(statuses)]
        price = 25 + (index % 52) * 25
        customer_id = f"CUST{(index * 13 + seed) % 5000:05d}"

        dirty_status = (
            f" {status} "
            if index == 0 or rng.random() < whitespace_rate
            else status
        )
        if index == 1 or rng.random() < currency_rate:
            dirty_price: object = rng.choice(
                [
                    f"${price:,}",
                    f"{price:,}",
                    f"${price}.00",
                ]
            )
        else:
            dirty_price = price

        dirty.append(
            [
                order_id,
                _dirty_timestamp(moment, index, timestamp_rate, rng),
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

    dirty, golden = _shuffle_pair(dirty, golden, rng)
    return StressCorpusCase(
        id=f"adv_commerce_{seed}",
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


def _survey_case(seed: int, rows: int) -> StressCorpusCase:
    rng = random.Random(seed * 1019 + 37)
    header = [
        "Respondent_ID",
        "Submitted_At",
        "Satisfaction",
        "Comment",
        "Followup_Email",
    ]
    satisfactions = ["Satisfied", "Neutral", "Dissatisfied", "Unknown"]
    comments = [
        "Great service",
        "Would use again",
        "The process was clear and straightforward",
        "No additional comments",
        "Support answered my question",
        "Great servce",
    ]
    timestamp_rate = _ensure_probability(rng, 0.30, 0.80)
    whitespace_rate = _ensure_probability(rng, 0.03, 0.18)
    bad_email_rate = _ensure_probability(rng, 0.03, 0.12)
    start = datetime(2026, 4, 13, 9, 0)

    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    for index in range(rows):
        respondent_id = f"R{seed:03d}{index + 1:05d}"
        moment = start + timedelta(minutes=index * 29)
        satisfaction = satisfactions[index % len(satisfactions)]
        comment = comments[(index * 7 + seed) % len(comments)]
        email = (
            f"invalid-email-{seed}-{index}"
            if rng.random() < bad_email_rate
            else f"user{seed}-{index}@example.com"
        )
        dirty_satisfaction = (
            f" {satisfaction} "
            if index == 0 or rng.random() < whitespace_rate
            else satisfaction
        )

        dirty.append(
            [
                respondent_id,
                _dirty_timestamp(moment, index, timestamp_rate, rng),
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

    dirty, golden = _shuffle_pair(dirty, golden, rng)
    return StressCorpusCase(
        id=f"adv_survey_{seed}",
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


def _finance_case(seed: int, rows: int) -> StressCorpusCase:
    rng = random.Random(seed * 1021 + 41)
    header = [
        "Transaction_ID",
        "Posting_Date",
        "Amount",
        "Memo",
        "Reference",
    ]
    memos = [
        "Card purchase",
        "Monthly deposit",
        "Transfer in",
        "Utility payment",
        "Card purchse",
    ]
    date_rate = _ensure_probability(rng, 0.30, 0.82)
    numeric_rate = _ensure_probability(rng, 0.06, 0.25)
    negative_rate = _ensure_probability(rng, 0.02, 0.10)
    start = date(2026, 5, 13)

    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    for index in range(rows):
        transaction_id = f"TXN{seed:03d}{index + 1:05d}"
        posting_date = start + timedelta(days=index % 150)
        amount = 50 + (index % 45) * 25
        negative = index == 0 or rng.random() < negative_rate
        canonical_amount = -amount if negative else amount

        if negative:
            dirty_amount: object = f"({amount:,})"
        elif index == 1 or rng.random() < numeric_rate:
            dirty_amount = rng.choice([f"${amount:,}", f"{amount:,}", f"${amount}.00"])
        else:
            dirty_amount = amount

        dirty.append(
            [
                transaction_id,
                _dirty_date(posting_date, index, date_rate, rng),
                dirty_amount,
                memos[(index + seed) % len(memos)],
                f"REF-{seed:03d}-{index + 1:05d}",
            ]
        )
        golden.append(
            [
                transaction_id,
                posting_date.isoformat(),
                canonical_amount,
                memos[(index + seed) % len(memos)],
                f"REF-{seed:03d}-{index + 1:05d}",
            ]
        )

    dirty, golden = _shuffle_pair(dirty, golden, rng)
    return StressCorpusCase(
        id=f"adv_finance_{seed}",
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


def _logistics_case(seed: int, rows: int) -> StressCorpusCase:
    rng = random.Random(seed * 1031 + 53)
    header = [
        "Shipment_ID",
        "Ship_Date",
        "Carrier",
        "Incident_Zip",
        "Status",
        "Weight_Kg",
    ]
    carriers = ["UPS", "FedEx", "DHL", "USPS"]
    statuses = ["Queued", "In Transit", "Delivered"]
    zip_values = ["01234", "02108", "10001", "30303", "94105"]
    date_rate = _ensure_probability(rng, 0.30, 0.80)
    whitespace_rate = _ensure_probability(rng, 0.03, 0.20)
    start = date(2026, 7, 13)

    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    for index in range(rows):
        shipment_id = f"SHIP{seed:03d}{index + 1:05d}"
        ship_date = start + timedelta(days=index % 130)
        carrier = carriers[(index + seed) % len(carriers)]
        postal = zip_values[(index * 3 + seed) % len(zip_values)]
        status = statuses[index % len(statuses)]
        weight = 10 + (index % 35) * 5
        dirty_status = (
            f" {status} "
            if index == 0 or rng.random() < whitespace_rate
            else status
        )

        dirty.append(
            [
                shipment_id,
                _dirty_date(ship_date, index, date_rate, rng),
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

    dirty, golden = _shuffle_pair(dirty, golden, rng)
    return StressCorpusCase(
        id=f"adv_logistics_{seed}",
        domain="logistics",
        row_count=rows,
        dirty_csv=_csv_text(header, dirty),
        golden_csv=_csv_text(header, golden),
        expected_actions=(
            StressActionExpectation(ActionType.STANDARDIZE_DATES, "Ship_Date"),
            StressActionExpectation(ActionType.TRIM_WHITESPACE, "Status"),
        ),
        protected_columns=("Shipment_ID", "Incident_Zip"),
    )


def _crm_case(seed: int, rows: int) -> StressCorpusCase:
    rng = random.Random(seed * 1033 + 67)
    header = [
        "Client_ID",
        "Contact_Date",
        "Job",
        "Lead_Status",
        "Email",
        "Phone",
        "Balance",
    ]
    jobs = ["management", "technician", "services", "admin"]
    statuses = ["New", "Contacted", "Qualified"]
    date_rate = _ensure_probability(rng, 0.30, 0.82)
    case_rate = _ensure_probability(rng, 0.03, 0.15)
    whitespace_rate = _ensure_probability(rng, 0.03, 0.18)
    bad_contact_rate = _ensure_probability(rng, 0.03, 0.10)
    start = date(2026, 9, 13)

    dirty: list[list[object]] = []
    golden: list[list[object]] = []
    for index in range(rows):
        client_id = f"CL{seed:03d}{index + 1:05d}"
        contact_date = start + timedelta(days=index % 120)
        job = jobs[index % len(jobs)]
        status = statuses[index % len(statuses)]
        email = (
            f"not-an-email-{index}"
            if rng.random() < bad_contact_rate
            else f"client{seed}-{index}@example.com"
        )
        phone = (
            f"-1{(404000000 + index * 1879 + seed) % 900000000:09d}"
        )
        balance = 100 + (index % 29) * 125

        case_variant = index == 0 or rng.random() < case_rate
        dirty_job = job.title() if case_variant else job
        dirty_status = (
            f" {status} "
            if index == 1 or rng.random() < whitespace_rate
            else status
        )

        dirty.append(
            [
                client_id,
                _dirty_date(contact_date, index, date_rate, rng),
                dirty_job,
                dirty_status,
                email,
                phone,
                balance,
            ]
        )
        golden.append(
            [
                client_id,
                contact_date.isoformat(),
                job,
                status,
                email,
                phone,
                balance,
            ]
        )

    dirty, golden = _shuffle_pair(dirty, golden, rng)
    return StressCorpusCase(
        id=f"adv_crm_{seed}",
        domain="CRM/contact",
        row_count=rows,
        dirty_csv=_csv_text(header, dirty),
        golden_csv=_csv_text(header, golden),
        expected_actions=(
            StressActionExpectation(ActionType.STANDARDIZE_DATES, "Contact_Date"),
            StressActionExpectation(ActionType.NORMALIZE_CATEGORIES, "Job"),
            StressActionExpectation(ActionType.TRIM_WHITESPACE, "Lead_Status"),
        ),
        protected_columns=("Client_ID", "Email", "Phone"),
    )


def randomized_adversarial_cases(
    *,
    seeds: tuple[int, ...] = DEFAULT_ADVERSARIAL_SEEDS,
    rows_per_case: int = 100,
) -> list[StressCorpusCase]:
    cases: list[StressCorpusCase] = []
    for seed in seeds:
        cases.extend(
            [
                _workforce_case(seed, rows_per_case),
                _commerce_case(seed, rows_per_case),
                _survey_case(seed, rows_per_case),
                _finance_case(seed, rows_per_case),
                _logistics_case(seed, rows_per_case),
                _crm_case(seed, rows_per_case),
            ]
        )
    return cases
