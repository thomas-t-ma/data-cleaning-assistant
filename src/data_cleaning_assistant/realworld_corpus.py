from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RealWorldCorpusCase:
    id: str
    domain: str
    description: str
    source_inspiration: str
    source_url: str
    dirty_csv: str
    golden_csv: str
    expected_review_issues: dict[str, tuple[str, ...]] = field(default_factory=dict)
    forbidden_action_columns: tuple[str, ...] = ()


def realworld_corpus_cases() -> list[RealWorldCorpusCase]:
    """Deterministic source-inspired golden pairs for general-purpose validation.

    The schemas and value shapes are inspired by public datasets, but the rows
    are generated specifically for this repository so ground truth is
    unambiguous and no third-party records need to be redistributed.
    """
    return [
        RealWorldCorpusCase(
            id="retail_orders",
            domain="sales/orders",
            description=(
                "Online-retail style transactions with mixed timestamp formatting, "
                "currency-formatted prices, surrounding whitespace, and a duplicate row."
            ),
            source_inspiration="UCI Online Retail / Online Retail II",
            source_url="https://archive.ics.uci.edu/dataset/352/online+retail",
            dirty_csv=(
                "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
                "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/13/2025 08:26,2.55,17850, United Kingdom \n"
                "536366,71053,WHITE METAL LANTERN,6,2025-12-13 09:15:00,£3.39,17850,United Kingdom\n"
                "536367,84406B,CREAM CUPID HEARTS COAT HANGER,8,Dec 13 2025 10:30,2.75,13047,United Kingdom\n"
                "536368,84029G,KNITTED UNION FLAG HOT WATER BOTTLE,6,12/14/2025 11:05,3.39,13047,United Kingdom\n"
                "536369,84029E,RED WOOLLY HOTTIE WHITE HEART.,6,12/15/2025 11:08,£3.39,13047,United Kingdom\n"
                "536369,84029E,RED WOOLLY HOTTIE WHITE HEART.,6,12/15/2025 11:08,£3.39,13047,United Kingdom\n"
            ),
            golden_csv=(
                "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
                "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2025-12-13 08:26:00,2.55,17850,United Kingdom\n"
                "536366,71053,WHITE METAL LANTERN,6,2025-12-13 09:15:00,3.39,17850,United Kingdom\n"
                "536367,84406B,CREAM CUPID HEARTS COAT HANGER,8,2025-12-13 10:30:00,2.75,13047,United Kingdom\n"
                "536368,84029G,KNITTED UNION FLAG HOT WATER BOTTLE,6,2025-12-14 11:05:00,3.39,13047,United Kingdom\n"
                "536369,84029E,RED WOOLLY HOTTIE WHITE HEART.,6,2025-12-15 11:08:00,3.39,13047,United Kingdom\n"
            ),
            forbidden_action_columns=("StockCode", "CustomerID"),
        ),
        RealWorldCorpusCase(
            id="survey_feedback",
            domain="survey/free text",
            description=(
                "Feedback export with mixed submission timestamps, a whitespace-only "
                "categorical defect, repeated prose, and one malformed email."
            ),
            source_inspiration="NYC 311-style free-response/service-request fields",
            source_url="https://data.cityofnewyork.us/resource/erm2-nwe9",
            dirty_csv=(
                "Respondent_ID,Submitted_At,Satisfaction,Comment,Followup_Email\n"
                "R001,10/13/2026 09:05,Satisfied,Great service,alice@example.com\n"
                "R002,2026-10-13 09:10:00, Satisfied ,Great service,bob@example.com\n"
                "R003,Oct 13 2026 09:12,Dissatisfied,Great servce,not-an-email\n"
                "R004,10/14/2026 09:15,Satisfied,Great service,dana@example.com\n"
                "R005,10/15/2026 09:20,Satisfied,Would use again,eli@example.com\n"
            ),
            golden_csv=(
                "Respondent_ID,Submitted_At,Satisfaction,Comment,Followup_Email\n"
                "R001,2026-10-13 09:05:00,Satisfied,Great service,alice@example.com\n"
                "R002,2026-10-13 09:10:00,Satisfied,Great service,bob@example.com\n"
                "R003,2026-10-13 09:12:00,Dissatisfied,Great servce,not-an-email\n"
                "R004,2026-10-14 09:15:00,Satisfied,Great service,dana@example.com\n"
                "R005,2026-10-15 09:20:00,Satisfied,Would use again,eli@example.com\n"
            ),
            expected_review_issues={"Followup_Email": ("email_format_anomaly",)},
            forbidden_action_columns=("Respondent_ID", "Comment", "Followup_Email"),
        ),
        RealWorldCorpusCase(
            id="wholesale_inventory",
            domain="inventory/wholesale",
            description=(
                "Wholesale-style spending snapshot with a protected zero-padded SKU, "
                "mixed date formatting, whitespace/case category variants, and N/A."
            ),
            source_inspiration="UCI Wholesale Customers",
            source_url="https://archive.ics.uci.edu/dataset/292/wholesale+customers",
            dirty_csv=(
                "SKU,Snapshot_Date,Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen\n"
                "00123,9/30/2026,Retail,Other,12669,9656,7561,2149,2674,1338\n"
                "00456,2026-09-30, Retail ,Lisbon,7057,9810,9568,N/A,3293,1776\n"
                "00789,09/30/2026,Horeca,Oporto,6353,8808,7684,2405,3516,7844\n"
                "00999,2026-09-30,horeca,Oporto,13265,1196,4221,6404,507,1788\n"
                "00111,Sep 30 2026,Horeca,Other,22615,5410,7198,3915,1777,5185\n"
            ),
            golden_csv=(
                "SKU,Snapshot_Date,Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen\n"
                "00123,2026-09-30,Retail,Other,12669,9656,7561,2149,2674,1338\n"
                "00456,2026-09-30,Retail,Lisbon,7057,9810,9568,,3293,1776\n"
                "00789,2026-09-30,Horeca,Oporto,6353,8808,7684,2405,3516,7844\n"
                "00999,2026-09-30,Horeca,Oporto,13265,1196,4221,6404,507,1788\n"
                "00111,2026-09-30,Horeca,Other,22615,5410,7198,3915,1777,5185\n"
            ),
            forbidden_action_columns=("SKU",),
        ),
        RealWorldCorpusCase(
            id="finance_statements",
            domain="finance",
            description=(
                "Account statement export with currency symbols, accounting negatives, "
                "mixed dates, and memo text that must remain prose."
            ),
            source_inspiration="Common bank/accounting CSV exports",
            source_url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
            dirty_csv=(
                "Account_ID,Statement_Date,Amount,Balance,Memo,Reference\n"
                "A001,10/13/2026,$1250.00,4300.50,Monthly deposit,REF-0001\n"
                "A002,2026-10-13,1250,5550.50,Monthly deposit,REF-0002\n"
                "A003,Oct 14 2026,(85.50),5465.00,Card purchase,REF-0003\n"
                "A004,10/14/2026,$42.00,5423.00,Card purchse,REF-0004\n"
                "A005,10/15/2026,500.00,5923.00,Transfer in,REF-0005\n"
                "A006,10/16/2026,75.25,5998.25,Transfer in,REF-0006\n"
            ),
            golden_csv=(
                "Account_ID,Statement_Date,Amount,Balance,Memo,Reference\n"
                "A001,2026-10-13,1250,4300.50,Monthly deposit,REF-0001\n"
                "A002,2026-10-13,1250.00,5550.50,Monthly deposit,REF-0002\n"
                "A003,2026-10-14,-85.5,5465.00,Card purchase,REF-0003\n"
                "A004,2026-10-14,42,5423.00,Card purchse,REF-0004\n"
                "A005,2026-10-15,500,5923.00,Transfer in,REF-0005\n"
                "A006,2026-10-16,75.25,5998.25,Transfer in,REF-0006\n"
            ),
            forbidden_action_columns=("Account_ID", "Memo", "Reference"),
        ),
        RealWorldCorpusCase(
            id="crm_contacts",
            domain="CRM/contact",
            description=(
                "CRM export with mixed contact dates, category whitespace/case drift, "
                "and intentionally malformed contact identifiers that must be review-only."
            ),
            source_inspiration="UCI Bank Marketing",
            source_url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
            dirty_csv=(
                "Client_ID,Contact_Date,Job,Marital,Balance,Email,Phone,Lead_Status\n"
                "C001,10/13/2026,management,married,2143,alice@example.com,-1651623197, New \n"
                "C002,2026-10-13, management ,single,29,bob@example.com,-1898471390,Qualified\n"
                "C003,Oct 14 2026,technician,married,2,not-an-email,-5596363211,Contacted\n"
                "C004,10/14/2026,Management,divorced,1506,dana@example.com,-3476490784,Qualified\n"
                "C005,10/15/2026,management,married,231,eli@example.com,-1586734256,New\n"
            ),
            golden_csv=(
                "Client_ID,Contact_Date,Job,Marital,Balance,Email,Phone,Lead_Status\n"
                "C001,2026-10-13,management,married,2143,alice@example.com,-1651623197,New\n"
                "C002,2026-10-13,management,single,29,bob@example.com,-1898471390,Qualified\n"
                "C003,2026-10-14,technician,married,2,not-an-email,-5596363211,Contacted\n"
                "C004,2026-10-14,management,divorced,1506,dana@example.com,-3476490784,Qualified\n"
                "C005,2026-10-15,management,married,231,eli@example.com,-1586734256,New\n"
            ),
            expected_review_issues={
                "Email": ("email_format_anomaly",),
                "Phone": ("phone_format_anomaly",),
            },
            forbidden_action_columns=("Client_ID", "Email", "Phone"),
        ),
        RealWorldCorpusCase(
            id="service_requests",
            domain="public service/free text",
            description=(
                "311-style service requests with timestamps, numeric-looking ZIP codes "
                "including leading zeros, status whitespace, and narrative resolution text."
            ),
            source_inspiration="NYC 311 Service Requests",
            source_url="https://data.cityofnewyork.us/resource/erm2-nwe9",
            dirty_csv=(
                "Unique_Key,Created_Date,Closed_Date,Agency,Complaint_Type,Descriptor,Incident_Zip,Status,Resolution_Description\n"
                "100001,10/13/2026 08:00,10/13/2026 12:30,DOT,Street Condition,Pothole,01234, Closed ,Crew inspected location\n"
                "100002,2026-10-13 08:15:00,2026-10-13 13:00:00,DEP,Water System,Hydrant Running,02108,Closed,Issue was resolved\n"
                "100003,Oct 13 2026 08:20,Oct 13 2026 14:05,NYPD,Noise - Residential,Loud Music/Party,10001,Open,Agency reviewing complaint\n"
                "100004,10/14/2026 08:35,10/14/2026 15:10,DOT,Street Condition,Pothole,30303,Closed,Crew inspected location\n"
                "100005,2026-10-14 08:50:00,2026-10-14 16:00:00,DSNY,Dirty Conditions,Trash,94105,Closed,Issue was resolved\n"
                "100006,10/15/2026 09:00,10/15/2026 17:15,DEP,Water System,Leak,BADZIP,Open,Agency reviewing complaint\n"
            ),
            golden_csv=(
                "Unique_Key,Created_Date,Closed_Date,Agency,Complaint_Type,Descriptor,Incident_Zip,Status,Resolution_Description\n"
                "100001,2026-10-13 08:00:00,2026-10-13 12:30:00,DOT,Street Condition,Pothole,01234,Closed,Crew inspected location\n"
                "100002,2026-10-13 08:15:00,2026-10-13 13:00:00,DEP,Water System,Hydrant Running,02108,Closed,Issue was resolved\n"
                "100003,2026-10-13 08:20:00,2026-10-13 14:05:00,NYPD,Noise - Residential,Loud Music/Party,10001,Open,Agency reviewing complaint\n"
                "100004,2026-10-14 08:35:00,2026-10-14 15:10:00,DOT,Street Condition,Pothole,30303,Closed,Crew inspected location\n"
                "100005,2026-10-14 08:50:00,2026-10-14 16:00:00,DSNY,Dirty Conditions,Trash,94105,Closed,Issue was resolved\n"
                "100006,2026-10-15 09:00:00,2026-10-15 17:15:00,DEP,Water System,Leak,BADZIP,Open,Agency reviewing complaint\n"
            ),
            forbidden_action_columns=(
                "Unique_Key",
                "Incident_Zip",
                "Resolution_Description",
            ),
        ),
    ]
