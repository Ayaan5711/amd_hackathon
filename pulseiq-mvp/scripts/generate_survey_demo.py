"""Generate the seeded Survey Analytics demo dataset.

Run from the pulseiq-mvp project root:

    python scripts/generate_survey_demo.py

Writes:
    app/data/survey_demo/responses.csv

Mirrors tests/fixtures/sample_survey.csv's columns (Employee_ID, Department,
Quarter, Satisfaction_Score, NPS, Engagement_Score, Comments, Manager_Rating,
Years_At_Company) across 4 departments x 4 quarters, with:

- Sales trending steadily down quarter-over-quarter (drives trends_analysis /
  "Notable Changes").
- A couple of deliberately rock-bottom rows (drives anomalies_and_quality).
- A comment in Sales and one in HR that embed an email / phone number (drives
  the PII detection path).
- Open-text comments spanning positive/negative/neutral sentiment using the
  same lexicon `app/packs/survey/report.py`'s `_keyword_themes` looks for
  (drives themes_and_sentiment).

Five additional demographic/"response" columns (Gender, Age_Band, City,
Outlook_General, Outlook_Food_Prices) are appended for the
`app/packs/survey/categorical.py` demographic-profile / Likert-response /
cross-tab analysis:

- Gender, Age_Band, City: weighted demographic breakdowns (every bucket clears
  `MIN_SEGMENT_SIZE`).
- Outlook_General: 5-option Likert ("More than current" / "Similar to current" /
  "Less than current" / "No change" / "Decline"), weighted by Gender so Male
  respondents skew markedly more towards "More than current" than Female
  respondents (drives the "Key Findings" gender-gap sentence).
- Outlook_Food_Prices: same 5 options, weighted by City so Guwahati skews most
  towards "More than current" (drives "which city has the highest ..." chat
  questions).

Seeded (SEED) for reproducibility.
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SURVEY_DEMO_PATH  # noqa: E402

SEED = 7
EMPLOYEES_PER_CELL = 7

DEPARTMENTS = ["Sales", "Marketing", "Engineering", "HR"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

# department -> quarter -> (satisfaction, nps, engagement, manager_rating) means.
# Sales trends steadily downward; the others stay roughly flat.
DEPARTMENT_BASE: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "Sales": {
        "Q1": (4.3, 8.0, 4.0, 4.0),
        "Q2": (3.6, 6.0, 3.3, 3.6),
        "Q3": (2.9, 4.0, 2.6, 3.0),
        "Q4": (2.1, 2.0, 1.9, 2.4),
    },
    "Marketing": {
        "Q1": (4.4, 8.5, 4.3, 4.4),
        "Q2": (4.5, 8.7, 4.4, 4.5),
        "Q3": (4.3, 8.3, 4.2, 4.3),
        "Q4": (4.4, 8.6, 4.3, 4.4),
    },
    "Engineering": {
        "Q1": (3.9, 7.0, 3.8, 3.9),
        "Q2": (4.0, 7.2, 3.9, 4.0),
        "Q3": (3.8, 6.8, 3.7, 3.8),
        "Q4": (4.0, 7.3, 3.9, 4.0),
    },
    "HR": {
        "Q1": (4.1, 7.5, 4.0, 4.1),
        "Q2": (4.2, 7.7, 4.1, 4.2),
        "Q3": (4.0, 7.4, 4.0, 4.0),
        "Q4": (4.2, 7.6, 4.1, 4.2),
    },
}

# Demographic columns (Gender, Age_Band, City) and Likert-style "response"
# columns (Outlook_General, Outlook_Food_Prices) for the
# `app/packs/survey/categorical.py` demographic-profile / response-distribution
# / cross-tab analysis.
GENDERS = ["Male", "Female"]
GENDER_WEIGHTS = [0.51, 0.49]

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_BAND_WEIGHTS = [0.28, 0.27, 0.20, 0.15, 0.10]

CITIES = ["Guwahati", "Mumbai", "Delhi", "Bengaluru", "Kolkata"]
CITY_WEIGHTS = [0.24, 0.22, 0.20, 0.18, 0.16]

OUTLOOK_OPTIONS = [
    "More than current",
    "Similar to current",
    "Less than current",
    "No change",
    "Decline",
]

# Outlook_General weighted by Gender: Male respondents skew markedly more
# towards "More than current" than Female respondents.
GENDER_OUTLOOK_WEIGHTS: dict[str, list[float]] = {
    "Male": [0.60, 0.20, 0.10, 0.07, 0.03],
    "Female": [0.30, 0.35, 0.15, 0.15, 0.05],
}

# Outlook_Food_Prices weighted by City: Guwahati skews most towards
# "More than current".
CITY_FOOD_OUTLOOK_WEIGHTS: dict[str, list[float]] = {
    "Guwahati": [0.68, 0.18, 0.07, 0.05, 0.02],
    "Mumbai": [0.35, 0.30, 0.15, 0.15, 0.05],
    "Delhi": [0.32, 0.33, 0.13, 0.13, 0.09],
    "Bengaluru": [0.30, 0.35, 0.15, 0.15, 0.05],
    "Kolkata": [0.32, 0.33, 0.15, 0.15, 0.05],
}

POSITIVE_COMMENTS = [
    "I really enjoy the supportive team this quarter - the new tools have been great.",
    "Management has been very helpful and the workload feels manageable lately.",
    "I love the flexible schedule and the training program has been excellent.",
    "Communication has improved a lot and things feel a lot smoother now.",
    "Great team environment - I feel appreciated and well supported.",
    "The onboarding process was smooth and my manager has been really helpful.",
    "Happy with the new project assignments, the extra support has helped a ton.",
]

NEGATIVE_COMMENTS = [
    "The workload has been really tight and I'm feeling pretty stressed most weeks.",
    "Communication from leadership is unclear and often confusing.",
    "Our tools are outdated and it's frustrating to get basic tasks done.",
    "I have concerns about career growth - the path forward needs more clarity.",
    "The new process is confusing and the documentation is lacking.",
    "This has been one of the worst quarters - morale is low and support is poor.",
    "Feeling unhappy with the workload, it's been a really tough and stressful stretch.",
]

NEUTRAL_COMMENTS = [
    "Nothing major to report this quarter, things are about the same as before.",
    "The training sessions were okay, would be nice to have more hands-on practice.",
    "Workload has been steady, no major changes to report this quarter.",
    "Still getting used to the new process, seems fine so far.",
]

PII_EMAIL_COMMENT = (
    "Please follow up with me directly at jordan.taylor92@gmail.com about the schedule change."
)
PII_PHONE_COMMENT = "You can reach me at (415) 555-0148 if you'd like to discuss this further."


def pick_comment(satisfaction: int, rng: random.Random) -> str:
    if satisfaction >= 4:
        return rng.choice(POSITIVE_COMMENTS)
    if satisfaction <= 2:
        return rng.choice(NEGATIVE_COMMENTS)
    return rng.choice(NEUTRAL_COMMENTS)


def clamp_round(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, round(value)))


def main() -> None:
    rng = random.Random(SEED)

    rows: list[dict[str, object]] = []
    employee_num = 1
    for department in DEPARTMENTS:
        for quarter in QUARTERS:
            satisfaction_mean, nps_mean, engagement_mean, manager_mean = DEPARTMENT_BASE[department][quarter]
            for _ in range(EMPLOYEES_PER_CELL):
                satisfaction = clamp_round(rng.gauss(satisfaction_mean, 0.5), 1, 5)
                nps = clamp_round(rng.gauss(nps_mean, 1.2), 0, 10)
                engagement = clamp_round(rng.gauss(engagement_mean, 0.5), 1, 5)
                manager_rating = clamp_round(rng.gauss(manager_mean, 0.5), 1, 5)
                years_at_company = rng.randint(1, 10)

                gender = rng.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
                age_band = rng.choices(AGE_BANDS, weights=AGE_BAND_WEIGHTS, k=1)[0]
                city = rng.choices(CITIES, weights=CITY_WEIGHTS, k=1)[0]
                outlook_general = rng.choices(
                    OUTLOOK_OPTIONS, weights=GENDER_OUTLOOK_WEIGHTS[gender], k=1
                )[0]
                outlook_food_prices = rng.choices(
                    OUTLOOK_OPTIONS, weights=CITY_FOOD_OUTLOOK_WEIGHTS[city], k=1
                )[0]

                rows.append({
                    "Employee_ID": f"EMP{employee_num:04d}",
                    "Department": department,
                    "Quarter": quarter,
                    "Satisfaction_Score": satisfaction,
                    "NPS": nps,
                    "Engagement_Score": engagement,
                    "Comments": pick_comment(satisfaction, rng),
                    "Manager_Rating": manager_rating,
                    "Years_At_Company": years_at_company,
                    "Gender": gender,
                    "Age_Band": age_band,
                    "City": city,
                    "Outlook_General": outlook_general,
                    "Outlook_Food_Prices": outlook_food_prices,
                })
                employee_num += 1

    # A couple of deliberately rock-bottom rows for anomalies_and_quality, in
    # departments that otherwise have no low scores so they stand out.
    for department, quarter in [("Marketing", "Q2"), ("Engineering", "Q3")]:
        for row in rows:
            if row["Department"] == department and row["Quarter"] == quarter:
                row.update({
                    "Satisfaction_Score": 1,
                    "NPS": 0,
                    "Engagement_Score": 1,
                    "Manager_Rating": 1,
                    "Comments": rng.choice(NEGATIVE_COMMENTS),
                })
                break

    # Embed PII in a couple of comments for the PII detection path.
    for department, quarter, comment in [
        ("Sales", "Q1", PII_EMAIL_COMMENT),
        ("HR", "Q3", PII_PHONE_COMMENT),
    ]:
        for row in rows:
            if row["Department"] == department and row["Quarter"] == quarter:
                row["Comments"] = comment
                break

    SURVEY_DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SURVEY_DEMO_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Employee_ID", "Department", "Quarter", "Satisfaction_Score", "NPS",
            "Engagement_Score", "Comments", "Manager_Rating", "Years_At_Company",
            "Gender", "Age_Band", "City", "Outlook_General", "Outlook_Food_Prices",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {SURVEY_DEMO_PATH}")


if __name__ == "__main__":
    main()
