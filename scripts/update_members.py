#!/usr/bin/env python3
"""
Script to generate people.md from the SPML lab members.xlsx file.
Usage: python scripts/update_members.py
"""

import pandas as pd
from pathlib import Path

# Configuration
EXCEL_PATH = Path.home() / "Sungsahn0215 Dropbox/SPML/administration/members.xlsx"
OUTPUT_PATH = Path(__file__).parent.parent / "_pages/people.md"

# Role mappings (Korean to category)
ROLE_CATEGORIES = {
    "교수": "faculty",
    "포닥": "postdoc",
    "박사과정": "phd",
    "석박통합과정": "phd",
    "석사과정": "masters",
    "석사": "alumni",
    "행정원": "staff",
}


def format_member(row):
    """Format a member entry with optional homepage link."""
    name = row["이름(영어)"]
    homepage = row.get("홈페이지")

    if pd.notna(homepage) and homepage and homepage != "-":
        return f"[{name}]({homepage})"
    return name


def main():
    print(f"Reading members from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # Filter active members and alumni
    active = df[df["상태"] == "재직"]
    alumni = df[df["상태"] == "졸업"]

    # Categorize active members
    faculty = []
    postdocs = []
    phd_students = []
    masters_students = []
    incoming = []
    staff = []

    for _, row in active.iterrows():
        role = row["역할"]
        category = ROLE_CATEGORIES.get(role, "other")
        entry = format_member(row)

        # Check if incoming (future start date)
        start_date = row.get("입학년월")
        if pd.notna(start_date) and str(start_date) >= "2026":
            incoming.append(entry)
        elif category == "faculty":
            faculty.append((entry, row["이름(영어)"]))
        elif category == "postdoc":
            postdocs.append(entry)
        elif category == "phd":
            phd_students.append(entry)
        elif category == "masters":
            masters_students.append(entry)
        elif category == "staff":
            staff.append(entry)

    # Format alumni
    alumni_list = [row["이름(영어)"] for _, row in alumni.iterrows()]

    # Build markdown content
    lines = [
        "---",
        "layout: page",
        "permalink: /people/",
        "title: people",
        "description: Members of the SPML Lab",
        "nav: true",
        "nav_order: 3",
        "---",
        "",
        "## Faculty",
        "",
    ]

    for entry, name in faculty:
        if name == "Sungsoo Ahn":
            lines.append(f"**{name}** - Assistant Professor, KAIST Graduate School of AI")
        else:
            lines.append(f"- {entry}")

    if postdocs:
        lines.extend(["", "---", "", "## Postdoctoral Researchers", ""])
        for entry in postdocs:
            lines.append(f"- {entry}")

    if phd_students:
        lines.extend(["", "---", "", "## PhD Students", ""])
        for entry in phd_students:
            lines.append(f"- {entry}")

    if masters_students:
        lines.extend(["", "---", "", "## Master's Students", ""])
        for entry in masters_students:
            lines.append(f"- {entry}")

    if incoming:
        lines.extend(["", "---", "", "## Incoming", ""])
        for entry in incoming:
            lines.append(f"- {entry}")

    if alumni_list:
        lines.extend(["", "---", "", "## Alumni", ""])
        for name in alumni_list:
            lines.append(f"- {name}")

    lines.append("")

    # Write to file
    output = "\n".join(lines)
    OUTPUT_PATH.write_text(output)

    total = len(faculty) + len(postdocs) + len(phd_students) + len(masters_students) + len(incoming)
    print(f"Generated people page with {total} active members and {len(alumni_list)} alumni")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
