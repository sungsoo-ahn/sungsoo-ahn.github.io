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


def format_member(row):
    """Format a member entry with optional homepage link and postdoc indicator."""
    name = row["이름(영어)"]
    homepage = row.get("홈페이지")
    role = row["역할"]

    # Add postdoc indicator
    if role == "포닥":
        name = f"{name} (Postdoc)"

    if pd.notna(homepage) and homepage and homepage != "-":
        return f"[{name}]({homepage})"
    return name


def main():
    print(f"Reading members from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # Filter active members and alumni
    active = df[df["상태"] == "재직"]
    alumni = df[df["상태"] == "졸업"]

    # Collect regular members (excluding faculty and staff)
    members = []

    for _, row in active.iterrows():
        role = row["역할"]
        # Skip faculty and staff
        if role in ["교수", "행정원"]:
            continue
        entry = format_member(row)
        members.append(entry)

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
        "## Members",
        "",
    ]

    for entry in members:
        lines.append(f"- {entry}")

    if alumni_list:
        lines.extend(["", "---", "", "## Alumni", ""])
        for name in alumni_list:
            lines.append(f"- {name}")

    lines.append("")

    # Write to file
    output = "\n".join(lines)
    OUTPUT_PATH.write_text(output)

    print(f"Generated people page with {len(members)} members and {len(alumni_list)} alumni")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
