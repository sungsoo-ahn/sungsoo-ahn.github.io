---
layout: page
permalink: /teaching/
title: teaching
description: Lecture notes and course materials.
nav: false
nav_order: 2
---

## Lecture-note reading paths

- [Machine Learning for Molecules]({{ '/teaching/machine-learning-for-molecules/' | relative_url }})
- [Geometric Deep Learning]({{ '/teaching/geometric-deep-learning/' | relative_url }})

These topic-based chapters are reusable across semesters. Individual course pages retain schedules and semester-specific material.

## Courses

{% assign visible_courses = site.data.courses | where_exp: "course", "course.hidden != true" %}
{% for course in visible_courses %}
{% if course.external_url %}### [{{ course.title }}]({{ course.external_url }}){% else %}### [{{ course.title }}]({{ course.permalink | relative_url }}){% endif %}

**{{ course.semester }}** · {{ course.institution }}{% if course.co_instructors %} · Co-taught with {{ course.co_instructors | join: " and " }}{% endif %}

{{ course.description }}

{% unless forloop.last %}---{% endunless %}
{% endfor %}
