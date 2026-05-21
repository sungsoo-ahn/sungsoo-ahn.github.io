---
layout: page
permalink: /teaching/
title: teaching
description: Lecture notes and course materials.
nav: false
nav_order: 2
---

{% assign visible_courses = site.data.courses | where_exp: "course", "course.hidden != true" %}
{% for course in visible_courses %}
{% if course.external_url %}### [{{ course.title }}]({{ course.external_url }}){% else %}### [{{ course.title }}]({{ course.permalink | relative_url }}){% endif %}

**{{ course.semester }}** · {{ course.institution }}{% if course.co_instructors %} · Co-taught with {{ course.co_instructors | join: " and " }}{% endif %}

{{ course.description }}

{% unless forloop.last %}---{% endunless %}
{% endfor %}
