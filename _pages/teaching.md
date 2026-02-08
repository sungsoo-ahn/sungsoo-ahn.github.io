---
layout: page
permalink: /teaching/
title: teaching
description: Lecture notes and course materials.
nav: true
nav_order: 2
---

{% for course in site.data.courses %}
### [{{ course.title }}]({{ course.permalink | relative_url }})

**{{ course.semester }}** · {{ course.institution }}{% if course.co_instructors %} · Co-taught with {{ course.co_instructors | join: " and " }}{% endif %}

{{ course.description }}

{% unless forloop.last %}---{% endunless %}
{% endfor %}
