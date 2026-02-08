---
layout: page
permalink: /teaching/
title: teaching
description: Lecture notes and course materials.
nav: true
nav_order: 2
---

{% assign courses = site.teaching | group_by: "course" %}
{% assign sorted_courses = courses | sort: "name" | reverse %}

{% for course in sorted_courses %}
  {% assign first_item = course.items | first %}

## {{ first_item.course_title }}
{{ first_item.course_semester }}

  {% assign preliminary = course.items | where: "preliminary", true | sort: "lecture_number" %}
  {% assign lectures = course.items | where_exp: "item", "item.preliminary != true" | sort: "lecture_number" %}

  {% if preliminary.size > 0 %}
### Preliminary Notes

  {% for note in preliminary %}
{{ note.lecture_number }}. [{{ note.title }}]({{ note.url | relative_url }}) — {{ note.description }}
  {% endfor %}
  {% endif %}

  {% if lectures.size > 0 %}
### Lectures

  {% for lecture in lectures %}
{{ lecture.lecture_number }}. [{{ lecture.title }}]({{ lecture.url | relative_url }}) — {{ lecture.date | date: "%b %d" }}. {{ lecture.description }}
  {% endfor %}
  {% endif %}

---

{% endfor %}
