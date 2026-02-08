---
layout: page
permalink: /teaching/2026-spring-protein-ai/
title: "Protein & Artificial Intelligence"
description: "Spring 2026 · KAIST"
---

{% assign course = site.data.courses | where: "id", "2026-spring-protein-ai" | first %}

**{{ course.semester }}** · {{ course.institution }}{% if course.co_instructors %} · Co-taught with {{ course.co_instructors | join: " and " }}{% endif %}

{{ course.description }}

**Prerequisites:** {{ course.prerequisites }}

**Assessment:** {{ course.assessment }}

{% assign notes = site.teaching | where: "course", course.id %}
{% assign preliminary = notes | where: "preliminary", true | sort: "lecture_number" %}
{% assign lectures = notes | where_exp: "item", "item.preliminary != true" | sort: "lecture_number" %}

{% if preliminary.size > 0 %}
### Preliminary Notes
<p class="post-description" style="margin-bottom: 0.5em;">{{ course.preliminary_description }}</p>

<ol start="0">
{% for note in preliminary %}
  <li value="{{ note.lecture_number }}"><a href="{{ note.url | relative_url }}">{{ note.title }}</a> — {{ note.description }}</li>
{% endfor %}
</ol>
{% endif %}

{% if lectures.size > 0 %}
### Lectures
<p class="post-description" style="margin-bottom: 0.5em;">{{ course.lectures_description }}</p>

<ol>
{% for lecture in lectures %}
  <li><a href="{{ lecture.url | relative_url }}">{{ lecture.title }}</a> — {{ lecture.date | date: "%b %d" }}. {{ lecture.description }}</li>
{% endfor %}
</ol>
{% endif %}

{% if course.homework.size > 0 %}
#### Homework Assignments

| # | Topic | Description |
|---|-------|-------------|
{% for hw in course.homework %}| {{ hw.number }} | {{ hw.topic }} | {{ hw.description }} |
{% endfor %}
{% endif %}

{% if course.references.size > 0 %}
#### Key References

{% for ref in course.references %}- {{ ref }}
{% endfor %}
{% endif %}
