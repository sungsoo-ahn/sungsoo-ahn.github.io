---
layout: page
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: false
---

  <div class="publications blog-index">
    {% assign postlist = site.posts | sort: "date" | reverse %}
    {% assign research_count = site.posts | where: "post_type", "research" | size %}
    {% assign tutorial_count = site.posts | where: "post_type", "tutorial" | size %}
    {% assign technical_note_count = site.posts | where: "post_type", "technical-note" | size %}

    <h1>SPML Lab Blog</h1>
    <p class="blog-index-note">
      Research updates, tutorials, and technical notes from SPML Lab for ML researchers entering scientific domains.
    </p>
    <div class="blog-type-summary" aria-label="Blog post types">
      <span>Post types</span>
      <span>Research {{ research_count }}</span>
      <span>Tutorials {{ tutorial_count }}</span>
      <span>Technical notes {{ technical_note_count }}</span>
    </div>

    <ol class="bibliography">
    {% for post in postlist %}
    {% assign post_type = post.post_type | default: "tutorial" %}
    {% assign post_type_label = post_type | replace: "-", " " | capitalize %}
    {% if post_type == "technical-note" %}
      {% assign post_type_label = "Technical note" %}
    {% endif %}
    {% if post.authors %}
      {% assign post_author_text = post.authors | join: ", " %}
    {% else %}
      {% assign post_author_text = post.author %}
    {% endif %}

    {% if post.external_source == blank %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}

    <li>
      <div class="row">
        <div class="col-sm-10">
          <div class="title">
            {% if post.redirect == blank %}
              <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            {% elsif post.redirect contains '://' %}
              <a href="{{ post.redirect }}" target="_blank" rel="noopener">{{ post.title }}</a>
            {% else %}
              <a href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
            {% endif %}
          </div>
          {% if post.description %}
            <div class="blog-list-description">{{ post.description }}</div>
          {% endif %}
          <div class="author">
            <span class="blog-post-type blog-post-type-{{ post_type }}">{{ post_type_label }}</span>{% if post_author_text %}; {{ post_author_text }}{% endif %}; {{ post.date | date: '%B %d, %Y' }}; {{ read_time }} min read{% if post.external_source %}; {{ post.external_source }}{% endif %}
          </div>
        </div>
      </div>
    </li>

    {% endfor %}
    </ol>

  </div>
