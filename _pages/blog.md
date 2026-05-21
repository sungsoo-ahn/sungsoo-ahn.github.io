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
    <h1>Blogs</h1>

    {% assign postlist = site.posts | sort: "date" | reverse %}

    <ol class="bibliography">
    {% for post in postlist %}

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
            {{ post.date | date: '%B %d, %Y' }}; {{ read_time }} min read{% if post.external_source %}; {{ post.external_source }}{% endif %}
          </div>
        </div>
      </div>
    </li>

    {% endfor %}
    </ol>

  </div>
