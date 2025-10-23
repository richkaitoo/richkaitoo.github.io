---
layout: archive
title: "Blog"
permalink: /blog/
author_profile: true
---
{% assign written_year = '' %}
{% assign all_posts = site.posts | default: [] %}
{% for post in all_posts %}
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
  {% if year != written_year %}
    <h2 id="{{ year | slugify }}" class="archive__subtitle">{{ year }}</h2>
    {% assign written_year = year %}
  {% endif %}
  {% include archive-single.html %}
{% endfor %}
