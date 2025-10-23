---
layout: archive
permalink: /blog/
title: "All Posts"
author_profile: true
classes: wide
paginate: 10
paginate_path: "/blog/page:num/"
---

{% include base_path %}

{% assign all_posts = site.posts | concat: site.projects | sort: 'date' | reverse %}

{% capture written_year %}None{% endcapture %}

{% for post in all_posts %}
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}

  {% if year != written_year %}
    <h2 id="{{ year | slugify }}" class="archive__subtitle">{{ year }}</h2>
    {% capture written_year %}{{ year }}{% endcapture %}
  {% endif %}

  {% include archive-single.html %}
{% endfor %}

{% include pagination.html %}

