# Goal: Content Generation & Release Management

> **Status**: ⚪ Not Started
> **Priority**: P2 (Medium)
> **Created**: 2025-01-27
> **Updated**: 2025-01-27

## Overview

Build a complete content lifecycle management system on top of the Market Intelligence tools (Goal 06). This goal covers the journey from insight → content generation → release → feedback → iteration, creating a closed-loop system for data-driven YouTube content creation.

## Success Criteria

- [ ] Content generation tools implemented (outlines, scripts, titles, descriptions)
- [ ] Release management tools implemented (scheduling, tracking)
- [ ] Feedback collection and analysis tools implemented
- [ ] Performance comparison against benchmarks
- [ ] Closed-loop iteration system working
- [ ] All tools documented with examples
- [ ] Test coverage ≥73% maintained
- [ ] Target release: v0.0.5

## Context & Background

### The Content Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTENT LIFECYCLE PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   DISCOVER   │───▶│   GENERATE   │───▶│   RELEASE    │───▶│   FEEDBACK   │
  │              │    │              │    │              │    │              │
  │ • Niche      │    │ • Outlines   │    │ • Schedule   │    │ • Comments   │
  │ • Gaps       │    │ • Scripts    │    │ • Publish    │    │ • Metrics    │
  │ • Trends     │    │ • Titles     │    │ • Track      │    │ • Sentiment  │
  │ • Benchmarks │    │ • Tags       │    │ • Compare    │    │ • Questions  │
  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        │                                                              │
        │                      ┌──────────────┐                        │
        └──────────────────────│   ITERATE    │◀───────────────────────┘
                               │              │
                               │ • Learn      │
                               │ • Adjust     │
                               │ • Improve    │
                               └──────────────┘
```

### Dependency on Goal 06

This goal **depends on Goal 06 (Market Intelligence Tools)** for:
- Niche analysis data
- Content gap identification
- Benchmark metrics
- Title/naming patterns
- Competition assessment

The ELT pipeline from Goal 06 provides the raw data and transforms that feed into content generation decisions.

## Proposed Tools

### Phase 1: Content Generation

#### 1. `generate_video_outline`

Generate a structured video outline based on topic and market intelligence.

**Parameters:**
- `topic` (str, required): Video topic/title idea
- `target_duration` (str, optional): "short" (5-10min), "medium" (15-25min), "long" (30+min)
- `style` (str, optional): "tutorial", "comparison", "review", "explainer"
- `use_intelligence` (bool, optional): Pull data from market intelligence (default: True)

**Returns:**
```json
{
  "topic": "FastAPI vs Robyn: OCR Performance Benchmark",
  "style": "comparison",
  "target_duration": "15-20 minutes",
  "outline": {
    "hook": {
      "duration": "30 seconds",
      "content": "Teaser of surprising results, why this matters"
    },
    "intro": {
      "duration": "2 minutes",
      "content": "What we're testing, why these frameworks, DeepSeek OCR context"
    },
    "sections": [
      {
        "title": "Test Setup & Methodology",
        "duration": "3 minutes",
        "key_points": ["Hardware specs", "Docker/K8s setup", "Metrics collected"],
        "visuals": ["Architecture diagram", "Code snippets"]
      },
      {
        "title": "FastAPI Implementation",
        "duration": "3 minutes",
        "key_points": ["Code walkthrough", "Async patterns", "Optimization tips"]
      },
      {
        "title": "Robyn Implementation",
        "duration": "3 minutes",
        "key_points": ["Code walkthrough", "Rust bindings", "Performance features"]
      },
      {
        "title": "Benchmark Results",
        "duration": "5 minutes",
        "key_points": ["Latency", "Throughput", "Resource usage", "Graphs"],
        "visuals": ["Performance charts", "Comparison tables"]
      },
      {
        "title": "Analysis & Recommendations",
        "duration": "3 minutes",
        "key_points": ["When to use each", "Trade-offs", "Production considerations"]
      }
    ],
    "conclusion": {
      "duration": "1 minute",
      "content": "Summary, call to action, next video teaser"
    }
  },
  "seo_recommendations": {
    "title_suggestions": [
      "FastAPI vs Robyn: DeepSeek OCR Performance Battle! (Shocking Results)",
      "I Benchmarked Python's Fastest Frameworks with AI OCR",
      "Robyn vs FastAPI: Which is ACTUALLY Faster? (2025 Benchmarks)"
    ],
    "tags": ["fastapi", "robyn", "python", "performance", "benchmark", "ocr", "deepseek"],
    "optimal_publish_time": "Tuesday 2-4 PM EST"
  }
}
```

#### 2. `generate_title_variations`

Generate optimized title variations based on successful patterns.

**Parameters:**
- `topic` (str, required): Core topic/concept
- `style` (str, optional): "clickbait", "informative", "question", "comparison"
- `count` (int, optional): Number of variations (default: 10)
- `analyze_patterns` (bool, optional): Use title pattern analysis (default: True)

**Returns:**
```json
{
  "topic": "Kubernetes GPU scheduling",
  "variations": [
    {
      "title": "Kubernetes GPU Scheduling: The Complete Guide (2025)",
      "style": "informative",
      "length": 52,
      "predicted_ctr": "high",
      "reasoning": "Year tag, 'complete guide' performs well in niche"
    },
    {
      "title": "Why Your K8s GPU Pods Are Wasting Money (And How to Fix It)",
      "style": "problem-solution",
      "length": 58,
      "predicted_ctr": "very high",
      "reasoning": "Pain point hook, actionable promise"
    },
    {
      "title": "I Optimized GPU Scheduling in Kubernetes - Here's What Happened",
      "style": "personal story",
      "length": 56,
      "predicted_ctr": "medium-high",
      "reasoning": "Personal angle, curiosity gap"
    }
  ],
  "pattern_insights": {
    "top_performing_patterns": ["numbers", "year", "problem-solution"],
    "optimal_length": "45-60 characters",
    "avoid": ["all caps", "excessive punctuation", "clickbait without payoff"]
  }
}
```

#### 3. `generate_description`

Generate SEO-optimized video description.

**Parameters:**
- `title` (str, required): Video title
- `outline` (dict, optional): Video outline from `generate_video_outline`
- `links` (list, optional): Links to include (GitHub, resources, etc.)
- `include_timestamps` (bool, optional): Generate timestamp sections (default: True)

**Returns:**
```json
{
  "description": "Full description text with SEO optimization...",
  "sections": {
    "hook": "First 2 lines that appear in search",
    "timestamps": "0:00 Intro\n0:30 Hook\n2:00 Setup...",
    "links": "Resources and links section",
    "tags": "Hashtags and keywords",
    "cta": "Call to action (subscribe, comment, etc.)"
  },
  "seo_score": 8.5,
  "keyword_density": {
    "primary": ["fastapi", "robyn", "benchmark"],
    "secondary": ["python", "performance", "ocr"]
  }
}
```

#### 4. `generate_tags`

Generate optimized tag set based on niche analysis.

**Parameters:**
- `topic` (str, required): Video topic
- `max_tags` (int, optional): Maximum tags (default: 15, YouTube limit is 500 chars)
- `include_trending` (bool, optional): Include currently trending tags (default: True)

**Returns:**
```json
{
  "tags": [
    "fastapi",
    "robyn python",
    "python performance",
    "benchmark 2025",
    "deepseek ocr",
    "python frameworks",
    "api performance",
    "kubernetes deployment",
    "ai inference",
    "machine learning ops"
  ],
  "total_characters": 156,
  "tag_strategy": {
    "high_volume": ["python", "kubernetes", "ai"],
    "medium_competition": ["fastapi", "benchmark"],
    "low_competition_niche": ["robyn python", "deepseek ocr"]
  }
}
```

#### 5. `suggest_thumbnail_concepts`

Suggest thumbnail concepts based on successful patterns.

**Parameters:**
- `title` (str, required): Video title
- `style` (str, optional): "comparison", "tutorial", "reaction", "data"
- `niche` (str, optional): Niche for pattern analysis

**Returns:**
```json
{
  "concepts": [
    {
      "concept": "Split comparison",
      "description": "FastAPI logo vs Robyn logo with VS in middle, performance graph overlay",
      "elements": ["FastAPI logo (left)", "VS text (center)", "Robyn logo (right)", "Graph showing winner"],
      "color_scheme": "High contrast - blue vs orange",
      "text_overlay": "FASTER?",
      "success_pattern": "Comparison thumbnails get 15% higher CTR in tech niche"
    },
    {
      "concept": "Shocking result",
      "description": "Your face with surprised expression, benchmark numbers prominently displayed",
      "elements": ["Face with reaction", "Big numbers", "Winner indicator"],
      "color_scheme": "Yellow/red for attention",
      "text_overlay": "3X FASTER!",
      "success_pattern": "Data-driven thumbnails perform well for benchmark content"
    }
  ],
  "best_practices": [
    "Keep text to 3-4 words max",
    "Use contrasting colors",
    "Face expressions increase CTR by 20%",
    "Readable at small sizes (mobile)"
  ]
}
```

### Phase 2: Release Management

#### 6. `create_release_plan`

Create a release plan with optimal timing and checklist.

**Parameters:**
- `video_title` (str, required): Video title
- `target_date` (str, optional): Target release date (ISO format)
- `niche` (str, optional): Niche for optimal timing analysis

**Returns:**
```json
{
  "video_title": "FastAPI vs Robyn: OCR Benchmark",
  "release_plan": {
    "optimal_publish_time": "2025-02-04T14:00:00-05:00",
    "reasoning": "Tuesday 2PM EST - highest engagement for tech content",
    "alternative_times": [
      "2025-02-05T10:00:00-05:00",
      "2025-02-06T14:00:00-05:00"
    ]
  },
  "pre_release_checklist": [
    { "task": "Thumbnail finalized", "deadline": "2 days before" },
    { "task": "Description with timestamps", "deadline": "1 day before" },
    { "task": "Tags optimized", "deadline": "1 day before" },
    { "task": "End screen configured", "deadline": "1 day before" },
    { "task": "Community post teaser", "deadline": "1 day before" },
    { "task": "Social media announcements scheduled", "deadline": "day of" }
  ],
  "post_release_tasks": [
    { "task": "Pin comment with resources", "timing": "immediately" },
    { "task": "Respond to early comments", "timing": "first 2 hours" },
    { "task": "Share to relevant communities", "timing": "first 4 hours" },
    { "task": "Monitor initial performance", "timing": "first 24 hours" }
  ]
}
```

#### 7. `track_video_performance`

Track a video's performance against benchmarks.

**Parameters:**
- `video_id` (str, required): YouTube video ID
- `compare_to_benchmark` (bool, optional): Compare to niche benchmarks (default: True)
- `include_trajectory` (bool, optional): Include growth trajectory analysis (default: True)

**Returns:**
```json
{
  "video_id": "abc123",
  "title": "FastAPI vs Robyn: OCR Benchmark",
  "published_at": "2025-02-04T14:00:00Z",
  "age_hours": 72,
  "current_metrics": {
    "views": 5200,
    "likes": 312,
    "comments": 45,
    "engagement_rate": 0.068
  },
  "benchmark_comparison": {
    "niche": "DevOps tutorials",
    "channel_size": "small (1-10k subs)",
    "views_vs_benchmark": {
      "actual": 5200,
      "p50_expected": 2000,
      "p75_expected": 5000,
      "percentile": 78,
      "assessment": "above average"
    },
    "engagement_vs_benchmark": {
      "actual": 0.068,
      "average": 0.035,
      "assessment": "excellent - nearly 2x average"
    }
  },
  "trajectory": {
    "first_24h_views": 2100,
    "growth_rate": "strong",
    "projected_7_day_views": 12000,
    "viral_potential": "moderate"
  },
  "insights": [
    "Strong initial engagement suggests good topic-audience fit",
    "Comment-to-view ratio indicates high audience interest",
    "Consider follow-up content on related topics"
  ]
}
```

### Phase 3: Feedback & Iteration

#### 8. `collect_video_feedback`

Collect and analyze audience feedback from comments.

**Parameters:**
- `video_id` (str, required): YouTube video ID
- `max_comments` (int, optional): Maximum comments to analyze (default: 100)
- `include_sentiment` (bool, optional): Include sentiment analysis (default: True)

**Returns:**
```json
{
  "video_id": "abc123",
  "comments_analyzed": 45,
  "sentiment": {
    "positive": 0.72,
    "neutral": 0.20,
    "negative": 0.08,
    "overall": "very positive"
  },
  "themes": [
    {
      "theme": "Requesting more benchmarks",
      "frequency": 12,
      "examples": ["Can you test with more frameworks?", "Would love to see Starlette"],
      "action": "Consider follow-up video"
    },
    {
      "theme": "Questions about production use",
      "frequency": 8,
      "examples": ["Have you used Robyn in production?", "What about at scale?"],
      "action": "Address in future content or pinned comment"
    },
    {
      "theme": "Positive feedback on format",
      "frequency": 15,
      "examples": ["Great explanation!", "Love the benchmark methodology"],
      "action": "Continue this format"
    }
  ],
  "questions_to_address": [
    "How does memory usage compare?",
    "What about async database operations?",
    "Can you test with PostgreSQL?"
  ],
  "content_opportunities": [
    "Robyn in production: 6-month review",
    "FastAPI vs Robyn vs Starlette: Ultimate Comparison",
    "Database benchmarks: AsyncPG vs Psycopg3"
  ]
}
```

#### 9. `generate_performance_report`

Generate a comprehensive performance report for a video.

**Parameters:**
- `video_id` (str, required): YouTube video ID
- `include_recommendations` (bool, optional): Include actionable recommendations (default: True)

**Returns:**
```json
{
  "video_id": "abc123",
  "title": "FastAPI vs Robyn: OCR Benchmark",
  "report_generated": "2025-02-11T12:00:00Z",
  "performance_summary": {
    "overall_grade": "A-",
    "views": 12500,
    "engagement_rate": 0.062,
    "subscriber_conversion": 2.1,
    "watch_time_minutes": 8500,
    "average_view_duration": "8:32",
    "retention_score": "good"
  },
  "what_worked": [
    "Strong hook - 85% retention at 30 seconds",
    "Comparison format drove engagement",
    "Title performed well (4.2% CTR)",
    "Published at optimal time"
  ],
  "areas_for_improvement": [
    "Drop-off at 12 minutes - consider shorter format",
    "Thumbnail could be more eye-catching",
    "More call-to-actions throughout"
  ],
  "recommendations": {
    "next_video_topics": [
      "Follow up on most-asked questions",
      "Expand to more frameworks",
      "Deep dive on winner (Robyn)"
    ],
    "format_adjustments": [
      "Target 12-15 minutes based on retention",
      "Add chapter markers for navigation",
      "Include summary/TLDR section"
    ],
    "seo_adjustments": [
      "Add more long-tail keywords",
      "Update description with FAQ section"
    ]
  }
}
```

#### 10. `update_content_strategy`

Update content strategy based on accumulated feedback and performance data.

**Parameters:**
- `channel_id` (str, optional): Channel to analyze (default: user's channel)
- `video_ids` (list, optional): Specific videos to analyze
- `time_period` (str, optional): "week", "month", "quarter" (default: "month")

**Returns:**
```json
{
  "analysis_period": "2025-01-01 to 2025-01-31",
  "videos_analyzed": 4,
  "strategy_insights": {
    "top_performing_content": {
      "type": "benchmark/comparison",
      "avg_views": 8500,
      "avg_engagement": 0.058
    },
    "underperforming_content": {
      "type": "tutorial",
      "avg_views": 2100,
      "avg_engagement": 0.032,
      "hypothesis": "Too long, not enough differentiation"
    }
  },
  "audience_preferences": {
    "preferred_length": "10-15 minutes",
    "preferred_style": "data-driven comparison",
    "topics_requested": ["more AI tools", "database comparisons", "Rust alternatives"],
    "peak_engagement_days": ["Tuesday", "Thursday"]
  },
  "strategic_recommendations": {
    "content_mix": {
      "benchmarks": "50%",
      "tutorials": "30%",
      "news/updates": "20%"
    },
    "upload_frequency": "1-2 per week recommended based on audience retention",
    "focus_areas": [
      "Double down on benchmark content",
      "Shorter tutorials with clear outcomes",
      "Build series around successful topics"
    ]
  },
  "next_month_plan": [
    { "week": 1, "topic": "vLLM vs TGI Benchmark", "type": "benchmark" },
    { "week": 2, "topic": "Quick Tip: K8s GPU Setup", "type": "short" },
    { "week": 3, "topic": "Vector DB Comparison", "type": "benchmark" },
    { "week": 4, "topic": "Month in AI Open Source", "type": "news" }
  ]
}
```

## Technical Architecture

### Integration with Goal 06 (ELT Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GOAL 06: MARKET INTELLIGENCE                         │
│                            (ELT Pipeline)                                    │
│                                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────────┐   ┌──────────────────┐        │
│  │ EXTRACT │ → │  LOAD   │ → │  TRANSFORM  │ → │  INTELLIGENCE    │        │
│  └─────────┘   └─────────┘   └─────────────┘   └──────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  GOAL 07: CONTENT LIFECYCLE MANAGEMENT                       │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │  GENERATE   │ → │   RELEASE   │ → │  FEEDBACK   │ → │   ITERATE   │     │
│  │             │   │             │   │             │   │             │     │
│  │ • Outlines  │   │ • Schedule  │   │ • Comments  │   │ • Learn     │     │
│  │ • Titles    │   │ • Track     │   │ • Metrics   │   │ • Adjust    │     │
│  │ • Tags      │   │ • Compare   │   │ • Sentiment │   │ • Improve   │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### New Module Structure

```
app/tools/youtube/
├── extract/              # Goal 06: EXTRACT
├── load/                 # Goal 06: LOAD
├── transform/            # Goal 06: TRANSFORM
├── intelligence/         # Goal 06: INTELLIGENCE
│
├── content/              # Goal 07: CONTENT GENERATION
│   ├── __init__.py
│   ├── outlines.py      # generate_video_outline
│   ├── titles.py        # generate_title_variations
│   ├── descriptions.py  # generate_description
│   ├── tags.py          # generate_tags
│   ├── thumbnails.py    # suggest_thumbnail_concepts
│   └── templates.py     # Content templates and patterns
│
├── release/              # Goal 07: RELEASE MANAGEMENT
│   ├── __init__.py
│   ├── planning.py      # create_release_plan
│   ├── tracking.py      # track_video_performance
│   └── scheduling.py    # Optimal timing calculations
│
├── feedback/             # Goal 07: FEEDBACK & ITERATION
│   ├── __init__.py
│   ├── collection.py    # collect_video_feedback
│   ├── analysis.py      # Sentiment and theme analysis
│   ├── reporting.py     # generate_performance_report
│   └── strategy.py      # update_content_strategy
│
└── ... (existing modules)
```

### AI/LLM Integration

Some tools benefit from LLM assistance:

| Tool | LLM Usage | Notes |
|------|-----------|-------|
| `generate_video_outline` | Optional | Structured prompting based on patterns |
| `generate_title_variations` | Optional | Creative generation with constraints |
| `generate_description` | Optional | SEO-optimized writing |
| `collect_video_feedback` | Recommended | Theme extraction, summarization |
| `update_content_strategy` | Recommended | Synthesis of multiple data sources |

**LLM Integration Strategy:**
- Use MCP's prompt chaining for complex generation
- Provide structured context from intelligence layer
- Keep LLM calls optional (fallback to template-based)
- Cache LLM outputs with appropriate TTL

## Constraints & Requirements

### Hard Requirements

- **Depends on Goal 06** - Market Intelligence must be implemented first
- **Use intelligence data** - All generation should be informed by data
- **Actionable outputs** - Every tool should produce actionable results
- **Transparent reasoning** - Explain why recommendations are made
- **Type annotations** for all functions
- **≥73% test coverage**

### Soft Requirements

- LLM integration should be optional (graceful degradation)
- Templates should be customizable
- Support for different content styles/niches
- Export formats for integration with other tools

### Out of Scope

- Actual video production/editing
- Automated uploading to YouTube
- Thumbnail image generation (concepts only)
- Real-time analytics (API limitation)

## Tasks

| Task ID | Description | Status | Depends On |
|---------|-------------|--------|------------|
| Task-01 | Create `content/` module structure | ⚪ Not Started | Goal 06 |
| Task-02 | Implement `generate_video_outline` | ⚪ Not Started | Task-01 |
| Task-03 | Implement `generate_title_variations` | ⚪ Not Started | Task-01 |
| Task-04 | Implement `generate_description` | ⚪ Not Started | Task-01 |
| Task-05 | Implement `generate_tags` | ⚪ Not Started | Task-01 |
| Task-06 | Implement `suggest_thumbnail_concepts` | ⚪ Not Started | Task-01 |
| Task-07 | Create `release/` module structure | ⚪ Not Started | Goal 06 |
| Task-08 | Implement `create_release_plan` | ⚪ Not Started | Task-07 |
| Task-09 | Implement `track_video_performance` | ⚪ Not Started | Task-07 |
| Task-10 | Create `feedback/` module structure | ⚪ Not Started | Goal 06 |
| Task-11 | Implement `collect_video_feedback` | ⚪ Not Started | Task-10 |
| Task-12 | Implement `generate_performance_report` | ⚪ Not Started | Task-09, Task-11 |
| Task-13 | Implement `update_content_strategy` | ⚪ Not Started | Task-12 |
| Task-14 | Add all tools to server.py | ⚪ Not Started | Task-02-13 |
| Task-15 | Write tests for all modules | ⚪ Not Started | Task-14 |
| Task-16 | Update documentation | ⚪ Not Started | Task-15 |
| Task-17 | Release v0.0.5 | ⚪ Not Started | Task-16 |

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLM dependency complexity | Medium | Medium | Make LLM optional, template fallbacks |
| Scope creep into video production | High | Medium | Clear boundaries, concepts only |
| Over-reliance on patterns | Medium | Low | Include originality recommendations |
| Stale strategy recommendations | Medium | Medium | Time-bound analysis, refresh prompts |

## Dependencies

### Upstream

- **Goal 06: Market Intelligence Tools** (required)
  - Niche analysis for informed generation
  - Benchmarks for performance comparison
  - Title patterns for variation generation
  - Content gaps for topic suggestions

### Downstream

- **Goal 05: SovereignStack Channel** (primary user)
- Future automation workflows
- Third-party tool integrations

## Notes & Decisions

### Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-27 | Separate from Goal 06 | Different lifecycle phase, clear dependency |
| 2025-01-27 | LLM optional | Graceful degradation, cost control |
| 2025-01-27 | Concepts not production | Stay focused, avoid scope creep |
| 2025-01-27 | Three phases: Generate → Release → Feedback | Natural content lifecycle |

### Open Questions

- [ ] Which LLM to use for generation? (Local vs API)
- [ ] How to handle multi-language content?
- [ ] Should we integrate with YouTube Studio API for publishing?
- [ ] How to store historical performance data for trend analysis?
- [ ] Integration with social media scheduling tools?

## References

- [Goal 06: Market Intelligence Tools](../06-Market-Intelligence-Tools/scratchpad.md)
- [Goal 05: SovereignStack YouTube Channel](../05-SovereignStack-YouTube-Channel/scratchpad.md)
- [YouTube Creator Academy](https://creatoracademy.youtube.com/)
- [vidIQ Content Strategy Guide](https://vidiq.com/blog/)
- [TubeBuddy Best Practices](https://www.tubebuddy.com/blog/)

## Example Workflow

### Full Content Lifecycle Example

```python
# 1. DISCOVER (Goal 06)
niche_data = await analyze_niche("AI deployment benchmarks")
gaps = await find_content_gaps("AI deployment")
benchmarks = await get_niche_benchmarks("AI deployment")

# 2. GENERATE (Goal 07 - Phase 1)
outline = await generate_video_outline(
    topic="vLLM vs TGI: LLM Serving Benchmark",
    target_duration="medium",
    style="comparison"
)
titles = await generate_title_variations(outline["topic"])
description = await generate_description(titles[0], outline)
tags = await generate_tags(outline["topic"])
thumbnails = await suggest_thumbnail_concepts(titles[0])

# 3. RELEASE (Goal 07 - Phase 2)
release_plan = await create_release_plan(titles[0])
# ... video production happens externally ...
# After publishing:
performance = await track_video_performance("video_id_here")

# 4. FEEDBACK (Goal 07 - Phase 3)
feedback = await collect_video_feedback("video_id_here")
report = await generate_performance_report("video_id_here")

# 5. ITERATE
strategy = await update_content_strategy(time_period="month")
# Feed strategy back into DISCOVER phase for next content
```

This creates a **closed-loop, data-driven content system** that continuously improves based on real audience feedback and performance data.
