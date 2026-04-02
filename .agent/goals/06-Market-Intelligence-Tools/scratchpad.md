# Goal: Market Intelligence Tools for YouTube MCP

> **Status**: ⚪ Not Started
> **Priority**: P1 (High)
> **Created**: 2025-01-27
> **Updated**: 2025-01-27

## Overview

Extend the YouTube MCP server with market intelligence capabilities using an **ELT (Extract-Load-Transform) pipeline architecture**. This separates data extraction from analysis, enabling quota-efficient operations and flexible transformations on cached raw data.

The tools will enable AI agents to perform comprehensive YouTube market research for channel planning, niche analysis, competitive research, and content strategy.

## Success Criteria

- [ ] `analyze_niche` tool implemented and tested
- [ ] `get_trending_videos` tool with category/region filtering
- [ ] `analyze_channel_competition` tool for competitive analysis
- [ ] `find_content_gaps` tool for opportunity identification
- [ ] `get_niche_benchmarks` tool for engagement metrics
- [ ] All tools documented with examples in README
- [ ] Test coverage ≥73% maintained
- [ ] v0.0.4 released to PyPI

## Context & Background

### The Problem

The YouTube Data API v3 provides raw data access but lacks market intelligence capabilities:
- ❌ No niche analysis tools
- ❌ No competitive benchmarking
- ❌ No content gap identification
- ❌ No trend aggregation beyond basic trending
- ❌ No channel naming analysis

### The Opportunity

By aggregating and analyzing data from multiple API calls, we can provide:
- ✅ Comprehensive niche analysis
- ✅ Competitive landscape understanding
- ✅ Content opportunity identification
- ✅ Engagement benchmarks for realistic expectations
- ✅ Data-driven channel strategy insights

### Use Case: SovereignStack Channel Launch

This goal was inspired by actual channel research needs:
1. Understand the "sovereign AI stack" niche landscape
2. Analyze competitors like Anton Putra, Core Dumped
3. Find content gaps and underserved topics
4. Set realistic engagement expectations
5. Inform content and naming strategy

## What YouTube API Provides (Building Blocks)

| Endpoint | Data Available | Quota Cost |
|----------|----------------|------------|
| `search.list` | Videos, channels, playlists by keyword | 100 units |
| `videos.list` | Metadata, statistics, tags, categories | 1 unit |
| `channels.list` | Subscribers, video count, total views | 1 unit |
| `commentThreads.list` | Comments with engagement | 1 unit |
| `videos.list (chart=mostPopular)` | Regional trending videos | 1 unit |
| `videoCategories.list` | Category definitions by region | 1 unit |

### What's NOT Available (Limitations)

- Audience demographics
- Search volume/keyword difficulty
- Revenue/monetization data
- Watch time analytics
- Click-through rates
- Audience overlap between channels

## Proposed Tools

### Phase 1: Core Market Intelligence

#### 1. `analyze_niche`

Comprehensive analysis of a topic/niche on YouTube.

**Parameters:**
- `topic` (str, required): Topic or niche to analyze (e.g., "kubernetes tutorials")
- `max_videos` (int, optional): Videos to sample (default: 50)
- `max_channels` (int, optional): Channels to sample (default: 20)
- `region` (str, optional): Region code for localization (default: "US")

**Returns:**
```json
{
  "topic": "kubernetes tutorials",
  "video_analysis": {
    "sample_size": 50,
    "avg_views": 125000,
    "median_views": 45000,
    "avg_likes": 3500,
    "avg_comments": 120,
    "engagement_rate": 0.028,
    "common_tags": ["kubernetes", "k8s", "devops", "docker", "cloud"],
    "title_patterns": {
      "avg_length": 52,
      "uses_numbers": 0.45,
      "uses_questions": 0.22,
      "common_words": ["tutorial", "guide", "complete", "beginners"]
    },
    "content_duration": {
      "avg_minutes": 18,
      "short_form_ratio": 0.15
    }
  },
  "channel_analysis": {
    "sample_size": 20,
    "avg_subscribers": 85000,
    "median_subscribers": 32000,
    "avg_video_count": 145,
    "avg_total_views": 12000000,
    "naming_patterns": {
      "personal_brand": 0.35,
      "topic_focused": 0.45,
      "creative_name": 0.20
    }
  },
  "competition_assessment": {
    "saturation_score": 7.2,
    "barrier_to_entry": "medium",
    "top_performers": ["TechWorld with Nana", "KodeKloud", "DevOps Toolkit"],
    "opportunity_areas": ["specialized topics", "language-specific", "enterprise focus"]
  }
}
```

#### 2. `get_trending_videos`

Get trending videos with category and region filtering.

**Parameters:**
- `category` (str, optional): Category name or ID (e.g., "Science & Technology")
- `region` (str, optional): Region code (default: "US")
- `max_results` (int, optional): Maximum results (default: 25)

**Returns:**
```json
{
  "region": "US",
  "category": "Science & Technology",
  "videos": [
    {
      "video_id": "abc123",
      "title": "Trending Tech Video",
      "channel_title": "Tech Channel",
      "view_count": 1500000,
      "like_count": 75000,
      "published_at": "2025-01-26T...",
      "trending_rank": 1
    }
  ],
  "analysis": {
    "avg_views": 850000,
    "common_topics": ["AI", "gadgets", "reviews"],
    "optimal_video_length": "12-18 minutes"
  }
}
```

#### 3. `analyze_channel_competition`

Analyze competitive landscape for a topic.

**Parameters:**
- `topic` (str, required): Topic to analyze competition for
- `max_channels` (int, optional): Channels to analyze (default: 20)
- `include_video_analysis` (bool, optional): Include recent video stats (default: True)

**Returns:**
```json
{
  "topic": "NixOS tutorials",
  "total_channels_found": 15,
  "competition_level": "low",
  "channels": [
    {
      "channel_id": "UCxyz...",
      "title": "Vimjoyer",
      "subscribers": 45000,
      "video_count": 89,
      "avg_views_per_video": 25000,
      "upload_frequency": "weekly",
      "engagement_rate": 0.042,
      "strengths": ["consistent uploads", "clear explanations"],
      "recent_video_performance": {
        "last_5_avg_views": 32000,
        "trend": "growing"
      }
    }
  ],
  "market_gaps": [
    "Enterprise NixOS deployment",
    "NixOS + Kubernetes integration",
    "Flakes deep dives"
  ],
  "entry_strategy": {
    "difficulty": "medium",
    "recommended_focus": "underserved topics with technical depth",
    "differentiation_opportunities": ["benchmark content", "production use cases"]
  }
}
```

### Phase 2: Content Strategy Tools

#### 4. `find_content_gaps`

Identify underserved topics and content opportunities.

**Parameters:**
- `topic` (str, required): Main topic area
- `depth` (str, optional): Analysis depth ("quick", "standard", "deep")
- `include_comments` (bool, optional): Analyze comments for questions (default: True)

**Returns:**
```json
{
  "topic": "AI deployment",
  "gaps_identified": [
    {
      "subtopic": "vLLM vs TGI benchmarks",
      "existing_content_count": 3,
      "estimated_demand": "high",
      "competition": "low",
      "opportunity_score": 8.5,
      "evidence": ["frequently asked in comments", "few comprehensive videos"]
    },
    {
      "subtopic": "Kubernetes GPU scheduling",
      "existing_content_count": 7,
      "estimated_demand": "medium",
      "competition": "medium",
      "opportunity_score": 6.2,
      "evidence": ["outdated existing content", "complex topic poorly explained"]
    }
  ],
  "questions_from_audience": [
    "How do I deploy LLMs on bare metal?",
    "What's the cheapest way to run inference at scale?",
    "Ollama vs vLLM for production?"
  ],
  "recommended_content": [
    {
      "title_suggestion": "vLLM vs TGI: Complete Performance Benchmark (2025)",
      "format": "long-form tutorial",
      "estimated_effort": "high",
      "potential_impact": "high"
    }
  ]
}
```

#### 5. `get_niche_benchmarks`

Get engagement benchmarks for realistic goal setting.

**Parameters:**
- `topic` (str, required): Niche/topic to benchmark
- `channel_size` (str, optional): Target size ("small", "medium", "large")

**Returns:**
```json
{
  "topic": "DevOps tutorials",
  "channel_size_segment": "small",
  "benchmarks": {
    "subscriber_range": "1,000 - 10,000",
    "views_per_video": {
      "p25": 500,
      "p50": 2000,
      "p75": 8000,
      "p90": 25000
    },
    "likes_per_video": {
      "p25": 25,
      "p50": 100,
      "p75": 400
    },
    "comments_per_video": {
      "p25": 5,
      "p50": 20,
      "p75": 80
    },
    "engagement_rate": {
      "low": 0.02,
      "average": 0.035,
      "high": 0.06
    }
  },
  "growth_expectations": {
    "first_month_subscribers": "50-200",
    "first_year_subscribers": "1,000-5,000",
    "factors": ["upload consistency", "SEO optimization", "topic selection"]
  }
}
```

#### 6. `analyze_successful_titles`

Analyze title patterns of successful videos.

**Parameters:**
- `topic` (str, required): Topic to analyze
- `success_metric` (str, optional): "views", "engagement", "recent" (default: "views")
- `max_videos` (int, optional): Videos to analyze (default: 50)

**Returns:**
```json
{
  "topic": "Python tutorials",
  "sample_size": 50,
  "title_patterns": {
    "avg_length": 48,
    "optimal_range": "40-60 characters",
    "structure_patterns": {
      "how_to": 0.32,
      "listicle": 0.25,
      "comparison": 0.18,
      "tutorial": 0.15,
      "question": 0.10
    },
    "effective_words": [
      {"word": "complete", "frequency": 0.28, "avg_views": 125000},
      {"word": "beginners", "frequency": 0.22, "avg_views": 98000},
      {"word": "2025", "frequency": 0.18, "avg_views": 145000}
    ],
    "number_usage": {
      "uses_numbers": 0.42,
      "avg_views_with_numbers": 115000,
      "avg_views_without": 85000
    },
    "emoji_usage": {
      "uses_emoji": 0.15,
      "correlation_with_performance": "neutral"
    }
  },
  "top_performing_titles": [
    "Python Tutorial for Beginners - Full Course in 12 Hours",
    "10 Python Tips and Tricks You Need to Know",
    "Python vs JavaScript: Which Should You Learn First?"
  ],
  "recommendations": [
    "Include year for evergreen content",
    "Use numbers for listicle content",
    "Keep titles under 60 characters for mobile display"
  ]
}
```

### Phase 3: Channel Launch Tools

#### 7. `analyze_channel_names`

Analyze naming patterns in a niche.

**Parameters:**
- `topic` (str, required): Niche to analyze
- `max_channels` (int, optional): Channels to sample (default: 50)

**Returns:**
```json
{
  "topic": "DevOps",
  "channels_analyzed": 50,
  "naming_patterns": {
    "categories": {
      "personal_brand": {
        "percentage": 0.35,
        "examples": ["Anton Putra", "Nana Janashia"],
        "avg_subscribers": 125000
      },
      "topic_keyword": {
        "percentage": 0.30,
        "examples": ["DevOps Toolkit", "KodeKloud"],
        "avg_subscribers": 95000
      },
      "creative_abstract": {
        "percentage": 0.20,
        "examples": ["Fireship", "Traversy Media"],
        "avg_subscribers": 450000
      },
      "company_brand": {
        "percentage": 0.15,
        "examples": ["HashiCorp", "Red Hat"],
        "avg_subscribers": 75000
      }
    },
    "word_frequency": ["dev", "code", "tech", "cloud", "ops"],
    "avg_name_length": 12,
    "uses_numbers": 0.05
  },
  "recommendations": [
    "Personal brand works well for authority building",
    "Topic keywords help with discoverability",
    "Creative names require more marketing effort but more memorable"
  ]
}
```

#### 8. `check_channel_name_availability`

Check if a channel name is already taken.

**Parameters:**
- `name` (str, required): Proposed channel name
- `check_variations` (bool, optional): Check similar variations (default: True)

**Returns:**
```json
{
  "name": "SovereignStack",
  "exact_match_exists": false,
  "similar_channels": [
    {
      "name": "Sovereign Tech",
      "subscribers": 1200,
      "similarity_score": 0.65
    }
  ],
  "name_quality_assessment": {
    "memorability": "high",
    "searchability": "high",
    "brandability": "high",
    "pronunciation": "clear"
  },
  "platform_availability": {
    "note": "Check these manually",
    "urls_to_check": [
      "youtube.com/@SovereignStack",
      "twitter.com/SovereignStack",
      "github.com/SovereignStack"
    ]
  },
  "recommendation": "Name appears available and well-suited for the niche"
}
```

## Technical Architecture

### ELT Pipeline Design

Following data science best practices, we use an **ELT (Extract-Load-Transform)** pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                     EXTRACT LAYER                            │
│  Raw data from YouTube API (quota-consuming)                 │
│  • search results (videos, channels)                         │
│  • video metadata & statistics                               │
│  • channel info & statistics                                 │
│  • comments & engagement                                     │
│  • transcripts                                               │
│  • trending data                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      LOAD LAYER                              │
│  Store raw data with RefCache (quota-free after first call) │
│  • Namespace: youtube.raw.*                                  │
│  • TTL based on data volatility                              │
│  • Enables re-transformation without re-extraction           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   TRANSFORM LAYER                            │
│  Aggregation, analysis, insights (quota-free)                │
│  • Statistical analysis (percentiles, averages, trends)      │
│  • Pattern extraction (titles, naming, content)              │
│  • Scoring & ranking (competition, opportunity)              │
│  • Gap identification (content, market)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   INTELLIGENCE LAYER                         │
│  High-level insights & actionable recommendations            │
│  • Niche analysis reports                                    │
│  • Competition assessments                                   │
│  • Content gap reports                                       │
│  • Strategic recommendations                                 │
└─────────────────────────────────────────────────────────────┘
```

### Why ELT over ETL?

| Aspect | ETL | ELT (Chosen) |
|--------|-----|--------------|
| Transform timing | Before load | After load |
| Flexibility | Low - coupled to extraction | High - multiple transforms on same data |
| Quota efficiency | Extract per analysis | Extract once, transform many |
| Debugging | Hard - data transformed | Easy - raw data preserved |
| New analyses | Requires re-extraction | Just add new transform |

### Benefits for YouTube MCP:

1. **Quota Efficiency**: Extract once (costs quota), transform many times (free)
2. **Reusability**: Same raw video data feeds niche analysis, title patterns, benchmarks
3. **Flexibility**: Add new insights without API calls
4. **Debugging**: Raw data preserved for inspection
5. **Experimentation**: Try new transforms cheaply

### New Module Structure (ELT-Based)

```
app/tools/youtube/
├── extract/              # EXTRACT: Raw data from API
│   ├── __init__.py
│   ├── videos.py        # extract_videos_raw(query, ...) -> raw list
│   ├── channels.py      # extract_channels_raw(query, ...) -> raw list
│   ├── comments.py      # extract_comments_raw(video_id, ...) -> raw list
│   ├── trending.py      # extract_trending_raw(region, ...) -> raw list
│   └── batch.py         # batch_extract() for efficiency
│
├── load/                 # LOAD: Cache raw data with RefCache
│   ├── __init__.py
│   ├── cache.py         # load_to_cache(data, namespace, ttl)
│   ├── namespaces.py    # youtube.raw.videos, youtube.raw.channels, etc.
│   └── retrieval.py     # get_cached_raw(namespace, key)
│
├── transform/            # TRANSFORM: Analysis functions (no API calls)
│   ├── __init__.py
│   ├── statistics.py    # percentiles, averages, trends, distributions
│   ├── patterns.py      # title patterns, naming patterns, content patterns
│   ├── scoring.py       # competition score, opportunity score, engagement
│   ├── gaps.py          # content gap identification algorithms
│   ├── sentiment.py     # comment sentiment analysis
│   └── nlp.py           # text analysis utilities
│
├── intelligence/         # INTELLIGENCE: High-level orchestration
│   ├── __init__.py
│   ├── niche.py         # analyze_niche (orchestrates full E-L-T pipeline)
│   ├── competition.py   # analyze_channel_competition
│   ├── content.py       # find_content_gaps, analyze_successful_titles
│   ├── trending.py      # get_trending_videos with analysis
│   ├── naming.py        # analyze_channel_names, check_availability
│   └── benchmarks.py    # get_niche_benchmarks
│
├── search.py            # Existing (feeds into extract layer)
├── metadata.py          # Existing (feeds into extract layer)
├── comments.py          # Existing (feeds into extract layer)
├── transcripts.py       # Existing (feeds into extract layer)
└── ...
```

### Quota Management with ELT

The ELT architecture dramatically improves quota efficiency:

**Extract Layer (Costs Quota):**
| Operation | Quota Cost | Cache TTL |
|-----------|------------|-----------|
| Search videos | 100 units | 6 hours |
| Video details (batch 50) | 1 unit | 24 hours |
| Channel info (batch 20) | 1 unit | 24 hours |
| Comments (per video) | 1 unit | 5 minutes |
| Trending | 1 unit | 6 hours |

**Transform Layer (Free!):**
All transforms operate on cached raw data - zero quota cost.

**First Call vs Subsequent Calls:**
| Tool | First Call (Extract) | Subsequent (Transform only) |
|------|---------------------|----------------------------|
| `analyze_niche` | ~170 units | 0 units |
| `analyze_channel_competition` | ~220 units | 0 units |
| `find_content_gaps` | ~300 units | 0 units |
| Different transform on same data | 0 units | 0 units |

**With 10,000 daily quota:**
- First `analyze_niche` call: ~170 units
- Next 100 transforms on that data: 0 units
- Effective cost per insight: <2 units (amortized)

### Data Flow Example: `analyze_niche("kubernetes")`

```
1. EXTRACT (costs quota, cached 6-24h)
   ├── search_videos("kubernetes", 50) → raw_videos
   ├── get_video_details(video_ids) → raw_video_stats
   ├── search_channels("kubernetes", 20) → raw_channels
   └── get_channel_info(channel_ids) → raw_channel_stats

2. LOAD (store in RefCache)
   ├── youtube.raw.videos.kubernetes → raw_videos
   ├── youtube.raw.video_stats.kubernetes → raw_video_stats
   ├── youtube.raw.channels.kubernetes → raw_channels
   └── youtube.raw.channel_stats.kubernetes → raw_channel_stats

3. TRANSFORM (free, repeatable)
   ├── calculate_statistics(raw_video_stats) → view/like/comment stats
   ├── extract_patterns(raw_videos) → title patterns, tags
   ├── score_competition(raw_channels) → competition level
   └── identify_opportunities(all_data) → recommendations

4. INTELLIGENCE (assemble final report)
   └── combine all transforms → NicheAnalysisReport
```

## Constraints & Requirements

### Hard Requirements

- **Use existing YouTube tools** as building blocks (don't duplicate)
- **Maintain quota efficiency** - cache aggressively
- **Return actionable insights** - not just raw data
- **Handle API errors gracefully** - partial results are OK
- **Type annotations** for all functions
- **≥73% test coverage**

### Soft Requirements

- Results should be useful for channel planning decisions
- Scoring/ranking should be transparent (explain methodology)
- Large result sets should use RefCache pagination
- Include confidence levels where applicable

### Out of Scope

- Real-time analytics (API doesn't support)
- Revenue/monetization analysis (not available)
- Audience demographics (requires YouTube Analytics API with channel ownership)
- Automated content creation

## Tasks

| Task ID | Description | Status | Depends On |
|---------|-------------|--------|------------|
| Task-01 | Create ELT module structure (`extract/`, `load/`, `transform/`) | 🟢 Complete | - |
| Task-02 | Implement Extract layer with caching | 🟢 Complete | Task-01 |
| Task-03 | ~~Load layer RefCache integration~~ | ⚫ SKIPPED | - |
| Task-04 | Test Transform layer functions | ⚪ Not Started | Task-01 |
| Task-05 | Implement `analyze_niche` tool | ⚪ Not Started | Task-02, Task-04 |
| Task-06 | Implement `get_trending_videos` tool | ⚪ Not Started | Task-02 |
| Task-07 | Implement `analyze_channel_competition` tool | ⚪ Not Started | Task-05 |
| Task-08 | Implement `find_content_gaps` tool | ⚪ Not Started | Task-05 |
| Task-09 | Implement `get_niche_benchmarks` tool | ⚪ Not Started | Task-04 |
| Task-10 | Implement `analyze_successful_titles` tool | ⚪ Not Started | Task-04 |
| Task-11 | Implement channel naming tools | ⚪ Not Started | Task-04 |
| Task-12 | Add all tools to server.py with @cache.cached | ⚪ Not Started | Task-05-11 |
| Task-13 | Write tests for all layers | ⚪ Not Started | Task-12 |
| Task-14 | Update documentation | ⚪ Not Started | Task-13 |
| Task-15 | Release v0.0.4 | ⚪ Not Started | Task-14 |

### Architecture Insight: RefCache Decorator Pattern (2025-01-27)

**Key Discovery:** RefCache integration is trivial - just add `@cache.cached(namespace="...")` decorator!

**Pattern from existing tools:**
```python
@mcp.tool
@cache.cached(namespace="youtube.intelligence")
async def analyze_niche(topic: str) -> dict[str, Any]:
    # Call extract functions (no cache parameter needed)
    videos = await extract_videos_raw(topic, max_results=50)
    # Transform raw data
    patterns = analyze_title_patterns(videos)
    # Return result (RefCache wraps automatically)
    return {"topic": topic, "patterns": patterns}
```

**Why Task-03 is SKIPPED:**
- RefCache decorator "just works" on FastMCP tools
- No special integration needed
- Simply add decorator when implementing intelligence tools (Task-05+)
- Extract layer optional caching works but isn't required for MVP

### Task-01 Complete ✅ (2025-01-27)

**ELT Module Structure Created**

**22 files created across 4 directories:**

- `extract/`: 6 files (videos.py, channels.py, comments.py, trending.py, batch.py, __init__.py)
- `load/`: 4 files (namespaces.py, ttl.py, cache.py, __init__.py)
- `transform/`: 6 files (utils.py, statistics.py, patterns.py, scoring.py, gaps.py, __init__.py)
- `intelligence/`: 6 files (niche.py, competition.py, content.py, trending.py, naming.py, benchmarks.py, __init__.py)

**Validation Results:**
- ✅ All ruff errors fixed (16 auto-fixed, 2 manual SIM102 fixes)
- ✅ All imports verified successful
- ✅ All 406 tests pass (5.54s runtime)
- ✅ Code formatted with `ruff format`

### Task-02 Implemented 🟠 (2025-01-27)

**Extract Layer Caching Completed**

**4 files modified with caching integration:**
- ✅ `extract/videos.py` - 4 functions with caching
- ✅ `extract/channels.py` - 4 functions with caching
- ✅ `extract/comments.py` - 3 functions with caching
- ✅ `extract/trending.py` - 2 functions with caching

**Implementation Pattern:**
- All extract functions accept optional `cache: Any | None = None` parameter
- Graceful degradation when cache=None (backward compatibility)
- Use `extract_or_cache` with proper namespace and key builders
- Batch functions cache individual items for maximum reusability

**Testing Results:**
- ✅ 210 tests pass (all non-semantic tests)
- ✅ All imports successful
- ✅ No ruff errors
- ✅ MockCache test: Cache hits are instant (0.0000s), misses hit API
- ✅ TTLs correct: 6h for search, 24h for details, 30min for comments

**Ready for Task-04:** Test transform layer functions with sample data.

### Task-03 SKIPPED ⚫ (2025-01-27)

**Reason:** RefCache integration is trivial - just add `@cache.cached` decorator when implementing intelligence tools (Task-05+). No separate integration testing needed.

### Next Steps (2025-01-27)

**Immediate:**
1. **Task-04:** Test transform layer functions work on sample data
2. **Task-05:** Implement `analyze_niche` as first intelligence tool
3. **Live test** in yt-mcp-dev with zeditor

**Pattern for intelligence tools:**
```python
# In server.py
@mcp.tool
@cache.cached(namespace="youtube.intelligence.niche")
async def analyze_niche(topic: str, max_videos: int = 50) -> dict[str, Any]:
    from app.tools.youtube.intelligence import analyze_niche_impl
    return await analyze_niche_impl(topic, max_videos)
```

**Implementation approach:**
1. Implement tool logic in `app/tools/youtube/intelligence/niche.py`
2. Add thin wrapper to `server.py` with @cache.cached decorator
3. Test live in yt-mcp-dev
4. Repeat for other tools

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Quota exhaustion | High | Medium | Aggressive caching, quota tracking |
| API rate limits | Medium | Low | Batch requests, exponential backoff |
| Inaccurate insights | Medium | Medium | Confidence scores, clear methodology |
| Scope creep | Medium | High | Stick to Phase 1 for MVP |

## Dependencies

### Upstream
- Existing YouTube tools (search, metadata, comments)
- mcp-refcache for caching large results
- YouTube Data API v3 quota

### Downstream
- Goal 05: SovereignStack Channel (primary user)
- Future channel planning workflows
- Content strategy automation

## Notes & Decisions

### Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-27 | Phase 1 focuses on 3 core tools | MVP approach, validate usefulness first |
| 2025-01-27 | Use existing tools as building blocks | Code reuse, DRY principle |
| 2025-01-27 | Cache intelligence results 6-24h | Balance freshness vs. quota |
| 2025-01-27 | **Adopt ELT pipeline architecture** | Quota efficiency, flexibility, reusability |
| 2025-01-27 | Separate extract/load/transform modules | Clean separation of concerns, testability |
| 2025-01-27 | Raw data cached separately from transforms | Enable multiple analyses on same extraction |

### Open Questions

- [ ] Should we expose quota usage in responses?
- [ ] What confidence level threshold for recommendations?
- [ ] Include historical trend analysis (requires storing data over time)?
- [ ] Add export formats (CSV, JSON) for further analysis?
- [ ] Should transform functions be exposed as separate tools or internal only?
- [ ] How to handle partial extraction failures (some videos succeed, some fail)?
- [ ] Should raw data namespace be configurable per user/session?

## References

- [YouTube Data API v3 Documentation](https://developers.google.com/youtube/v3)
- [YouTube API Quota Calculator](https://developers.google.com/youtube/v3/determine_quota_cost)
- Goal 05: SovereignStack YouTube Channel (use case)
- Goal 07: Content Generation & Release Management (downstream)
- Existing YouTube tools in `app/tools/youtube/`

### ELT Pipeline References
- [ELT vs ETL: Key Differences](https://www.fivetran.com/blog/elt-vs-etl)
- [Modern Data Stack Patterns](https://www.getdbt.com/analytics-engineering/modular-data-modeling-technique/)
- [Data Pipeline Best Practices](https://dagster.io/blog/data-pipeline-best-practices)

## Implementation Notes

### Statistical Helpers Needed

```python
# utils.py functions to implement:

def calculate_percentiles(values: list[int], percentiles: list[int]) -> dict[str, int]:
    """Calculate p25, p50, p75, p90 etc."""

def calculate_engagement_rate(views: int, likes: int, comments: int) -> float:
    """Standard engagement rate formula."""

def extract_common_patterns(titles: list[str]) -> dict[str, Any]:
    """NLP-lite pattern extraction from titles."""

def score_competition(channels: list[dict], videos: list[dict]) -> float:
    """Calculate competition score 1-10."""

def identify_content_gaps(videos: list[dict], comments: list[dict]) -> list[dict]:
    """Find underserved topics from comments and sparse coverage."""
```

### Caching Strategy (ELT-Aware)

```python
# EXTRACT Layer Cache TTLs (raw data)
RAW_DATA_TTL = {
    "youtube.raw.search_videos": timedelta(hours=6),
    "youtube.raw.search_channels": timedelta(hours=6),
    "youtube.raw.video_details": timedelta(hours=24),
    "youtube.raw.channel_info": timedelta(hours=24),
    "youtube.raw.comments": timedelta(minutes=30),
    "youtube.raw.trending": timedelta(hours=6),
    "youtube.raw.transcripts": timedelta(days=7),  # Rarely changes
}

# TRANSFORM Layer Cache TTLs (derived insights)
# These are cheaper to recompute, so shorter TTL is acceptable
TRANSFORM_TTL = {
    "youtube.transform.statistics": timedelta(hours=6),
    "youtube.transform.patterns": timedelta(hours=12),
    "youtube.transform.scores": timedelta(hours=6),
}

# INTELLIGENCE Layer Cache TTLs (final reports)
INTELLIGENCE_TTL = {
    "analyze_niche": timedelta(hours=12),
    "get_trending_videos": timedelta(hours=6),
    "analyze_channel_competition": timedelta(hours=24),
    "find_content_gaps": timedelta(hours=12),
    "get_niche_benchmarks": timedelta(hours=24),
    "analyze_successful_titles": timedelta(hours=24),
    "analyze_channel_names": timedelta(hours=24),
    "check_channel_name_availability": timedelta(hours=1),  # More volatile
}
```

### Pipeline Orchestration Pattern

```python
async def analyze_niche(topic: str, region: str = "US") -> dict[str, Any]:
    """Full ELT pipeline orchestration for niche analysis."""

    # EXTRACT (or get from cache)
    raw_videos = await extract_or_cache(
        key=f"videos:{topic}",
        extractor=lambda: extract_videos_raw(topic, max_results=50),
        ttl=RAW_DATA_TTL["youtube.raw.search_videos"]
    )
    raw_channels = await extract_or_cache(
        key=f"channels:{topic}",
        extractor=lambda: extract_channels_raw(topic, max_results=20),
        ttl=RAW_DATA_TTL["youtube.raw.search_channels"]
    )

    # LOAD is implicit in extract_or_cache (RefCache handles it)

    # TRANSFORM (always runs on cached data - free)
    video_stats = transform_video_statistics(raw_videos)
    channel_stats = transform_channel_statistics(raw_channels)
    patterns = extract_title_patterns(raw_videos)
    competition = score_competition(raw_channels, raw_videos)

    # INTELLIGENCE (assemble report)
    return {
        "topic": topic,
        "video_analysis": video_stats,
        "channel_analysis": channel_stats,
        "title_patterns": patterns,
        "competition_assessment": competition,
        "recommendations": generate_recommendations(all_data),
    }
```
