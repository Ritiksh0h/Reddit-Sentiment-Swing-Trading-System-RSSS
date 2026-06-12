"""
Module: data/reddit_loader.py
Purpose: PRAW-based Reddit ingestion for live/recent data.
         NOT used for Phase 0 — use HuggingFace historical datasets instead.
         Reddit API approval is pending. Do not call PRAW functions until approved.
Phase: 1 — Data Pipeline
Dependencies: praw, config/settings.py, utils/logger.py
Last modified: 2026-06-10

STATUS: STUB — Reddit API credentials pending. Phase 0 uses HuggingFace datasets.
        Implement after receiving API approval.
"""

from typing import Iterator, Optional

from utils.logger import get_logger

log = get_logger(__name__)

# SPEC_GAP: PRAW client initialization should check that credentials are set
# before constructing the Reddit instance, and raise a clear error if not.


def create_reddit_client():  # type: ignore[return]
    """
    Create and return an authenticated PRAW Reddit client.

    Raises:
        RuntimeError: If REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET are unset.

    Returns:
        praw.Reddit instance.

    TODO: Phase 1 — implement after API credentials received.
    """
    # TODO: Phase 1
    raise NotImplementedError(
        "Reddit API credentials not yet configured. "
        "Use HuggingFace historical data for Phase 0."
    )


def stream_subreddit_posts(
    subreddit_name: str,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    """
    Stream new posts from a subreddit using PRAW.

    Each yielded dict has keys:
        post_id, subreddit, title, body, upvotes, comment_count,
        author_karma, created_utc (Unix float), scraped_at (ISO UTC)

    Args:
        subreddit_name: Name without "r/" prefix, e.g. "wallstreetbets"
        limit: Max posts to return (None = stream indefinitely)

    Yields:
        Post dicts matching RedditPostSchema.

    TODO: Phase 1 — implement with exponential backoff for rate limits (§4.1).
    """
    # TODO: Phase 1
    raise NotImplementedError("PRAW streaming not implemented yet — Phase 1.")


def fetch_subreddit_posts_historical(
    subreddit_name: str,
    start_timestamp: float,
    end_timestamp: float,
) -> list[dict]:
    """
    Fetch historical posts from a subreddit within a time range via PRAW search.

    Note: PRAW has limited historical access. For data before ~30 days,
    use HuggingFace datasets or Pushshift.

    Args:
        subreddit_name: Subreddit name without "r/"
        start_timestamp: Unix timestamp (start, inclusive)
        end_timestamp: Unix timestamp (end, exclusive)

    Returns:
        List of post dicts.

    TODO: Phase 1
    """
    # TODO: Phase 1
    raise NotImplementedError("Historical PRAW fetch not implemented — Phase 1.")
