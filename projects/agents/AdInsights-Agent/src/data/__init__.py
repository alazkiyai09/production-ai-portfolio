"""
Ad Platform Data Module for AdInsights-Agent

Provides real API client connections to major advertising platforms.
"""

from .ad_platform_client import (
    # Enums
    PlatformType,
    # Data Classes
    PlatformCredentials,
    CampaignMetrics,
    # Base Classes
    BaseAdPlatformClient,
    # Platform Clients
    GoogleAdsClient,
    MetaAdsClient,
    TikTokAdsClient,
    LinkedInAdsClient,
    # Factory
    AdPlatformClientFactory,
    # Utility Functions
    fetch_all_platforms,
    normalize_metrics,
)

__all__ = [
    # Enums
    "PlatformType",
    # Data Classes
    "PlatformCredentials",
    "CampaignMetrics",
    # Base Classes
    "BaseAdPlatformClient",
    # Platform Clients
    "GoogleAdsClient",
    "MetaAdsClient",
    "TikTokAdsClient",
    "LinkedInAdsClient",
    # Factory
    "AdPlatformClientFactory",
    # Utility Functions
    "fetch_all_platforms",
    "normalize_metrics",
]
