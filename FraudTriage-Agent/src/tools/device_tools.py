"""
Device and IP-related tools for gathering context.

These tools check device fingerprints, IP reputation,
and geolocation for fraud triage.
"""

import logging
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config.settings import settings

logger = logging.getLogger(__name__)


class DeviceFingerprintInput(BaseModel):
    """Input schema for device fingerprint tool."""

    device_id: str = Field(..., description="Device ID or fingerprint")


class IPReputationInput(BaseModel):
    """Input schema for IP reputation check tool."""

    ip_address: str = Field(..., description="IP address to check")


async def get_device_fingerprint(device_id: str) -> dict[str, Any]:
    """
    Get device fingerprint information.

    Args:
        device_id: Device ID or fingerprint

    Returns:
        Device fingerprint data
    """
    logger.info(f"Fetching device fingerprint for {device_id}")

    if settings.mock_external_apis:
        return _mock_device_fingerprint(device_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.device_fingerprint_url}/devices/{device_id}",
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching device fingerprint: {e}")
        return {"error": str(e)}


async def check_ip_reputation(ip_address: str) -> dict[str, Any]:
    """
    Check IP reputation and geolocation.

    Args:
        ip_address: IP address to check

    Returns:
        IP reputation data
    """
    logger.info(f"Checking IP reputation for {ip_address}")

    if settings.mock_external_apis:
        return _mock_ip_reputation(ip_address)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.device_fingerprint_url}/ip/{ip_address}",
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error checking IP reputation: {e}")
        return {"error": str(e)}


# LangChain tools
get_device_fingerprint_tool = StructuredTool.from_function(
    coroutine=get_device_fingerprint,
    name="get_device_fingerprint",
    description="Get device fingerprint information including device type, OS, browser, and previous usage. Use this to check if a device is new or known.",
    args_schema=DeviceFingerprintInput,
)

check_ip_reputation_tool = StructuredTool.from_function(
    coroutine=check_ip_reputation,
    name="check_ip_reputation",
    description="Check IP reputation, geolocation, and risk indicators. Use this to identify high-risk IP addresses (VPNs, proxies, botnets, etc.)",
    args_schema=IPReputationInput,
)


# Mock data functions
def _mock_device_fingerprint(device_id: str) -> dict[str, Any]:
    """Generate mock device fingerprint."""
    # Different device types based on device_id
    is_new = "new" in device_id.lower() or hash(device_id) % 3 == 0

    return {
        "device_id": device_id,
        "device_type": "mobile" if hash(device_id) % 2 == 0 else "desktop",
        "operating_system": "iOS 17.2" if hash(device_id) % 2 == 0 else "Windows 11",
        "browser": "Safari" if hash(device_id) % 2 == 0 else "Chrome 120",
        "first_seen": "2025-01-28" if is_new else "2023-06-15",
        "last_seen": "2025-01-30",
        "is_new_device": is_new,
        "transaction_count": 1 if is_new else 47,
        "associated_accounts": 1 if is_new else 2,
        "risk_score": 70 if is_new else 10,
        "risk_indicators": [
            "First transaction from this device",
            "Device fingerprint not recognized",
        ] if is_new else [],
    }


def _mock_ip_reputation(ip_address: str) -> dict[str, Any]:
    """Generate mock IP reputation."""
    # Some high-risk IPs based on patterns
    is_high_risk = any([
        ip_address.startswith("197."),  # Nigeria
        ip_address.startswith("41."),   # Nigeria
        ip_address.startswith("185."),  # Known proxy range
        hash(ip_address) % 4 == 0,
    ])

    country_map = {
        "197.": "Nigeria",
        "41.": "Nigeria",
        "185.": "Russia",
        "5.": "United States",
        "8.": "United States",
    }

    country = "Unknown"
    for prefix, country_name in country_map.items():
        if ip_address.startswith(prefix):
            country = country_name
            break

    return {
        "ip_address": ip_address,
        "country": country,
        "city": "Lagos" if country == "Nigeria" else ("Moscow" if country == "Russia" else "New York"),
        "is_vpn": is_high_risk and hash(ip_address) % 2 == 0,
        "is_proxy": is_high_risk,
        "is_tor_exit_node": False,
        "is_datacenter_ip": is_high_risk,
        "risk_score": 80 if is_high_risk else 15,
        "risk_indicators": [
            "High-risk country",
            "Known proxy/VPN",
        ] if is_high_risk else ["Residential IP"],
        "asn": f"AS{hash(ip_address) % 65535}",
    }
