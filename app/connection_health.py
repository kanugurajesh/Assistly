"""
Connection health check utilities for monitoring database and API connections.
"""
import logging
import sys
from typing import Dict, Tuple
from datetime import datetime

# Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure stdout uses UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


def check_mongodb_health(client) -> Tuple[bool, str]:
    """
    Check MongoDB connection health.

    Args:
        client: MongoDB client instance

    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        # Ping the database with a short timeout
        client.admin.command('ping', maxTimeMS=5000)

        # Get server info
        server_info = client.server_info()
        version = server_info.get('version', 'unknown')

        logger.info(f"MongoDB health check: OK (version: {version})")
        return True, f"MongoDB healthy (v{version})"

    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return False, f"MongoDB connection error: {str(e)}"


def check_qdrant_health(client) -> Tuple[bool, str]:
    """
    Check Qdrant connection health.

    Args:
        client: Qdrant client instance

    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        # Try to get collections list as a health check
        collections = client.get_collections()
        collection_count = len(collections.collections)

        logger.info(f"Qdrant health check: OK ({collection_count} collections)")
        return True, f"Qdrant healthy ({collection_count} collections)"

    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return False, f"Qdrant connection error: {str(e)}"


def check_openai_health(client) -> Tuple[bool, str]:
    """
    Check OpenAI API connection health with a minimal request.

    Args:
        client: OpenAI client instance

    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        # Make a minimal API call to check connection
        # Using a very short completion request
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )

        if response.choices:
            logger.info("OpenAI health check: OK")
            return True, "OpenAI API healthy"
        else:
            logger.warning("OpenAI health check: No response received")
            return False, "OpenAI API returned empty response"

    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        return False, f"OpenAI API error: {str(e)}"


def check_all_connections() -> Dict[str, Dict]:
    """
    Check health of all service connections.

    Returns:
        Dictionary with health status for each service
    """
    from pathlib import Path

    # Add parent directory to path to import utils module
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from rag_pipeline import qdrant_client, openai_client
    from utils import get_mongodb_client, close_mongodb_client

    results = {
        'timestamp': datetime.now().isoformat(),
        'services': {}
    }

    # Check MongoDB
    try:
        mongo_client = get_mongodb_client()
        is_healthy, message = check_mongodb_health(mongo_client)
        results['services']['mongodb'] = {
            'healthy': is_healthy,
            'message': message
        }
        close_mongodb_client(mongo_client)
    except Exception as e:
        results['services']['mongodb'] = {
            'healthy': False,
            'message': f"Failed to create MongoDB client: {str(e)}"
        }

    # Check Qdrant
    try:
        is_healthy, message = check_qdrant_health(qdrant_client)
        results['services']['qdrant'] = {
            'healthy': is_healthy,
            'message': message
        }
    except Exception as e:
        results['services']['qdrant'] = {
            'healthy': False,
            'message': f"Failed to check Qdrant: {str(e)}"
        }

    # Check OpenAI (optional - costs tokens)
    # Commented out by default to avoid costs
    # try:
    #     is_healthy, message = check_openai_health(openai_client)
    #     results['services']['openai'] = {
    #         'healthy': is_healthy,
    #         'message': message
    #     }
    # except Exception as e:
    #     results['services']['openai'] = {
    #         'healthy': False,
    #         'message': f"Failed to check OpenAI: {str(e)}"
    #     }

    # Overall health
    all_healthy = all(
        svc.get('healthy', False)
        for svc in results['services'].values()
    )
    results['overall_healthy'] = all_healthy

    return results


def log_connection_status() -> None:
    """Log current connection status for all services."""
    logger.info("=" * 60)
    logger.info("CONNECTION HEALTH CHECK")
    logger.info("=" * 60)

    results = check_all_connections()

    for service, status in results['services'].items():
        status_emoji = "✓" if status['healthy'] else "✗"
        logger.info(f"{status_emoji} {service.upper()}: {status['message']}")

    overall_emoji = "✓" if results['overall_healthy'] else "✗"
    logger.info("=" * 60)
    logger.info(f"{overall_emoji} OVERALL STATUS: {'HEALTHY' if results['overall_healthy'] else 'UNHEALTHY'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Test connection health checks
    log_connection_status()
