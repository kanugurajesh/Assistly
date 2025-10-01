"""
Input validation for RAG pipeline to prevent security issues and ensure data quality.
"""
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates user inputs for the RAG pipeline"""

    # Configuration constants
    MAX_QUERY_LENGTH = 2000
    MIN_QUERY_LENGTH = 3
    MAX_TICKET_SUBJECT_LENGTH = 500
    MAX_TICKET_BODY_LENGTH = 5000

    # Patterns to detect potential prompt injection
    PROMPT_INJECTION_PATTERNS = [
        r'ignore\s+(previous|all\s+)?instructions?',
        r'disregard\s+(previous|all\s+)?instructions?',
        r'system\s*:',
        r'assistant\s*:',
        r'you\s+are\s+now',
        r'act\s+as\s+if',
        r'pretend\s+(to\s+be|that)',
        r'<\s*script\s*>',  # XSS attempt
        r'eval\s*\(',  # Code injection
    ]

    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query for RAG pipeline.

        Args:
            query: User's search query or question

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is None
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"

        query_stripped = query.strip()

        # Check length limits
        if len(query_stripped) < cls.MIN_QUERY_LENGTH:
            return False, f"Query too short (minimum {cls.MIN_QUERY_LENGTH} characters)"

        if len(query_stripped) > cls.MAX_QUERY_LENGTH:
            return False, f"Query too long (maximum {cls.MAX_QUERY_LENGTH} characters)"

        # Check for potential prompt injection
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                logger.warning(
                    f"Potential prompt injection detected in query: {query_stripped[:50]}..."
                )
                # Don't block completely, but log for monitoring
                # In production, you might want to return False here

        # Check for excessive special characters (potential attack)
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in query_stripped) / len(query_stripped)
        if special_char_ratio > 0.3:
            logger.warning(f"Query contains {special_char_ratio:.1%} special characters: {query_stripped[:50]}...")

        return True, None

    @classmethod
    def validate_ticket(cls, subject: str, body: str) -> Tuple[bool, Optional[str]]:
        """
        Validate ticket subject and body.

        Args:
            subject: Ticket subject line
            body: Ticket body content

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Subject validation
        if subject and len(subject) > cls.MAX_TICKET_SUBJECT_LENGTH:
            return False, f"Subject too long (maximum {cls.MAX_TICKET_SUBJECT_LENGTH} characters)"

        # Body validation
        if not body or not body.strip():
            return False, "Ticket body cannot be empty"

        if len(body) > cls.MAX_TICKET_BODY_LENGTH:
            return False, f"Ticket body too long (maximum {cls.MAX_TICKET_BODY_LENGTH} characters)"

        # Check for prompt injection in body
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, body, re.IGNORECASE):
                logger.warning(
                    f"Potential prompt injection detected in ticket body: {body[:50]}..."
                )

        return True, None

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """
        Sanitize text by removing potentially dangerous patterns.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Remove null bytes
        sanitized = text.replace('\x00', '')

        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())

        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in sanitized if char.isprintable() or char in '\n\t')

        return sanitized

    @classmethod
    def validate_session_id(cls, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate session ID format.

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not session_id:
            return False, "Session ID cannot be empty"

        # Session IDs should be UUID format or similar
        # UUIDs are 36 characters with specific format
        if not re.match(r'^[a-f0-9\-]{8,128}$', session_id, re.IGNORECASE):
            logger.warning(f"Invalid session ID format: {session_id[:20]}...")
            return False, "Invalid session ID format"

        return True, None


# Convenience functions for easy import
def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate user query"""
    return InputValidator.validate_query(query)


def validate_ticket(subject: str, body: str) -> Tuple[bool, Optional[str]]:
    """Validate ticket content"""
    return InputValidator.validate_ticket(subject, body)


def sanitize_text(text: str) -> str:
    """Sanitize text input"""
    return InputValidator.sanitize_text(text)
