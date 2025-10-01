"""
Memory manager for conversational AI using LangChain ChatMessageHistory.
Provides in-memory conversation storage without external databases.

Production-ready features:
- Thread-safe operations with reentrant locks
- Comprehensive error handling and validation
- Structured logging for all operations
- Input sanitization and type validation
- Session and message size limits
- Automatic session cleanup
- Configurable via config dictionary
"""

from typing import Dict, List, Optional, Union
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import time
import uuid
import logging
import threading
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)


class MemoryManagerError(Exception):
    """Base exception for memory manager errors."""
    pass


class InvalidSessionError(MemoryManagerError):
    """Raised when session ID is invalid."""
    pass


class SessionLimitExceededError(MemoryManagerError):
    """Raised when maximum session limit is exceeded."""
    pass


class MessageSizeLimitExceededError(MemoryManagerError):
    """Raised when message size exceeds limit."""
    pass


def thread_safe(func):
    """Decorator to make methods thread-safe."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)
    return wrapper


class ConversationMemoryManager:
    """
    Manages conversation memory using LangChain's ChatMessageHistory.
    Stores conversations in memory with session-based separation.
    """

    def __init__(
        self,
        max_messages_per_session: int = 20,
        session_timeout_minutes: int = 60,
        auto_cleanup_interval: int = 100,
        max_sessions: int = 1000,
        max_message_size: int = 50000
    ):
        """
        Initialize the memory manager.

        Args:
            max_messages_per_session: Maximum number of messages to keep per session
            session_timeout_minutes: Minutes after which inactive sessions expire
            auto_cleanup_interval: Number of operations after which to trigger automatic cleanup
            max_sessions: Maximum number of concurrent sessions allowed
            max_message_size: Maximum size of a single message in characters

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate parameters
        if max_messages_per_session < 1:
            raise ValueError("max_messages_per_session must be at least 1")
        if session_timeout_minutes < 1:
            raise ValueError("session_timeout_minutes must be at least 1")
        if auto_cleanup_interval < 1:
            raise ValueError("auto_cleanup_interval must be at least 1")
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if max_message_size < 1:
            raise ValueError("max_message_size must be at least 1")

        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.sessions: Dict[str, Dict] = {}
        self.max_messages = max_messages_per_session
        self.session_timeout = session_timeout_minutes * 60  # Convert to seconds
        self.auto_cleanup_interval = auto_cleanup_interval
        self.max_sessions = max_sessions
        self.max_message_size = max_message_size
        self.operation_count = 0
        self.cleanup_stats = {'total_cleanups': 0, 'total_expired_removed': 0}

        logger.info(
            f"Memory manager initialized: max_messages={max_messages_per_session}, "
            f"timeout={session_timeout_minutes}min, max_sessions={max_sessions}, "
            f"max_message_size={max_message_size}"
        )

    def get_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())

    def _validate_session_id(self, session_id: Union[str, None]) -> None:
        """
        Validate session ID format and type.

        Args:
            session_id: Session ID to validate

        Raises:
            InvalidSessionError: If session ID is invalid
            TypeError: If session ID is not a string
        """
        if session_id is None:
            return  # Allow None for auto-generation

        if not isinstance(session_id, str):
            raise TypeError(f"Session ID must be a string, got {type(session_id).__name__}")

        if not session_id:
            raise InvalidSessionError("Session ID must be a non-empty string")

        if len(session_id) > 255:
            raise InvalidSessionError("Session ID exceeds maximum length of 255 characters")

        # Check for valid characters (alphanumeric, hyphens, underscores)
        if not all(c.isalnum() or c in '-_' for c in session_id):
            raise InvalidSessionError("Session ID contains invalid characters. Only alphanumeric, hyphens, and underscores are allowed")

    def _validate_message(self, message: Union[str, None]) -> None:
        """
        Validate message content and type.

        Args:
            message: Message content to validate

        Raises:
            TypeError: If message is not a string
            ValueError: If message is invalid
            MessageSizeLimitExceededError: If message exceeds size limit
        """
        if not isinstance(message, str):
            raise TypeError(f"Message must be a string, got {type(message).__name__}")

        if not message.strip():
            raise ValueError("Message cannot be empty or whitespace only")

        if len(message) > self.max_message_size:
            raise MessageSizeLimitExceededError(
                f"Message size ({len(message)}) exceeds limit ({self.max_message_size})"
            )

    def _validate_include_last_n(self, include_last_n: Union[int, None]) -> None:
        """
        Validate include_last_n parameter.

        Args:
            include_last_n: Number to validate

        Raises:
            TypeError: If not an integer
            ValueError: If value is invalid
        """
        if include_last_n is None:
            return

        if not isinstance(include_last_n, int):
            raise TypeError(f"include_last_n must be an integer, got {type(include_last_n).__name__}")

        if include_last_n < 1:
            raise ValueError("include_last_n must be at least 1")

    @thread_safe
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Get existing session or create a new one.

        Args:
            session_id: Optional existing session ID

        Returns:
            Session ID (existing or newly created)

        Raises:
            InvalidSessionError: If session ID is invalid
            SessionLimitExceededError: If session limit is exceeded
        """
        try:
            # Trigger automatic cleanup periodically
            self._maybe_auto_cleanup()

            if session_id:
                self._validate_session_id(session_id)
                if session_id in self.sessions:
                    # Update last accessed time
                    self.sessions[session_id]['last_accessed'] = time.time()
                    logger.debug(f"Accessed existing session: {session_id[:8]}...")
                    return session_id

            # Check session limit before creating new session
            if len(self.sessions) >= self.max_sessions:
                # Try cleanup first
                cleaned = self.cleanup_expired_sessions()
                logger.info(f"Cleaned up {cleaned} expired sessions to make room")

                # Check again after cleanup
                if len(self.sessions) >= self.max_sessions:
                    raise SessionLimitExceededError(
                        f"Maximum session limit ({self.max_sessions}) reached. "
                        f"Try again later or clear old sessions."
                    )

            # Create new session
            new_session_id = session_id or self.get_session_id()
            self.sessions[new_session_id] = {
                'chat_history': InMemoryChatMessageHistory(),
                'created_at': time.time(),
                'last_accessed': time.time()
            }
            logger.info(f"Created new session: {new_session_id[:8]}... (total: {len(self.sessions)})")
            return new_session_id

        except (InvalidSessionError, SessionLimitExceededError):
            raise
        except Exception as e:
            logger.error(f"Error in get_or_create_session: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to get or create session: {str(e)}") from e

    @thread_safe
    def add_user_message(self, session_id: str, message: str) -> None:
        """
        Add a user message to the conversation history.

        Args:
            session_id: Session identifier
            message: User message content

        Raises:
            InvalidSessionError: If session ID is invalid
            ValueError: If message is invalid
            MessageSizeLimitExceededError: If message exceeds size limit
        """
        try:
            self._validate_session_id(session_id)
            self._validate_message(message)

            if session_id not in self.sessions:
                session_id = self.get_or_create_session(session_id)

            chat_history = self.sessions[session_id]['chat_history']
            chat_history.add_message(HumanMessage(content=message))

            # Update last accessed time
            self.sessions[session_id]['last_accessed'] = time.time()

            # Trim messages if exceeding limit
            self._trim_session_messages(session_id)

            # Trigger periodic cleanup
            self._maybe_auto_cleanup()

            logger.debug(f"Added user message to session {session_id[:8]}... (size: {len(message)})")

        except (InvalidSessionError, ValueError, MessageSizeLimitExceededError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Error adding user message: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to add user message: {str(e)}") from e

    @thread_safe
    def add_ai_message(self, session_id: str, message: str) -> None:
        """
        Add an AI message to the conversation history.

        Args:
            session_id: Session identifier
            message: AI message content

        Raises:
            InvalidSessionError: If session ID is invalid
            ValueError: If message is invalid
            MessageSizeLimitExceededError: If message exceeds size limit
        """
        try:
            self._validate_session_id(session_id)
            self._validate_message(message)

            if session_id not in self.sessions:
                session_id = self.get_or_create_session(session_id)

            chat_history = self.sessions[session_id]['chat_history']
            chat_history.add_message(AIMessage(content=message))

            # Update last accessed time
            self.sessions[session_id]['last_accessed'] = time.time()

            # Trim messages if exceeding limit
            self._trim_session_messages(session_id)

            # Trigger periodic cleanup
            self._maybe_auto_cleanup()

            logger.debug(f"Added AI message to session {session_id[:8]}... (size: {len(message)})")

        except (InvalidSessionError, ValueError, MessageSizeLimitExceededError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Error adding AI message: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to add AI message: {str(e)}") from e

    @thread_safe
    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages in conversation order

        Raises:
            InvalidSessionError: If session ID is invalid
        """
        try:
            self._validate_session_id(session_id)

            if session_id not in self.sessions:
                logger.debug(f"Session {session_id[:8]}... not found, returning empty history")
                return []

            # Update last accessed time
            self.sessions[session_id]['last_accessed'] = time.time()

            # Trigger periodic cleanup
            self._maybe_auto_cleanup()

            return self.sessions[session_id]['chat_history'].messages

        except InvalidSessionError:
            raise
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to get conversation history: {str(e)}") from e

    @thread_safe
    def get_conversation_context(self, session_id: str, include_last_n: Optional[int] = None) -> str:
        """
        Get formatted conversation context for RAG prompts.

        Args:
            session_id: Session identifier
            include_last_n: Number of recent message pairs to include

        Returns:
            Formatted conversation context string

        Raises:
            InvalidSessionError: If session ID is invalid
            ValueError: If include_last_n is invalid
        """
        try:
            self._validate_include_last_n(include_last_n)

            messages = self.get_conversation_history(session_id)

            if not messages:
                return ""

            # Limit to recent messages if specified
            if include_last_n:
                messages = messages[-include_last_n * 2:]  # *2 for user+ai pairs

            # Format messages for context
            context_parts = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    context_parts.append(f"User: {message.content}")
                elif isinstance(message, AIMessage):
                    context_parts.append(f"Assistant: {message.content}")

            return "\n".join(context_parts)

        except (InvalidSessionError, ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to get conversation context: {str(e)}") from e

    @thread_safe
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was cleared, False if session didn't exist

        Raises:
            InvalidSessionError: If session ID is invalid
        """
        try:
            self._validate_session_id(session_id)

            if session_id in self.sessions:
                message_count = len(self.sessions[session_id]['chat_history'].messages)
                self.sessions[session_id]['chat_history'] = InMemoryChatMessageHistory()
                self.sessions[session_id]['last_accessed'] = time.time()
                logger.info(f"Cleared session {session_id[:8]}... ({message_count} messages)")
                return True

            logger.debug(f"Session {session_id[:8]}... not found for clearing")
            return False

        except InvalidSessionError:
            raise
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to clear session: {str(e)}") from e

    @thread_safe
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session entirely.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if session didn't exist

        Raises:
            InvalidSessionError: If session ID is invalid
        """
        try:
            self._validate_session_id(session_id)

            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session {session_id[:8]}... (remaining: {len(self.sessions)})")
                return True

            logger.debug(f"Session {session_id[:8]}... not found for deletion")
            return False

        except InvalidSessionError:
            raise
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to delete session: {str(e)}") from e

    @thread_safe
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Session info dict or None if session doesn't exist

        Raises:
            InvalidSessionError: If session ID is invalid
        """
        try:
            self._validate_session_id(session_id)

            if session_id not in self.sessions:
                return None

            session_data = self.sessions[session_id]
            message_count = len(session_data['chat_history'].messages)

            return {
                'session_id': session_id,
                'message_count': message_count,
                'created_at': session_data['created_at'],
                'last_accessed': session_data['last_accessed'],
                'is_active': (time.time() - session_data['last_accessed']) < self.session_timeout
            }

        except InvalidSessionError:
            raise
        except Exception as e:
            logger.error(f"Error getting session info: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to get session info: {str(e)}") from e

    @thread_safe
    def list_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs (not expired).

        Returns:
            List of active session IDs
        """
        try:
            current_time = time.time()
            active_sessions = []

            for session_id, session_data in self.sessions.items():
                if (current_time - session_data['last_accessed']) < self.session_timeout:
                    active_sessions.append(session_id)

            logger.debug(f"Found {len(active_sessions)} active sessions out of {len(self.sessions)}")
            return active_sessions

        except Exception as e:
            logger.error(f"Error listing active sessions: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to list active sessions: {str(e)}") from e

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions from memory.
        Note: This method is called from within thread_safe methods, so it doesn't need the decorator.

        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = time.time()
            expired_sessions = []

            for session_id, session_data in self.sessions.items():
                if (current_time - session_data['last_accessed']) >= self.session_timeout:
                    expired_sessions.append(session_id)

            # Remove expired sessions
            for session_id in expired_sessions:
                del self.sessions[session_id]

            # Update cleanup statistics
            if expired_sessions:
                self.cleanup_stats['total_cleanups'] += 1
                self.cleanup_stats['total_expired_removed'] += len(expired_sessions)
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            return len(expired_sessions)

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            # Don't raise, just return 0 to prevent cleanup failures from breaking operations
            return 0

    @thread_safe
    def get_memory_stats(self) -> Dict:
        """
        Get statistics about memory usage.

        Returns:
            Dictionary with memory statistics
        """
        try:
            # Clean up expired sessions before calculating stats
            self._maybe_auto_cleanup(force=True)

            total_sessions = len(self.sessions)
            active_sessions = len(self.list_active_sessions())
            total_messages = sum(
                len(session_data['chat_history'].messages)
                for session_data in self.sessions.values()
            )

            stats = {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'expired_sessions': total_sessions - active_sessions,
                'total_messages': total_messages,
                'max_messages_per_session': self.max_messages,
                'max_sessions': self.max_sessions,
                'max_message_size': self.max_message_size,
                'session_timeout_minutes': self.session_timeout / 60,
                'auto_cleanup_interval': self.auto_cleanup_interval,
                'cleanup_stats': self.cleanup_stats.copy()
            }

            logger.debug(f"Memory stats: {total_sessions} total, {active_sessions} active, {total_messages} messages")
            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to get memory stats: {str(e)}") from e

    def _trim_session_messages(self, session_id: str) -> None:
        """
        Trim messages in a session to stay within the maximum limit.
        Removes oldest messages first while trying to maintain conversation pairs.
        Note: This method is called from within thread_safe methods.

        Args:
            session_id: Session identifier
        """
        try:
            if session_id not in self.sessions:
                return

            chat_history = self.sessions[session_id]['chat_history']
            messages = chat_history.messages

            if len(messages) <= self.max_messages:
                return

            # Calculate how many messages to remove
            messages_to_remove = len(messages) - self.max_messages

            # Try to remove in pairs (user+ai) to maintain conversation flow
            if messages_to_remove % 2 == 1:
                messages_to_remove += 1

            # Ensure we don't remove more than available
            messages_to_remove = min(messages_to_remove, len(messages) - 2)  # Keep at least last 2

            if messages_to_remove > 0:
                # Create new history with trimmed messages
                new_history = InMemoryChatMessageHistory()
                for message in messages[messages_to_remove:]:
                    new_history.add_message(message)

                self.sessions[session_id]['chat_history'] = new_history
                logger.debug(f"Trimmed {messages_to_remove} messages from session {session_id[:8]}...")

        except Exception as e:
            logger.error(f"Error trimming session messages: {str(e)}", exc_info=True)
            # Don't raise to prevent trim failures from breaking operations

    def _maybe_auto_cleanup(self, force: bool = False) -> int:
        """
        Perform automatic cleanup if conditions are met.
        Note: This method is called from within thread_safe methods.

        Args:
            force: Force cleanup regardless of operation count

        Returns:
            Number of sessions cleaned up
        """
        try:
            self.operation_count += 1

            if force or self.operation_count >= self.auto_cleanup_interval:
                cleaned_up = self.cleanup_expired_sessions()
                self.operation_count = 0  # Reset counter
                return cleaned_up

            return 0

        except Exception as e:
            logger.error(f"Error in auto cleanup: {str(e)}", exc_info=True)
            return 0

    @thread_safe
    def get_cleanup_stats(self) -> Dict:
        """
        Get cleanup operation statistics.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            return {
                'total_cleanup_operations': self.cleanup_stats['total_cleanups'],
                'total_expired_sessions_removed': self.cleanup_stats['total_expired_removed'],
                'auto_cleanup_interval': self.auto_cleanup_interval,
                'current_operation_count': self.operation_count,
                'next_cleanup_in': max(0, self.auto_cleanup_interval - self.operation_count)
            }
        except Exception as e:
            logger.error(f"Error getting cleanup stats: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to get cleanup stats: {str(e)}") from e

    @thread_safe
    def force_cleanup(self) -> int:
        """
        Force an immediate cleanup of expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        try:
            cleaned = self._maybe_auto_cleanup(force=True)
            logger.info(f"Force cleanup completed: {cleaned} sessions removed")
            return cleaned
        except Exception as e:
            logger.error(f"Error in force cleanup: {str(e)}", exc_info=True)
            raise MemoryManagerError(f"Failed to force cleanup: {str(e)}") from e


# Global memory manager instance
_global_memory_manager = None
_global_manager_lock = threading.Lock()


def get_memory_manager(**kwargs) -> ConversationMemoryManager:
    """
    Get the global memory manager instance (singleton pattern).
    Thread-safe singleton implementation.

    Args:
        **kwargs: Optional parameters to pass to ConversationMemoryManager constructor
                 (only used if creating a new instance)

    Returns:
        ConversationMemoryManager instance
    """
    global _global_memory_manager

    # Double-checked locking pattern for thread-safe singleton
    if _global_memory_manager is None:
        with _global_manager_lock:
            if _global_memory_manager is None:
                try:
                    _global_memory_manager = ConversationMemoryManager(**kwargs)
                    logger.info("Global memory manager instance created")
                except Exception as e:
                    logger.error(f"Failed to create global memory manager: {str(e)}", exc_info=True)
                    raise

    return _global_memory_manager


def reset_memory_manager() -> None:
    """
    Reset the global memory manager (useful for testing).
    Thread-safe reset implementation.
    """
    global _global_memory_manager

    with _global_manager_lock:
        if _global_memory_manager is not None:
            logger.info("Resetting global memory manager")
        _global_memory_manager = None


def get_memory_manager_from_config(config: Optional[Dict] = None) -> ConversationMemoryManager:
    """
    Get memory manager with configuration from config dict or environment.

    Args:
        config: Optional configuration dictionary with keys:
                - max_messages_per_session
                - session_timeout_minutes
                - auto_cleanup_interval
                - max_sessions
                - max_message_size

    Returns:
        ConversationMemoryManager instance
    """
    try:
        if config is None:
            config = {}

        # Extract memory-related config with defaults
        memory_config = {
            'max_messages_per_session': config.get('max_messages_per_session', 20),
            'session_timeout_minutes': config.get('session_timeout_minutes', 60),
            'auto_cleanup_interval': config.get('auto_cleanup_interval', 100),
            'max_sessions': config.get('max_sessions', 1000),
            'max_message_size': config.get('max_message_size', 50000)
        }

        logger.info(f"Creating memory manager with config: {memory_config}")
        return get_memory_manager(**memory_config)

    except Exception as e:
        logger.error(f"Error creating memory manager from config: {str(e)}", exc_info=True)
        raise MemoryManagerError(f"Failed to create memory manager from config: {str(e)}") from e


# Example usage and testing
if __name__ == "__main__":
    import time as test_time

    print("Testing Enhanced Memory Manager with Auto-Cleanup")
    print("=" * 50)

    # Test the memory manager with shorter cleanup interval for testing
    memory = ConversationMemoryManager(
        max_messages_per_session=6,
        session_timeout_minutes=0.02,  # 1.2 seconds for testing
        auto_cleanup_interval=5  # Cleanup every 5 operations
    )

    # Create multiple sessions
    session1 = memory.get_or_create_session()
    session2 = memory.get_or_create_session()
    session3 = memory.get_or_create_session()
    print(f"Created sessions: {session1[:8]}..., {session2[:8]}..., {session3[:8]}...")

    # Add conversations to trigger operations
    for i in range(3):
        memory.add_user_message(session1, f"Message {i} from user")
        memory.add_ai_message(session1, f"Response {i} from AI")

        if i == 1:  # Check cleanup stats mid-way
            cleanup_stats = memory.get_cleanup_stats()
            print(f"\nCleanup Stats (mid-test): {cleanup_stats}")

    # Wait for sessions to expire
    print("\nWaiting for sessions to expire...")
    test_time.sleep(2)

    # Create a new session to trigger cleanup
    session4 = memory.get_or_create_session()
    print(f"Created new session: {session4[:8]}...")

    # Get final stats
    print("\nFinal Statistics:")
    stats = memory.get_memory_stats()
    print(f"Memory Stats: {stats}")

    cleanup_stats = memory.get_cleanup_stats()
    print(f"Cleanup Stats: {cleanup_stats}")

    # Test force cleanup
    print("\nTesting force cleanup...")
    cleaned = memory.force_cleanup()
    print(f"Force cleanup removed {cleaned} sessions")

    print("\nTesting completed successfully!")