"""
Memory manager for conversational AI using LangChain ChatMessageHistory.
Provides in-memory conversation storage without external databases.
"""

from typing import Dict, List, Optional
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import time
import uuid


class ConversationMemoryManager:
    """
    Manages conversation memory using LangChain's ChatMessageHistory.
    Stores conversations in memory with session-based separation.
    """

    def __init__(self, max_messages_per_session: int = 20, session_timeout_minutes: int = 60, auto_cleanup_interval: int = 100):
        """
        Initialize the memory manager.

        Args:
            max_messages_per_session: Maximum number of messages to keep per session
            session_timeout_minutes: Minutes after which inactive sessions expire
            auto_cleanup_interval: Number of operations after which to trigger automatic cleanup
        """
        self.sessions: Dict[str, Dict] = {}
        self.max_messages = max_messages_per_session
        self.session_timeout = session_timeout_minutes * 60  # Convert to seconds
        self.auto_cleanup_interval = auto_cleanup_interval
        self.operation_count = 0
        self.cleanup_stats = {'total_cleanups': 0, 'total_expired_removed': 0}

    def get_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Get existing session or create a new one.

        Args:
            session_id: Optional existing session ID

        Returns:
            Session ID (existing or newly created)
        """
        # Trigger automatic cleanup periodically
        self._maybe_auto_cleanup()

        if session_id and session_id in self.sessions:
            # Update last accessed time
            self.sessions[session_id]['last_accessed'] = time.time()
            return session_id

        # Create new session
        new_session_id = session_id or self.get_session_id()
        self.sessions[new_session_id] = {
            'chat_history': InMemoryChatMessageHistory(),
            'created_at': time.time(),
            'last_accessed': time.time()
        }
        return new_session_id

    def add_user_message(self, session_id: str, message: str) -> None:
        """
        Add a user message to the conversation history.

        Args:
            session_id: Session identifier
            message: User message content
        """
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

    def add_ai_message(self, session_id: str, message: str) -> None:
        """
        Add an AI message to the conversation history.

        Args:
            session_id: Session identifier
            message: AI message content
        """
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

    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages in conversation order
        """
        if session_id not in self.sessions:
            return []

        # Update last accessed time
        self.sessions[session_id]['last_accessed'] = time.time()

        # Trigger periodic cleanup
        self._maybe_auto_cleanup()

        return self.sessions[session_id]['chat_history'].messages

    def get_conversation_context(self, session_id: str, include_last_n: Optional[int] = None) -> str:
        """
        Get formatted conversation context for RAG prompts.

        Args:
            session_id: Session identifier
            include_last_n: Number of recent message pairs to include

        Returns:
            Formatted conversation context string
        """
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

    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was cleared, False if session didn't exist
        """
        if session_id in self.sessions:
            self.sessions[session_id]['chat_history'] = InMemoryChatMessageHistory()
            self.sessions[session_id]['last_accessed'] = time.time()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session entirely.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if session didn't exist
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Session info dict or None if session doesn't exist
        """
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

    def list_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs (not expired).

        Returns:
            List of active session IDs
        """
        current_time = time.time()
        active_sessions = []

        for session_id, session_data in self.sessions.items():
            if (current_time - session_data['last_accessed']) < self.session_timeout:
                active_sessions.append(session_id)

        return active_sessions

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions from memory.

        Returns:
            Number of sessions cleaned up
        """
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

        return len(expired_sessions)

    def get_memory_stats(self) -> Dict:
        """
        Get statistics about memory usage.

        Returns:
            Dictionary with memory statistics
        """
        # Clean up expired sessions before calculating stats
        self._maybe_auto_cleanup(force=True)

        total_sessions = len(self.sessions)
        active_sessions = len(self.list_active_sessions())
        total_messages = sum(
            len(session_data['chat_history'].messages)
            for session_data in self.sessions.values()
        )

        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'expired_sessions': total_sessions - active_sessions,
            'total_messages': total_messages,
            'max_messages_per_session': self.max_messages,
            'session_timeout_minutes': self.session_timeout / 60,
            'auto_cleanup_interval': self.auto_cleanup_interval,
            'cleanup_stats': self.cleanup_stats.copy()
        }

    def _trim_session_messages(self, session_id: str) -> None:
        """
        Trim messages in a session to stay within the maximum limit.
        Removes oldest messages first while trying to maintain conversation pairs.

        Args:
            session_id: Session identifier
        """
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

    def _maybe_auto_cleanup(self, force: bool = False) -> int:
        """
        Perform automatic cleanup if conditions are met.

        Args:
            force: Force cleanup regardless of operation count

        Returns:
            Number of sessions cleaned up
        """
        self.operation_count += 1

        if force or self.operation_count >= self.auto_cleanup_interval:
            cleaned_up = self.cleanup_expired_sessions()
            self.operation_count = 0  # Reset counter
            return cleaned_up

        return 0

    def get_cleanup_stats(self) -> Dict:
        """
        Get cleanup operation statistics.

        Returns:
            Dictionary with cleanup statistics
        """
        return {
            'total_cleanup_operations': self.cleanup_stats['total_cleanups'],
            'total_expired_sessions_removed': self.cleanup_stats['total_expired_removed'],
            'auto_cleanup_interval': self.auto_cleanup_interval,
            'current_operation_count': self.operation_count,
            'next_cleanup_in': max(0, self.auto_cleanup_interval - self.operation_count)
        }

    def force_cleanup(self) -> int:
        """
        Force an immediate cleanup of expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        return self._maybe_auto_cleanup(force=True)


# Global memory manager instance
_global_memory_manager = None


def get_memory_manager(**kwargs) -> ConversationMemoryManager:
    """
    Get the global memory manager instance (singleton pattern).

    Args:
        **kwargs: Optional parameters to pass to ConversationMemoryManager constructor
                 (only used if creating a new instance)

    Returns:
        ConversationMemoryManager instance
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = ConversationMemoryManager(**kwargs)
    return _global_memory_manager


def reset_memory_manager() -> None:
    """Reset the global memory manager (useful for testing)."""
    global _global_memory_manager
    _global_memory_manager = None


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