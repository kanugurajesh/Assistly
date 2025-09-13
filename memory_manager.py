"""
Memory manager for conversational AI using LangChain ChatMessageHistory.
Provides in-memory conversation storage without external databases.
"""

from typing import Dict, List, Optional
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import time
import uuid


class ConversationMemoryManager:
    """
    Manages conversation memory using LangChain's ChatMessageHistory.
    Stores conversations in memory with session-based separation.
    """

    def __init__(self, max_messages_per_session: int = 20, session_timeout_minutes: int = 60):
        """
        Initialize the memory manager.

        Args:
            max_messages_per_session: Maximum number of messages to keep per session
            session_timeout_minutes: Minutes after which inactive sessions expire
        """
        self.sessions: Dict[str, Dict] = {}
        self.max_messages = max_messages_per_session
        self.session_timeout = session_timeout_minutes * 60  # Convert to seconds

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

        return len(expired_sessions)

    def get_memory_stats(self) -> Dict:
        """
        Get statistics about memory usage.

        Returns:
            Dictionary with memory statistics
        """
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
            'session_timeout_minutes': self.session_timeout / 60
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


# Global memory manager instance
_global_memory_manager = None


def get_memory_manager() -> ConversationMemoryManager:
    """
    Get the global memory manager instance (singleton pattern).

    Returns:
        ConversationMemoryManager instance
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = ConversationMemoryManager()
    return _global_memory_manager


def reset_memory_manager() -> None:
    """Reset the global memory manager (useful for testing)."""
    global _global_memory_manager
    _global_memory_manager = None


# Example usage and testing
if __name__ == "__main__":
    # Test the memory manager
    memory = ConversationMemoryManager(max_messages_per_session=6)

    # Create a session
    session_id = memory.get_or_create_session()
    print(f"Created session: {session_id}")

    # Add some conversation
    memory.add_user_message(session_id, "Hello, I'm Alice")
    memory.add_ai_message(session_id, "Hello Alice! Nice to meet you. How can I help you today?")
    memory.add_user_message(session_id, "I need help with connecting Snowflake to Atlan")
    memory.add_ai_message(session_id, "I can help you with that! You'll need specific permissions and connection details...")

    # Get conversation context
    context = memory.get_conversation_context(session_id)
    print("\nConversation Context:")
    print(context)

    # Get session info
    info = memory.get_session_info(session_id)
    print(f"\nSession Info: {info}")

    # Test message trimming
    memory.add_user_message(session_id, "What are the exact permissions needed?")
    memory.add_ai_message(session_id, "Here are the required permissions...")
    memory.add_user_message(session_id, "Thank you!")
    memory.add_ai_message(session_id, "You're welcome! Anything else?")

    print(f"\nMessages after trimming: {len(memory.get_conversation_history(session_id))}")

    # Get memory stats
    stats = memory.get_memory_stats()
    print(f"\nMemory Stats: {stats}")