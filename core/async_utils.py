"""
Async utilities for proper event loop and HTTP client management
"""

import asyncio
import atexit
from typing import Any, Coroutine, TypeVar
import openai

T = TypeVar('T')

class AsyncContextManager:
    """Centralized async context manager for proper cleanup"""
    
    def __init__(self):
        self._loop = None
        self._client = None
        self._cleanup_registered = False
    
    def get_or_create_loop(self):
        """Get existing event loop or create a new one"""
        try:
            # Try to get the current event loop
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # Create a new event loop if none exists or current is closed
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        # Register cleanup on exit if not already done
        if not self._cleanup_registered:
            atexit.register(self._cleanup)
            self._cleanup_registered = True
        
        return self._loop
    
    def get_openai_client(self) -> openai.OpenAI:
        """Get or create OpenAI client with proper async handling"""
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client
    
    def run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run async coroutine with proper cleanup"""
        loop = self.get_or_create_loop()
        
        if loop.is_running():
            # If loop is already running, create a task
            task = asyncio.create_task(coro)
            return loop.run_until_complete(task)
        else:
            # If loop is not running, run the coroutine
            return loop.run_until_complete(coro)
    
    def _cleanup(self):
        """Cleanup async resources properly"""
        if self._loop and not self._loop.is_closed():
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                
                # Wait for all tasks to complete cancellation
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                
                # Close the loop
                self._loop.close()
            except Exception:
                # Ignore cleanup errors to prevent exit issues
                pass

# Global instance
async_manager = AsyncContextManager()

def run_async_safe(coro: Coroutine[Any, Any, T]) -> T:
    """Safe wrapper for running async coroutines"""
    return async_manager.run_async(coro)

def get_openai_client() -> openai.OpenAI:
    """Get OpenAI client with proper async handling"""
    return async_manager.get_openai_client()
