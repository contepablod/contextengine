"""
Async utilities for parallel agent execution.
Enables executing independent agents concurrently.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List

logger = logging.getLogger(__name__)


class AsyncAgentExecutor:
    """
    Executes multiple agents in parallel when possible.
    Falls back to sequential execution if agents have dependencies.
    """

    def __init__(self, max_concurrent: int = 5):
        """
        Initialize executor.

        Args:
            max_concurrent: Maximum number of concurrent agent executions
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_concurrent(
        self, tasks: Dict[str, Coroutine[Any, Any, Any]]
    ) -> Dict[str, Any]:
        """
        Execute multiple agents concurrently.

        Args:
            tasks: Dictionary of task_name -> coroutine

        Returns:
            Dictionary of task_name -> result
        """

        async def bounded_task(name: str, coro: Coroutine) -> tuple[str, Any]:
            async with self.semaphore:
                try:
                    result = await coro
                    logger.info(f"Agent {name} completed successfully")
                    return name, result
                except Exception as e:
                    logger.error(f"Agent {name} failed: {e}", exc_info=True)
                    raise

        try:
            results = await asyncio.gather(
                *[bounded_task(name, coro) for name, coro in tasks.items()],
                return_exceptions=False,
            )
            return dict(results)
        except Exception as e:
            logger.error(f"Concurrent execution failed: {e}")
            raise

    async def execute_sequential(
        self, tasks: List[tuple[str, Coroutine]]
    ) -> Dict[str, Any]:
        """
        Execute agents sequentially (for dependent operations).

        Args:
            tasks: List of (name, coroutine) tuples

        Returns:
            Dictionary of name -> result
        """
        results = {}

        for name, coro in tasks:
            try:
                logger.info(f"Starting agent {name}")
                result = await coro
                results[name] = result
                logger.info(f"Agent {name} completed")
            except Exception as e:
                logger.error(f"Agent {name} failed: {e}", exc_info=True)
                raise

        return results


class AsyncBatchProcessor:
    """
    Process items in batches asynchronously.
    Useful for bulk embedding or retrieval operations.
    """

    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize processor.

        Args:
            batch_size: Items per batch
            max_workers: Max concurrent batch processors
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process_batch(
        self,
        items: List[Any],
        processor: Callable[[List[Any]], Coroutine[Any, Any, List[Any]]],
    ) -> List[Any]:
        """
        Process items in batches.

        Args:
            items: Items to process
            processor: Async function that takes batch and returns results

        Returns:
            Flattened list of results
        """

        async def process_single_batch(batch: List[Any]) -> List[Any]:
            async with self.semaphore:
                return await processor(batch)

        # Split into batches
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        try:
            batch_results = await asyncio.gather(
                *[process_single_batch(batch) for batch in batches]
            )
            # Flatten results
            return [item for batch in batch_results for item in batch]
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            raise


class AsyncRetryPolicy:
    """
    Retry policy for async operations with exponential backoff.
    """

    def __init__(self, max_retries: int = 3, base_delay_s: float = 1.0):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum retry attempts
            base_delay_s: Initial delay between retries
        """
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s

    async def execute_with_retry(
        self,
        coro_func: Callable[..., Coroutine],
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute async function with retries.

        Args:
            coro_func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from successful execution
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries + 1}")
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.base_delay_s * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {e}")

        raise last_exception


# Utility functions

async def gather_with_timeout(
    *coros: Coroutine, timeout_s: float = 30.0
) -> List[Any]:
    """
    Gather multiple coroutines with a timeout.

    Args:
        *coros: Coroutines to execute
        timeout_s: Timeout in seconds

    Returns:
        List of results

    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        logger.error(f"Operations timed out after {timeout_s}s")
        raise


async def execute_async_task(
    name: str, coro: Coroutine, timeout_s: float = 30.0
) -> Any:
    """
    Execute a single async task with timeout and logging.

    Args:
        name: Task name for logging
        coro: Coroutine to execute
        timeout_s: Timeout in seconds

    Returns:
        Task result
    """
    logger.info(f"Starting async task: {name}")
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_s)
        logger.info(f"Task {name} completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Task {name} timed out after {timeout_s}s")
        raise
    except Exception as e:
        logger.error(f"Task {name} failed: {e}", exc_info=True)
        raise
