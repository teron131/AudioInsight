import asyncio
import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Set

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..logging_config import get_logger
from .config import LLMConfig
from .utils import get_api_credentials

logger = get_logger(__name__)

# Shared thread pool executor for all LLM operations to avoid overhead
_shared_executor = None


def get_shared_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor for LLM operations."""
    global _shared_executor
    if _shared_executor is None:
        # Increased from 4 to 8 workers for better parallel processing
        _shared_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-executor")
    return _shared_executor


class WorkItem:
    """Wrapper for work items with deduplication support."""

    def __init__(self, data: Any, item_id: Optional[str] = None):
        self.data = data
        self.item_id = item_id or str(uuid.uuid4())
        self.created_at = time.time()

    def get_content_hash(self) -> str:
        """Get a hash of the content for deduplication."""
        if isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()
        elif isinstance(self.data, (tuple, list)) and len(self.data) > 0 and isinstance(self.data[0], str):
            # For tuples like (text, speaker_info, timestamps)
            return hashlib.md5(str(self.data[0]).encode()).hexdigest()
        else:
            return hashlib.md5(str(self.data).encode()).hexdigest()


class EventBasedProcessor(ABC):
    """
    Base class for event-based processing with queue management and worker tasks.

    Provides common functionality for:
    - Event-based triggering instead of polling
    - Queue-based processing to prevent overflow
    - Worker task management with coordination
    - Trigger condition checking
    - Concurrent processing for better performance
    - Adaptive frequency matching actual processing times
    - Work deduplication and atomic state management
    """

    def __init__(self, queue_maxsize: int = 10, cooldown_seconds: float = 2.0, max_concurrent_workers: int = 2, enable_work_coordination: bool = True):
        """Initialize the event-based processor.

        Args:
            queue_maxsize: Maximum size of the processing queue
            cooldown_seconds: Initial cooldown time (will be adapted based on actual processing)
            max_concurrent_workers: Maximum number of concurrent worker tasks
            enable_work_coordination: Enable work coordination to prevent duplicate processing
        """
        # Event-based triggering
        self.new_event = asyncio.Event()
        self.processing_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.worker_tasks = []
        self.max_concurrent_workers = max_concurrent_workers
        self.enable_work_coordination = enable_work_coordination

        # Processing state
        self.is_running = False
        self.is_processing = False
        self.last_processing_time = 0.0

        # Initialize completion time to current time so first call isn't delayed
        import time

        self.last_completion_time = time.time()

        self.cooldown_seconds = cooldown_seconds
        self.active_workers = 0

        # Adaptive frequency tracking
        self.recent_processing_times = []  # Rolling window of recent processing times
        self.max_recent_samples = 10  # Keep last 10 processing times for adaptive calculation
        self.adaptive_cooldown = cooldown_seconds  # Dynamic cooldown based on actual performance
        self.min_cooldown = 0.1  # Minimum cooldown to prevent overwhelming
        self.max_cooldown = 10.0  # Maximum cooldown as safety limit

        # Performance tracking
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.last_performance_log = 0.0

        # Tracking
        self.accumulated_data = ""
        self.last_processed_data = ""

        # Work coordination and deduplication
        if self.enable_work_coordination:
            self._coordination_lock = asyncio.Lock()
            self._processing_items: Set[str] = set()  # Track items being processed
            self._processed_hashes: Set[str] = set()  # Track completed work by content hash
            self._max_processed_hashes = 100  # Limit memory usage
            logger.info(f"{self.__class__.__name__}: Work coordination enabled")

    async def start_worker(self):
        """Start the worker task for processing with coordination support."""
        if self.worker_tasks == [] and not self.is_running:
            self.is_running = True

            # For stateful operations, use single worker to avoid race conditions
            # For stateless operations, use multiple workers for performance
            actual_workers = 1 if self.enable_work_coordination and self._is_stateful_processor() else self.max_concurrent_workers

            if actual_workers != self.max_concurrent_workers:
                logger.info(f"{self.__class__.__name__}: Using single worker mode for stateful operations (requested {self.max_concurrent_workers}, using {actual_workers})")

            # Start workers in parallel for faster initialization
            self.worker_tasks = []
            worker_tasks_creation = []

            for i in range(actual_workers):
                # Create worker tasks in parallel - don't await each one
                worker_task = asyncio.create_task(self._worker())
                self.worker_tasks.append(worker_task)
                worker_tasks_creation.append(worker_task)

            # Small delay to allow workers to start up properly
            await asyncio.sleep(0.01)

            logger.info(f"{self.__class__.__name__} {len(self.worker_tasks)} workers started (coordination: {self.enable_work_coordination})")

    def _is_stateful_processor(self) -> bool:
        """Check if this processor manages stateful operations that require coordination.

        Override in subclasses that manage incremental state like parsing.
        """
        # Default: assume stateless unless overridden
        return False

    async def stop_worker(self):
        """Stop the worker task."""
        if self.worker_tasks:
            self.is_running = False
            try:
                # Signal workers to stop
                for _ in range(len(self.worker_tasks)):
                    await self.processing_queue.put(None)

                # Wait for all workers to complete with timeout
                await asyncio.wait_for(asyncio.gather(*self.worker_tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout stopping {self.__class__.__name__} workers, cancelling remaining tasks")
                for task in self.worker_tasks:
                    if not task.done():
                        task.cancel()
            except Exception as e:
                logger.warning(f"Error stopping {self.__class__.__name__} workers: {e}")
            finally:
                self.worker_tasks = []
                # Clear coordination state
                if self.enable_work_coordination:
                    self._processing_items.clear()
                    self._processed_hashes.clear()
                logger.info(f"{self.__class__.__name__} workers stopped")

    async def _worker(self):
        """Worker task that processes queued items with coordination."""
        worker_id = id(asyncio.current_task())
        logger.info(f"{self.__class__.__name__} worker {worker_id} started")

        while self.is_running:
            try:
                # Use timeout to prevent hanging workers
                item = await asyncio.wait_for(self.processing_queue.get(), timeout=5.0)

                if item is None:  # Shutdown signal
                    break

                # Handle work coordination if enabled
                if self.enable_work_coordination:
                    should_process, work_item = await self._coordinate_work(item, worker_id)
                    if not should_process:
                        self.processing_queue.task_done()
                        continue
                    actual_item = work_item.data
                else:
                    actual_item = item

                # Track active workers for better queue status
                self.active_workers += 1
                start_time = time.time()

                try:
                    # Process the item
                    await self._process_item(actual_item)

                    # Track performance metrics with adaptive frequency
                    processing_time = time.time() - start_time
                    self.total_processed += 1
                    self.total_processing_time += processing_time

                    # Record completion for adaptive frequency calculation
                    self._record_processing_completion(processing_time)

                    # Log performance occasionally
                    current_time = time.time()
                    if current_time - self.last_performance_log > 60.0:  # Every minute
                        avg_time = self.total_processing_time / max(self.total_processed, 1)
                        logger.info(f"{self.__class__.__name__} performance: {self.total_processed} items processed, avg time: {avg_time:.2f}s, adaptive cooldown: {self.adaptive_cooldown:.2f}s")
                        self.last_performance_log = current_time

                except Exception as e:
                    logger.warning(f"{self.__class__.__name__} worker {worker_id} processing failed: {e}")
                finally:
                    self.active_workers -= 1

                    # Clean up coordination state
                    if self.enable_work_coordination and isinstance(item, WorkItem):
                        await self._cleanup_work_coordination(item)

                    self.processing_queue.task_done()

            except asyncio.TimeoutError:
                # Worker timeout - continue to check if still running
                continue
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__} worker {worker_id}: {e}")

        logger.info(f"{self.__class__.__name__} worker {worker_id} stopped")

    async def _coordinate_work(self, item: Any, worker_id: int) -> tuple[bool, Optional[WorkItem]]:
        """Coordinate work to prevent duplicate processing.

        Returns:
            tuple: (should_process, work_item)
        """
        work_item = item if isinstance(item, WorkItem) else WorkItem(item)

        async with self._coordination_lock:
            # Check if this exact item is already being processed
            if work_item.item_id in self._processing_items:
                logger.debug(f"Worker {worker_id}: Skipping duplicate item {work_item.item_id}")
                return False, None

            # Check if we've already processed this content
            content_hash = work_item.get_content_hash()
            if content_hash in self._processed_hashes:
                logger.debug(f"Worker {worker_id}: Skipping already processed content {content_hash[:8]}")
                return False, None

            # Mark as being processed
            self._processing_items.add(work_item.item_id)
            logger.debug(f"Worker {worker_id}: Processing item {work_item.item_id} (hash: {content_hash[:8]})")

            return True, work_item

    async def _cleanup_work_coordination(self, work_item: WorkItem):
        """Clean up coordination state after processing."""
        async with self._coordination_lock:
            # Remove from processing set
            self._processing_items.discard(work_item.item_id)

            # Add to processed hashes
            content_hash = work_item.get_content_hash()
            self._processed_hashes.add(content_hash)

            # Limit memory usage for processed hashes
            if len(self._processed_hashes) > self._max_processed_hashes:
                # Remove oldest hashes (simple FIFO approximation)
                oldest_hashes = list(self._processed_hashes)[:10]
                for old_hash in oldest_hashes:
                    self._processed_hashes.discard(old_hash)

    @abstractmethod
    async def _process_item(self, item: Any):
        """Process a single item from the queue.

        Args:
            item: The item to process
        """
        pass

    def should_process(self, data: str, min_size: int = 100) -> bool:
        """Check if processing should be triggered based on adaptive frequency matching actual processing times.

        Args:
            data: Current accumulated data
            min_size: Minimum data size to trigger processing

        Returns:
            bool: True if processing should be triggered
        """
        import time

        current_time = time.time()

        # Calculate adaptive cooldown based on recent processing times
        self._update_adaptive_cooldown()

        # Use completion time rather than queue time for frequency calculation
        # This ensures we trigger new requests as soon as the previous ones actually complete
        time_since_completion = current_time - self.last_completion_time

        # Skip if we haven't waited long enough based on adaptive frequency
        # Use 90% of adaptive cooldown to slightly favor higher frequency
        adaptive_threshold = self.adaptive_cooldown * 0.9
        if time_since_completion < adaptive_threshold:
            logger.debug(f"{self.__class__.__name__}: Waiting for adaptive frequency (completed {time_since_completion:.2f}s ago, need {adaptive_threshold:.2f}s)")
            return False

        # Skip if queue is getting full (but not completely full)
        queue_threshold = int(self.processing_queue.maxsize * 0.9)  # 90% capacity
        if self.processing_queue.qsize() >= queue_threshold:
            logger.warning(f"{self.__class__.__name__} queue nearly full ({self.processing_queue.qsize()}/{self.processing_queue.maxsize}), throttling")
            return False

        # Check minimum size requirement
        if len(data) < min_size:
            return False

        return True

    def _update_adaptive_cooldown(self):
        """Update adaptive cooldown based on recent processing times.

        This ensures call frequency matches actual LLM processing speed:
        - If LLM calls take 1 second on average, trigger new calls every ~1 second
        - If LLM calls take 3 seconds, trigger every ~3 seconds
        - Always maintain non-blocking behavior
        """
        if not self.recent_processing_times:
            return

        # Calculate average processing time from recent samples
        avg_processing_time = sum(self.recent_processing_times) / len(self.recent_processing_times)

        # Set adaptive cooldown to match processing time for optimal frequency
        # Add small buffer (10%) to account for variance and prevent queue buildup
        target_cooldown = avg_processing_time * 1.1

        # Apply bounds to prevent extreme values
        previous_cooldown = self.adaptive_cooldown
        self.adaptive_cooldown = max(self.min_cooldown, min(self.max_cooldown, target_cooldown))

        # Log significant changes in adaptive frequency
        if abs(self.adaptive_cooldown - previous_cooldown) > 0.1:
            frequency_hz = 1.0 / self.adaptive_cooldown if self.adaptive_cooldown > 0 else 0
            logger.info(f"{self.__class__.__name__}: 🎯 Adaptive frequency updated to {frequency_hz:.1f} Hz (every {self.adaptive_cooldown:.2f}s) based on avg processing time {avg_processing_time:.2f}s")
        else:
            logger.debug(f"{self.__class__.__name__}: Adaptive cooldown updated to {self.adaptive_cooldown:.2f}s (avg processing: {avg_processing_time:.2f}s)")

    def _record_processing_completion(self, processing_time: float):
        """Record completion of a processing operation for adaptive frequency calculation.

        Args:
            processing_time: Time taken to complete the processing
        """
        import time

        # Update completion time
        self.last_completion_time = time.time()

        # Add to recent processing times for adaptive calculation
        self.recent_processing_times.append(processing_time)

        # Keep only recent samples
        if len(self.recent_processing_times) > self.max_recent_samples:
            self.recent_processing_times.pop(0)

        # Update adaptive cooldown immediately
        self._update_adaptive_cooldown()

        # Log processing completion for visibility
        if len(self.recent_processing_times) >= 3:  # After a few samples
            frequency_hz = 1.0 / self.adaptive_cooldown if self.adaptive_cooldown > 0 else 0
            logger.debug(f"{self.__class__.__name__}: ✅ Processing completed in {processing_time:.2f}s, optimal frequency: {frequency_hz:.1f} Hz")

    async def queue_for_processing(self, item: Any) -> bool:
        """Queue an item for processing with deduplication support.

        Args:
            item: Item to queue

        Returns:
            bool: True if successfully queued, False otherwise
        """
        try:
            # Wrap item for coordination if enabled
            if self.enable_work_coordination:
                work_item = item if isinstance(item, WorkItem) else WorkItem(item)

                # Quick check for recent duplicates before queuing
                async with self._coordination_lock:
                    content_hash = work_item.get_content_hash()
                    if content_hash in self._processed_hashes:
                        logger.debug(f"Skipping duplicate content before queuing: {content_hash[:8]}")
                        return False

                self.processing_queue.put_nowait(work_item)
            else:
                self.processing_queue.put_nowait(item)

            self.new_event.set()
            return True
        except asyncio.QueueFull:
            logger.warning(f"{self.__class__.__name__} queue full, dropping item")
            return False

    def update_processing_time(self):
        """Update the last processing time."""
        import time

        self.last_processing_time = time.time()

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status including adaptive frequency information.

        Returns:
            dict: Queue status information
        """
        avg_processing_time = self.total_processing_time / max(self.total_processed, 1)

        status = {
            "queue_size": self.processing_queue.qsize(),
            "active_workers": self.active_workers,
            "is_running": self.is_running,
            "has_workers": self.worker_tasks != [],
            "max_workers": self.max_concurrent_workers,
            "total_processed": self.total_processed,
            "avg_processing_time": avg_processing_time,
            "adaptive_cooldown": self.adaptive_cooldown,
            "recent_processing_times": self.recent_processing_times[-3:],  # Last 3 samples for debugging
            "optimal_frequency_hz": 1.0 / self.adaptive_cooldown if self.adaptive_cooldown > 0 else 0,
        }

        # Add coordination info if enabled
        if self.enable_work_coordination:
            status.update(
                {
                    "work_coordination_enabled": True,
                    "processing_items": len(getattr(self, "_processing_items", set())),
                    "processed_hashes": len(getattr(self, "_processed_hashes", set())),
                }
            )

        return status

    async def scale_workers(self, target_workers: int):
        """Dynamically scale the number of workers up or down.

        Args:
            target_workers: Target number of concurrent workers
        """
        if not self.is_running:
            logger.warning(f"Cannot scale {self.__class__.__name__} workers when not running")
            return

        # For stateful processors, enforce single worker
        if self.enable_work_coordination and self._is_stateful_processor():
            if target_workers != 1:
                logger.info(f"{self.__class__.__name__}: Stateful processor - maintaining single worker (requested {target_workers})")
            return

        current_workers = len(self.worker_tasks)

        if target_workers == current_workers:
            return  # Already at target

        if target_workers > current_workers:
            # Scale up - add new workers
            for i in range(target_workers - current_workers):
                worker_task = asyncio.create_task(self._worker())
                self.worker_tasks.append(worker_task)
                await asyncio.sleep(0.01)  # Small delay between new workers

            self.max_concurrent_workers = target_workers
            logger.info(f"Scaled {self.__class__.__name__} workers up to {target_workers}")

        else:
            # Scale down - stop excess workers
            workers_to_stop = current_workers - target_workers
            for _ in range(workers_to_stop):
                await self.processing_queue.put(None)  # Signal worker to stop

            # Wait a bit for workers to stop gracefully
            await asyncio.sleep(0.1)

            # Remove stopped tasks from the list
            self.worker_tasks = [task for task in self.worker_tasks if not task.done()]
            self.max_concurrent_workers = target_workers
            logger.info(f"Scaled {self.__class__.__name__} workers down to {target_workers}")


class UniversalLLM:
    """
    Universal LLM client that provides a consistent interface for all LLM operations.

    This class handles:
    - LLM initialization with proper API configuration
    - Structured and unstructured output generation
    - Async execution with proper error handling
    - Consistent logging and error management
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the universal LLM client.

        Args:
            config: Configuration for the LLM client
        """
        self.config = config or LLMConfig()
        self._llm = None
        self._structured_llms = {}  # Cache for structured LLMs by output type

    def _get_llm(self) -> ChatOpenAI:
        """Get or create the base LLM instance."""
        if self._llm is None:
            api_key, base_url = get_api_credentials()

            if self.config.api_key:
                api_key = self.config.api_key

            if not api_key:
                raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.")

            self._llm = ChatOpenAI(
                model=self.config.model_id,
                api_key=api_key,
                base_url=base_url if "/" in self.config.model_id else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                timeout=self.config.timeout,
            )

            logger.info(f"Initialized LLM with model: {self.config.model_id}")

        return self._llm

    def get_structured_llm(self, output_schema: BaseModel) -> ChatOpenAI:
        """Get a structured LLM for a specific output schema.

        Args:
            output_schema: Pydantic model defining the output structure

        Returns:
            ChatOpenAI: LLM configured for structured output
        """
        schema_name = output_schema.__name__

        if schema_name not in self._structured_llms:
            base_llm = self._get_llm()

            # Use function_calling method for ChatOpenAI compatibility
            if isinstance(base_llm, ChatOpenAI):
                structured_llm = base_llm.with_structured_output(output_schema, method="function_calling")
            else:
                structured_llm = base_llm.with_structured_output(output_schema)

            self._structured_llms[schema_name] = structured_llm
            logger.debug(f"Created structured LLM for schema: {schema_name}")

        return self._structured_llms[schema_name]

    async def invoke_text(self, prompt: ChatPromptTemplate, variables: Dict[str, Any]) -> str:
        """Invoke LLM for text output.

        Args:
            prompt: Chat prompt template
            variables: Variables to fill in the prompt

        Returns:
            str: Generated text content
        """
        try:
            llm = self._get_llm()
            chain = prompt | llm

            # Use shared executor instead of creating new one each time
            executor = get_shared_executor()
            result = await asyncio.get_event_loop().run_in_executor(executor, lambda: chain.invoke(variables))

            return result.content.strip()

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise

    async def invoke_structured(self, prompt: ChatPromptTemplate, variables: Dict[str, Any], output_schema: BaseModel) -> BaseModel:
        """Invoke LLM for structured output.

        Args:
            prompt: Chat prompt template
            variables: Variables to fill in the prompt
            output_schema: Pydantic model for structured output

        Returns:
            BaseModel: Structured response object
        """
        try:
            structured_llm = self.get_structured_llm(output_schema)
            chain = prompt | structured_llm

            # Use shared executor instead of creating new one each time
            executor = get_shared_executor()
            result = await asyncio.get_event_loop().run_in_executor(executor, lambda: chain.invoke(variables))

            return result

        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            raise

    async def invoke_batch_text(self, prompt: ChatPromptTemplate, variable_list: list[Dict[str, Any]]) -> list[str]:
        """Invoke LLM for multiple text outputs in batch.

        Args:
            prompt: Chat prompt template
            variable_list: List of variable dictionaries

        Returns:
            list[str]: List of generated text content
        """
        try:
            llm = self._get_llm()
            chain = prompt | llm

            # Use shared executor instead of creating new one each time
            executor = get_shared_executor()
            results = await asyncio.get_event_loop().run_in_executor(executor, lambda: chain.batch(variable_list))

            return [result.content.strip() for result in results]

        except Exception as e:
            logger.error(f"Batch text generation failed: {e}")
            raise

    async def invoke_batch_structured(self, prompt: ChatPromptTemplate, variable_list: list[Dict[str, Any]], output_schema: BaseModel) -> list[BaseModel]:
        """Invoke LLM for multiple structured outputs in batch.

        Args:
            prompt: Chat prompt template
            variable_list: List of variable dictionaries
            output_schema: Pydantic model for structured output

        Returns:
            list[BaseModel]: List of structured response objects
        """
        try:
            structured_llm = self.get_structured_llm(output_schema)
            chain = prompt | structured_llm

            # Use shared executor instead of creating new one each time
            executor = get_shared_executor()
            results = await asyncio.get_event_loop().run_in_executor(executor, lambda: chain.batch(variable_list))

            return results

        except Exception as e:
            logger.error(f"Batch structured generation failed: {e}")
            raise

    def update_config(self, **kwargs):
        """Update LLM configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated LLM config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

        # Clear cached LLM instances to apply new config
        self._llm = None
        self._structured_llms.clear()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration.

        Returns:
            dict: Model configuration information
        """
        api_key, base_url = get_api_credentials()

        return {
            "model_id": self.config.model_id,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "timeout": self.config.timeout,
            "provider": "openrouter" if base_url else "openai",
            "has_api_key": bool(api_key or self.config.api_key),
        }
