import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

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


class EventBasedProcessor(ABC):
    """
    Base class for event-based processing with queue management and worker tasks.

    Provides common functionality for:
    - Event-based triggering instead of polling
    - Queue-based processing to prevent overflow
    - Worker task management
    - Trigger condition checking
    - Concurrent processing for better performance
    """

    def __init__(self, queue_maxsize: int = 10, cooldown_seconds: float = 2.0, max_concurrent_workers: int = 2):
        """Initialize the event-based processor.

        Args:
            queue_maxsize: Maximum size of the processing queue
            cooldown_seconds: Minimum time between processing operations
            max_concurrent_workers: Maximum number of concurrent worker tasks
        """
        # Event-based triggering
        self.new_event = asyncio.Event()
        self.processing_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.worker_tasks = []
        self.max_concurrent_workers = max_concurrent_workers

        # Processing state
        self.is_running = False
        self.is_processing = False
        self.last_processing_time = 0.0
        self.cooldown_seconds = cooldown_seconds
        self.active_workers = 0

        # Performance tracking
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.last_performance_log = 0.0

        # Tracking
        self.accumulated_data = ""
        self.last_processed_data = ""

    async def start_worker(self):
        """Start the worker task for processing with staggered startup to reduce bottlenecks."""
        if self.worker_tasks == [] and not self.is_running:
            self.is_running = True

            # Start workers gradually to reduce initialization bottleneck
            self.worker_tasks = []
            for i in range(self.max_concurrent_workers):
                worker_task = asyncio.create_task(self._worker())
                self.worker_tasks.append(worker_task)

                # Small delay between worker creations to stagger initialization load
                if i < self.max_concurrent_workers - 1:  # Don't delay after the last worker
                    await asyncio.sleep(0.01)  # 10ms delay between workers

            logger.info(f"{self.__class__.__name__} workers started")

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
                logger.info(f"{self.__class__.__name__} workers stopped")

    async def _worker(self):
        """Worker task that processes queued items serially."""
        worker_id = id(asyncio.current_task())
        logger.info(f"{self.__class__.__name__} worker {worker_id} started")

        while self.is_running:
            try:
                # Use timeout to prevent hanging workers
                item = await asyncio.wait_for(self.processing_queue.get(), timeout=5.0)

                if item is None:  # Shutdown signal
                    break

                # Track active workers for better queue status
                self.active_workers += 1
                start_time = time.time()

                try:
                    # Process the item
                    await self._process_item(item)

                    # Track performance metrics
                    processing_time = time.time() - start_time
                    self.total_processed += 1
                    self.total_processing_time += processing_time

                    # Log performance occasionally
                    current_time = time.time()
                    if current_time - self.last_performance_log > 60.0:  # Every minute
                        avg_time = self.total_processing_time / max(self.total_processed, 1)
                        logger.info(f"{self.__class__.__name__} performance: {self.total_processed} items processed, avg time: {avg_time:.2f}s")
                        self.last_performance_log = current_time

                except Exception as e:
                    logger.warning(f"{self.__class__.__name__} worker {worker_id} processing failed: {e}")
                finally:
                    self.active_workers -= 1
                    self.processing_queue.task_done()

            except asyncio.TimeoutError:
                # Worker timeout - continue to check if still running
                continue
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__} worker {worker_id}: {e}")

        logger.info(f"{self.__class__.__name__} worker {worker_id} stopped")

    @abstractmethod
    async def _process_item(self, item: Any):
        """Process a single item from the queue.

        Args:
            item: The item to process
        """
        pass

    def should_process(self, data: str, min_size: int = 100) -> bool:
        """Check if processing should be triggered based on conditions.

        Args:
            data: Current accumulated data
            min_size: Minimum data size to trigger processing

        Returns:
            bool: True if processing should be triggered
        """
        import time

        current_time = time.time()

        # Skip if in cooldown - but reduced threshold for better responsiveness
        cooldown_threshold = self.cooldown_seconds * 0.8  # 20% reduction
        if (current_time - self.last_processing_time) < cooldown_threshold:
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

    async def queue_for_processing(self, item: Any) -> bool:
        """Queue an item for processing.

        Args:
            item: Item to queue

        Returns:
            bool: True if successfully queued, False otherwise
        """
        try:
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
        """Get current queue status.

        Returns:
            dict: Queue status information
        """
        avg_processing_time = self.total_processing_time / max(self.total_processed, 1)

        return {
            "queue_size": self.processing_queue.qsize(),
            "active_workers": self.active_workers,
            "is_running": self.is_running,
            "has_workers": self.worker_tasks != [],
            "max_workers": self.max_concurrent_workers,
            "total_processed": self.total_processed,
            "avg_processing_time": avg_processing_time,
        }

    async def scale_workers(self, target_workers: int):
        """Dynamically scale the number of workers up or down.

        Args:
            target_workers: Target number of concurrent workers
        """
        if not self.is_running:
            logger.warning(f"Cannot scale {self.__class__.__name__} workers when not running")
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
