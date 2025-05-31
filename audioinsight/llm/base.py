import asyncio
from typing import Any, Dict, Optional, Union

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..logging_config import get_logger
from .types import LLMConfig
from .utils import get_api_credentials

logger = get_logger(__name__)


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

            result = await asyncio.get_event_loop().run_in_executor(None, lambda: chain.invoke(variables))

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

            result = await asyncio.get_event_loop().run_in_executor(None, lambda: chain.invoke(variables))

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

            results = await asyncio.get_event_loop().run_in_executor(None, lambda: chain.batch(variable_list))

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

            results = await asyncio.get_event_loop().run_in_executor(None, lambda: chain.batch(variable_list))

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
