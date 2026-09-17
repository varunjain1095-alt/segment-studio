import os
import time
import logging
from anthropic import Anthropic, APIConnectionError, APITimeoutError, RateLimitError
from dotenv import load_dotenv
from src.utils.logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger("claude_client")

class ClaudeClient:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY is not set in environment variables.")
        self.client = Anthropic(api_key=self.api_key)

    def call(self, prompt: str, system_prompt: str = "", max_retries: int = 5, initial_delay: float = 1.0) -> str:
        """Calls Anthropic Claude with prompt, system_prompt, retry logic, and logging."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")

        logger.info(
            f"Sending request to Claude: prompt_len={len(prompt)}, sys_prompt_len={len(system_prompt)}",
            extra={"prompt_length": len(prompt), "system_prompt_length": len(system_prompt), "model": self.model}
        )

        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    system=system_prompt if system_prompt else None,
                    messages=[{"role": "user", "content": prompt}]
                )
                latency = time.time() - start_time
                response_text = "".join([block.text for block in response.content if hasattr(block, "text")])
                
                logger.info(
                    f"Claude call successful: response_len={len(response_text)}, latency={latency:.2f}s",
                    extra={
                        "prompt_length": len(prompt),
                        "model": self.model,
                        "response_length": len(response_text),
                        "latency": latency
                    }
                )
                return response_text

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                logger.warning(
                    f"Claude API transient error on attempt {attempt}/{max_retries}: {str(e)}. Retrying in {delay:.2f}s..."
                )
                if attempt == max_retries:
                    logger.error("Claude call failed: maximum retries reached.")
                    raise e
                time.sleep(delay)
                delay *= 2.0  # Exponential backoff
            except Exception as e:
                logger.error(f"Claude call failed with non-retryable error: {str(e)}", exc_info=True)
                raise e

# Create a singleton instance
claude_client = ClaudeClient()
