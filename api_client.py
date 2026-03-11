from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import AsyncOpenAI
from config import config

# Load the .env file to set environment variables (like OPENAI_API_KEY)
load_dotenv()


def create_api_client():
    """Factory function to create API client singleton instances based on the configuration."""
    creators = {
        "openai": OpenAIClient(),
    }
    return creators[config["api_client"]["type"]]


api_client = (
    create_api_client()
)  # Create a singleton instance of the API client based on the configuration


class APIClient(ABC):
    def __init__(self):
        self.parameters = config["api_client"]["parameters"]

    @abstractmethod
    def get_summary(self, prompt: str) -> str:
        pass


class OpenAIClient(APIClient):
    """Initialize OpenAI client (it automatically picks up the OPENAI_API_KEY environment variable)"""

    def __init__(self):
        super().__init__()
        self.client = AsyncOpenAI()

    async def get_summary(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.parameters["model"],
            messages=self.parameters["messages"]
            + [
                {
                    "role": "user",
                    "content": f"Summarize this text in one paragraph: {prompt}",
                }
            ],
            temperature=self.parameters["temperature"],
            max_completion_tokens=self.parameters["max_completion_tokens"],
        )

        # Extract the summary content from the response
        summary = response.choices[0].message.content.strip()
        return summary
