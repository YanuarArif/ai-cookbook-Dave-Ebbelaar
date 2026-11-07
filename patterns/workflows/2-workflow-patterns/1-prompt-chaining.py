import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, TypeVar

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file (project root & upward search)
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")
load_dotenv(find_dotenv())

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Provider configuration: auto-detect DeepSeek or OpenAI
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if deepseek_key:
    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )
    model = "deepseek-chat"
elif openai_key:
    client = OpenAI(api_key=openai_key)
    model = "gpt-4o"
else:
    raise ValueError(
        "No API key found. Please define DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file."
    )

# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        default=None,
        description="Generated calendar link if applicable",
    )


# --------------------------------------------------------------
# Helper: Universal Structured Output for OpenAI and DeepSeek
# --------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def get_structured_completion(
    messages: list[dict],
    response_model: type[T],
) -> T:
    """Handles structured output across OpenAI (parse) and DeepSeek (json_object)."""
    is_deepseek = "deepseek.com" in str(client.base_url)

    if is_deepseek:
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        format_instruction = (
            f"\n\nYou must respond strictly with a valid JSON object matching this schema:\n"
            f"{schema_json}\n"
            f"Ensure the output contains only the valid json string without additional commentary."
        )

        formatted_messages = [dict(m) for m in messages]
        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] += format_instruction
        else:
            formatted_messages.insert(
                0, {"role": "system", "content": format_instruction}
            )

        completion = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            response_format={"type": "json_object"},
        )
        content = (completion.choices[0].message.content or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return response_model.model_validate_json(content.strip())
    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
        )
        return completion.choices[0].message.parsed


# --------------------------------------------------------------
# Step 2: Define the functions
# --------------------------------------------------------------


def extract_event_info(user_input: str) -> EventExtraction:
    """First LLM call to determine if input is a calendar event"""
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    result = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event.",
            },
            {"role": "user", "content": user_input},
        ],
        response_model=EventExtraction,
    )
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )
    return result


def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    result = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.",
            },
            {"role": "user", "content": description},
        ],
        response_model=EventDetails,
    )
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result


def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    result = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign off with your name; Susie",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_model=EventConfirmation,
    )
    logger.info("Confirmation message generated successfully")
    return result


# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info(user_input)

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(initial_extraction.description)

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation


# --------------------------------------------------------------
# Step 4: Test the chain
# --------------------------------------------------------------

if __name__ == "__main__":
    user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

    result = process_calendar_request(user_input)
    if result:
        print(f"Confirmation: {result.confirmation_message}")
        if result.calendar_link:
            print(f"Calendar Link: {result.calendar_link}")
    else:
        print("This doesn't appear to be a calendar event request.")

    # --------------------------------------------------------------
    # Step 5: Test the chain with an invalid input
    # --------------------------------------------------------------

    user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"

    result = process_calendar_request(user_input)
    if result:
        print(f"Confirmation: {result.confirmation_message}")
        if result.calendar_link:
            print(f"Calendar Link: {result.calendar_link}")
    else:
        print("This doesn't appear to be a calendar event request.")
