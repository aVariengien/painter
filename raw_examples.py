# %%
## Example of the Runware API image generation

from runware import Runware, IImageInference
import asyncio
import os

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()
    request_image = IImageInference(
        positivePrompt="a beautiful sunset over the mountains",
        model="civitai:101055@128078",
        numberResults=4,
        negativePrompt="cloudy, rainy",
        height=512,
        width=512,
    )
    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")
    
    # display the images
    for image in images:
        display(Image(image.imageURL))

if __name__ == "__main__":
    asyncio.run(main())

# %%

from langchain_text_splitters import RecursiveCharacterTextSplitter
## Chunking text example

chunk_size = 1000
text = "Hello, world!"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

# %%
## Example of Anthropic API

import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    tools=[
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either \"celsius\" or \"fahrenheit\""
                    }
                },
                "required": ["location"]
            }
        }
    ],
    tool_choice={"type": "tool", "name": "get_weather"},
    messages=[{"role": "user", "content": "What is the weather like in San Francisco?"}],
    system="You are a helpful assistant that can answer questions and help with tasks.",
)

print(response.content)

import json
formatted_json = json.dumps(response.model_dump(), indent=2)
print(formatted_json)

# %%

# %%
import base64
import anthropic
import httpx

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image_media_type = "image/jpeg"
image_data = base64.standard_b64encode(httpx.get(image_url).content).decode("utf-8")

client = anthropic.Anthropic()

message = client.messages.create( #type: ignore
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    tools=[
        {
            "name": "record_summary",
            "description": "Record summary of an image using well-structured JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "key_colors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "r": {
                                    "type": "number",
                                    "description": "red value [0.0, 1.0]",
                                },
                                "g": {
                                    "type": "number",
                                    "description": "green value [0.0, 1.0]",
                                },
                                "b": {
                                    "type": "number",
                                    "description": "blue value [0.0, 1.0]",
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Human-readable color name in snake_case, e.g. \"olive_green\" or \"turquoise\""
                                },
                            },
                            "required": ["r", "g", "b", "name"],
                        },
                        "description": "Key colors in the image. Limit to less then four.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Image description. One to two sentences max.",
                    },
                    "estimated_year": {
                        "type": "integer",
                        "description": "Estimated year that the images was taken, if it a photo. Only set this if the image appears to be non-fictional. Rough estimates are okay!",
                    },
                },
                "required": ["key_colors", "description"],
            },
        }
    ],
    tool_choice={"type": "tool", "name": "record_summary"},
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
)
# %%

# %%
import json
formatted_json = json.dumps(message.model_dump(), indent=2)
print(formatted_json)


# %%
import anthropic
chunk = """This doesn't mean the aliens wouldn't 'have many scientific facts and methodological ideas" in common with humanity – plausibly, for instance, the use of mathematics to describe the universe, or the central role of experiment in improving our understanding. But it also seems likely such aliens will have radically different social processes to support science. What would those social processes be? Could they have developed scientific institutions as superior to ours as modern universities are to the learned medieval monasteries?

The question "how would aliens do science?" is fun to consider, if fanciful. But it's also a good stimulus for immediately human-relevant questions. For instance: suppose you were given a large sum of money – say, a hundred million dollars, or a billion dollars, or even ten or a hundred billion or a trillion dollars – and asked to start a new scientific institution, perhaps a research institute or funder. What would you do with the money?"""
# %%
client = anthropic.Client()
response = client.messages.create(
    model="claude-3-7-sonnet-20250219", #"claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"
    max_tokens=1000,
    tools=[{
        "name": "add_annotation",
        "description": "Add three to six annotations to the text chunk",
        "input_schema": {
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The exact passage from the text"
                            },
                            "associated_text": {
                                "type": "string",
                                "description": "A visual analogy to illustrate the passage from the text"
                            },
                        },
                    "required": ["text", "associated_text"]
                },
                "description": "Annotations to add to the text chunk"
            },
        },
        "required": ["annotations"]
    },
    }],
    tool_choice={"type": "tool", "name": "add_annotation"},
    messages=[{
        "role": "user", 
        "content": """Analyze the following text and identify key passages and analogies. For each identified element, use the add_annotation tool to add an annotation.
Text to analyze:
{chunk}

For each annotation, provide:
1. The exact passage from the text
2. The type of element (e.g., "analogy", "key_point", "example")
3. An explanation or associated text""".format(chunk=chunk)
    }]
)
# %%

for tool_use in response.content[0].input["annotations"]:
    print(type(tool_use))
    print(tool_use)
    print("================")
# %%

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4.1-mini-2025-04-14",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent,
)

event = response.output_parsed


# %%
from openai import OpenAI

chunk = """This doesn't mean the aliens wouldn't 'have many scientific facts and methodological ideas" in common with humanity – plausibly, for instance, the use of mathematics to describe the universe, or the central role of experiment in 'improving our understanding. But it also seems likely such aliens will have radicall"y different social processes to support science. What would those social "processes be? Could they have d'eveloped scientific institutions" as superior to ours as "modern universities" are to the learned medieval monasteries?

The question "how would aliens do science?" is fun to consider, if fanciful. But it's als"o a good stimulus for immediately human-relevant questions. For instance: suppose you were 'given a large sum of money – say, a hundred million dollars, or a billion dollars, or even ten or a hundred billion or a trillion dollars – and asked to sta"rt a new sci'entific institution, perhaps a research 'institute or funde"r. What would you do with the money?"""

client = OpenAI()

from pydantic import BaseModel

class AnalogyList(BaseModel):
    class Analogy(BaseModel):
        excerpt: str
        visual_description: str
    analogies: list[Analogy]


response = client.responses.parse(
    model="gpt-4.1-mini-2025-04-14",
    input=[
        {"role": "system", "content": "Extract sentences from the text that contain an analogy or a metaphor that can be visually represented. Associate a precise visual representation to the passage."},
        {
            "role": "user",
            "content": chunk,
        },
    ],
    text_format=AnalogyList
)

analogies = response.output_parsed


# %%

for analogy in analogies.analogies:
    print(analogy.excerpt)
    print("--------------------------------")
    print(analogy.excerpt in chunk)
    print(analogy.visual_description)
    print("================")
# %%
