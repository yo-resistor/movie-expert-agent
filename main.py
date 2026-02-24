import openai
import json
import requests
import os

from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessage
from datetime import datetime

load_dotenv()

client = openai.OpenAI()

messages = []

MOVIE_API_URL = "https://nomad-movies.nomadcoders.workers.dev"

def _movie_api_get(url: str) -> dict:
    """GET a movie API URL; return JSON or a dict with 'error' and 'detail' on failure."""
    response = requests.get(url)
    if not response.ok:
        return {"error": f"API returned {response.status_code}", "detail": response.text.strip()}
    try:
        return response.json()
    except json.JSONDecodeError as e:
        return {"error": "Invalid response from movie API", "detail": response.text[:300] or str(e)}

def get_popular_movies() -> dict:   
    """Get information about popular movies from the endpoint: {MOVIE_API_URL}/movies."""
    return _movie_api_get(f"{MOVIE_API_URL}/movies")

def get_movie_details(id: str) -> dict:
    """Get detailed information about a movie from the endpoint: {MOVIE_API_URL}/movies/:id."""
    return _movie_api_get(f"{MOVIE_API_URL}/movies/{id}")

def get_movie_credits(id: str) -> dict:
    """Get credits for a movie from the endpoint: {MOVIE_API_URL}/movies/:id/credits."""
    return _movie_api_get(f"{MOVIE_API_URL}/movies/{id}/credits")

OUTPUTS_DIR = "outputs"

def save_result_to_markdown(content: str, filename: str | None = None) -> dict:
    """Save content to a markdown file in the outputs folder. Creates outputs/ if needed."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if not filename:
        filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    path = os.path.join(OUTPUTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"success": True, "path": path, "filename": filename}

# Define Function Map
FUNCTION_MAP = {
    "get_popular_movies": get_popular_movies,
    "get_movie_details": get_movie_details,
    "get_movie_credits": get_movie_credits,
    "save_result_to_markdown": save_result_to_markdown,
}

# Define Tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_popular_movies",
            "description": f"Get information about popular movies from the endpoint: {MOVIE_API_URL}/movies.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": f"Get detailed information about a movie from the endpoint: {MOVIE_API_URL}/movies/:id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string", 
                        "description": "The ID of the movie"
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_credits",
            "description": f"Get credits for a movie from the endpoint: {MOVIE_API_URL}/movies/:id/credits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string", 
                        "description": "The ID of the movie"
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_result_to_markdown",
            "description": "Save text or markdown content to a file in the outputs folder. Use when the user asks to save results, export a list, or write output to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The markdown or text content to save (e.g. formatted movie list, summary, or report)."
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional. Base name for the file (e.g. 'popular_movies.md'). If omitted, a timestamped name is used."
                    },
                },
                "required": ["content"],
            },
        },
    },
]

# Normalize Content
def _normalize_content(content) -> str:
    """Ensure content is always a string (API expects string or content parts with 'type')."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            getattr(part, "text", part.get("text", "")) if isinstance(part, dict) else getattr(part, "text", "")
            for part in content
        )
    return str(content)

# Process AI Response
def process_ai_response(message: ChatCompletionMessage) -> None:
    content = _normalize_content(message.content)
    if message.tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        # Call Tool Functions
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"Calling function: {function_name} with arguments: {arguments}")

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
            
            function_to_run = FUNCTION_MAP.get(function_name)
            
            result = function_to_run(**arguments) if function_to_run else "Function not found."

            print(f"Ran {function_name} with args {arguments} for a result of {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                }
            ) 

        call_ai()
    else:
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
            }
        )
        print("AI:", message.content)

# Call AI
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    process_ai_response(response.choices[0].message)

# Main Function
def main() -> None:
    while True:
        message = input("Send a message to the AI...")
        if message == "quit" or message == "exit" or message == "bye" or message == "q":
            break
        messages.append({"role": "user", "content": message})
        print("User:", message)
        call_ai()

if __name__ == "__main__":
    main()