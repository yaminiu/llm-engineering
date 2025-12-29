from openai import OpenAI
import sys
import json

OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')

def build_message_roles(scraping_context, scraping_task):
    system_content = (
"""You are expertise of rust programming laugne.to assiist with web scraping tasks.
Let's model a road trip!
Define a `start_trip` function that creates and returns
a String of "The plan is..."
Invoke the `start_trip` function in `main` and save its
return value to a `trip` variable.
We want to pass the String to three separate functions
that will mutate the String without transferring ownership.
Define a `visit_philadelphia` function that concatenates
the text "Philadephia" to the end of the String. Invoke
the function in `main`. Then, invoke `push_str` on the String
to concatenate the content " and " to the end. Mak sure to
include the spaces.
Define a `visit_new_york` function that concatenates the
text "New York" to the end of the String. Invoke the function
in `main`. Repeat the previous logic to concatenate " and "
to the end of the String.
Define a `visit_boston` function that concatenates the
text "Boston." to the end of the String. Invoke the function
in `main`. Concatenate a period to the end of the
String/sentence.
Define a `show_itinerary` function that will print out
the final version of the String. Find a way to do so
without transferring ownership.
Invoke `show_itinerary`. The final output should be:
"The plan is...Philadelphia and New York and Boston."
"""
      
    )
    user_content = f"Target website: {scraping_context.strip()}\nExtraction task: {scraping_task.strip()}\nPlease extract the specified data and provide results in JSON format."
    assistant_content = (
        "I will use Python web scraping libraries to fetch and parse the target web page, extract the requested data, and return it in JSON format. "
        "If needed, I will leverage an LLM to interpret complex HTML and refine extraction logic."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": "run the scripts and get the real data. install pip libraries if needed."},
    ]

def main():
    if len(sys.argv) >= 3:
        scraping_context = sys.argv[1]
        scraping_task = sys.argv[2]
    else:
        print("Enter the target website URL to scrape:")
        scraping_context = input("> ")
        print("Enter the data extraction task (e.g., 'Extract all product titles'):")
        scraping_task = input("> ")

    message_roles = build_message_roles(scraping_context, scraping_task)

    print("\nGenerated message roles (JSON):")
    print(json.dumps(message_roles, indent=2))

    # Optionally, send to AI model and print result
    stream = ollama.chat.completions.create(
        model="gemma3:latest",
        messages=message_roles,
        temperature=0.2,
        stream=True,
    )

    print("\nAI Assistant Reply:")
    buf = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        token = getattr(delta, "content", "")
        if token:
            buf.append(token)
            print(token, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    main()
