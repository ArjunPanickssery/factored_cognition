from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

def call_openai(prompt, max_tokens=10):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def extract_topics(question):
    prompt = "Given the following question, list all topics that are relevant to answering it, including people, places, and things.\n\n"
    prompt = "Do not respond with anything other than the topics themselves. List each topic on a new line.\n\n"
    prompt += f"Question: {question}\n"
    response = call_openai(prompt, max_tokens=100)
    topics = response.split("\n")
    return topics


def construct_subprompt(notes, paragraph, topics):
    subprompt = "Notes:\n"
    for i, note in enumerate(notes):
        subprompt += f"{i}: {note}\n"
    subprompt += "\n"
    subprompt += f"Given the following paragraph, record a new note including any information that may be relevant to the following topics: {', '.join(topics)}\n"
    subprompt += "Do not add any number to the beginning of your note.\n\n"
    subprompt += f"Paragraph: {paragraph}\n"
    return subprompt


def construct_synthesis_prompt(notes, question, options):
    prompt = "Notes:\n"
    for i, note in enumerate(notes):
        prompt += f"{i}: {note}\n"

    prompt += "\n"
    prompt += f"Given these notes, {question}\n"

    for i, option in enumerate(options):
        prompt += f"{i+1}: {option}\n"

    return prompt


def call_agent(article, question, options):
    topics = extract_topics(question)

    notes = []
    paragraphs = article.split("\n")

    for paragraph in paragraphs:
        prompt = construct_subprompt(notes, paragraph, topics)
        response = call_openai(prompt)
        notes.append(response)

    prompt = construct_synthesis_prompt(notes, question, options)
    response = call_openai(prompt)
    return response
