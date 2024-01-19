import json
from typing import Optional
from bs4 import BeautifulSoup
import argparse
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))


# Function to convert HTML content to plain text
def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n").strip()


def format_prompt(article_text, question, question_options):
    """
    Formats the article text, question, and question options into a prompt.

    Args:
    - article_text (str): The text of the article.
    - question (str): The question to be answered.
    - question_options (list): A list of answer options for the question.

    Returns:
    - str: A formatted prompt string.
    """
    # Formatting the article text
    formatted_article = f"Article:\n{article_text}\n\n"

    # Formatting the question
    formatted_question = f"Question:\n{question}\n\n"

    # Formatting the options
    formatted_options = "Options:\n"
    for idx, option in enumerate(question_options, start=1):
        formatted_options += f"{idx}. {option}\n"

    # Combining all parts into one prompt
    prompt = formatted_article + formatted_question + formatted_options
    return prompt


def parse_response(response) -> Optional[int]:
    # Find the first number in the response
    for token in response.split():
        if token.isdigit():
            return int(token)

    # If no number is found, return None
    return None


load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))


def call_openai(prompt, max_tokens=5):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def call_openai(system_prompt, prompt, model, max_tokens=100, stop=None):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Say this is a test"}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


def main():
    # Parsing the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gpt-4", help="The openai model identifier."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1,
        help="The number of questions to ask per article.",
    )
    parser.add_argument(
        "--questions-file", type=str, help="The path to the questions file."
    )
    parser.add_argument(
        "--max-articles",
        type=Optional[int],
        default=1,
        help="The maximum number of articles to use.",
    )
    args = parser.parse_args()

    # Fetching the questions
    print("Fetching the questions...")
    data = []
    with open(args.questions_file, "r", encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))

    print(len(data))
    print(data[0].keys())

    print("Running the evaluation...")
    # Results will be stored as a dict with the following keys:
    # - article_id: The ID of the article.
    # - article_idx: The index of the article in the dataset.
    # - question_idx: The index of the question in the article.
    # - chosen_option_idx: The index of the option chosen by the model.
    # - expected_option_idx: The index of the correct option.
    # - correct: Whether the model chose the correct option.
    # - difficulty: The difficulty of the question.
    results = []
    for i, row in enumerate(data):
        if args.max_articles is not None and i >= args.max_articles:
            break

        # Extracting the article text
        article_text = html_to_text(row["article"])

        # Extracting the questions and their options
        questions_data = row["questions"][: args.num_questions]
        for j, question_data in enumerate(questions_data):
            question = question_data["question"]
            question_options = question_data["options"]
            expected_option_idx = question_data["gold_label"]

            # Formatting the prompt
            formatted_prompt = format_prompt(article_text, question, question_options)
            response = call_openai("", formatted_prompt, args.model)
            chosen_option_idx = parse_response(response)

            if chosen_option_idx is None:
                print(f"Failed to parse response: {response}")
                continue

            # Storing the results
            results.append(
                {
                    "article_id": row["article_id"],
                    "article_idx": i,
                    "question_idx": j,
                    "chosen_option_idx": chosen_option_idx,
                    "expected_option_idx": expected_option_idx,
                    "correct": chosen_option_idx == expected_option_idx,
                    "difficulty": question_data["difficulty"],
                }
            )


if __name__ == "__main__":
    main()
