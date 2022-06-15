from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from dotenv import load_dotenv
import praw
import os

load_dotenv("./.env")

mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)
num_replies = 3

# add these to .env file in project dir
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
)

subreddit = reddit.subreddit("all")


# User chooses with a number corresponding to a potential reply
def get_choice():
    try:
        choice = int(input())
        if choice < num_replies and choice > -1:
            return choice
        else:
            raise Exception("Invalid choice")
    except:
        print("Rejecting")


# Print choice and which number to press to select it
def print_pretty_choice(idx, choice):
    print(f"\033[0;32mReply {idx} >> {choice} \033[0m")


# Choose or reject interactions
def decide(comment, replies):
    print("\033[0;31mComment >> ", comment.body)
    [print_pretty_choice(i, reply) for i, reply in enumerate(replies)]
    choice = get_choice()
    if choice:
        comment.reply(body=replies[choice])


def is_promising(comment):
    return comment.startswith(
        (
            "what",
            "does",
            "who",
            "why",
            "has ",
            "how",
            "when",
            "where",
        )
    )


def main():
    for comment in subreddit.stream.comments(skip_existing=True):
        body = comment.body.lower()
        if len(body) < 90 and not "*" in body and is_promising(body):
            inputs = tokenizer(
                comment.body,
                return_tensors="pt",
            )
            replies = model.generate(
                **inputs,
                # num_beams=num_replies,
                # no_repeat_ngram_size=2,
                # early_stopping=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=num_replies,
            )
            replies = tokenizer.batch_decode(replies, skip_special_tokens=True)
            decide(comment, replies)


main()
