from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from dotenv import load_dotenv
from time import sleep
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
            reply = model.generate(
                **inputs,
                # num_beams=num_replies,
                # no_repeat_ngram_size=2,
                # early_stopping=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            reply = tokenizer.batch_decode(reply, skip_special_tokens=True)[0]
            comment.reply(body=reply)
            sleep(220)


main()
