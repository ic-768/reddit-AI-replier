from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from dotenv import load_dotenv
from time import sleep
import praw
from praw.models import Comment
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


def snooze():
    sleep(170)


# def is_promising(comment):
#    return comment.startswith(
#        (
#            "will",
#            "are",
#            "is",
#            "would",
#            "did",
#            "what",
#            "does",
#            "who",
#            "why",
#            "has ",
#            "how",
#            "when",
#            "where",
#        )
#    )


def generate_response(input):
    inputs = tokenizer(
        input,
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
    return tokenizer.batch_decode(reply, skip_special_tokens=True)[0]


def should_reply(text):
    return (
        len(text) < 90 and len(text) > 15 and not "*" in text
    )  # and is_promising(text)


def is_mod_comment(text):
    return "am a bot" in text or "moderators" in text


def main():
   for comment in subreddit.stream.comments(skip_existing=True):
       body = comment.body.lower()
       if should_reply(body) and not is_mod_comment(body):
           reply = generate_response(comment.body)
           try:
               comment.reply(body=reply)
               sleep(150)
           except Exception as e:
               print(e)


main()


#def main():
#    for comment in subreddit.stream.comments(skip_existing=True, pause_after=-1):
#            for i in reddit.inbox.unread(limit=None):
#                if isinstance(
#                    i, Comment
#                ):  # make sure it's a comment, because it could also be a message
#                    try:
#                        if not is_mod_comment(i.body.lower()):
#                            reply = generate_response(i.body)
#                            i.reply(body=reply)
#                            i.mark_read()
#                            snooze()
#                    except Exception as e:
#                        print("exception in reply", e)
#            if comment is not None:
#                body = comment.body.lower()
#                if should_reply(body) and not is_mod_comment(body):
#                    reply = generate_response(comment.body)
#                    try:
#                        comment.reply(body=reply)
#                        snooze()
#                    except Exception as e:
#                        print("exception in reply", e)
#
#
#main()
