from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from dotenv import load_dotenv
import praw
import re
import os

load_dotenv('./.env')

mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

# add these to .env file in project dir
reddit = praw.Reddit(client_id =os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent =os.getenv("REDDIT_USER_AGENT"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD")
        )

subreddit = reddit.subreddit('all')

# Accept or reject an interaction
def decide(comment,reply):
    print("\033[0;31mComment >> ",comment.body)
    print("\033[0;32mReply >> ",reply,"\033[0m")
    yesOrNo=input()
    if yesOrNo.lower() == "y":
        comment.reply(body=reply)
        print("posted")
    else:
        print('moving on')

def isPromising(text):
    return text.startswith(
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
        text=comment.body.lower()
        if len(text)<90 and not '*' in text and isPromising(text):
           inputs = tokenizer(comment.body, return_tensors="pt")
           reply_ids = model.generate(**inputs)
           reply=tokenizer.batch_decode(reply_ids)[0]
           formattedReply=re.sub('<.+?>\s?',"",reply)
           decide(comment,formattedReply)


main()
