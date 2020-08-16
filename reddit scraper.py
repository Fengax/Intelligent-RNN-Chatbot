import praw
from langdetect import detect

reddit = praw.Reddit(client_id="L_R7lS48EVgpOA",
                     client_secret="xh7kk4KhkvMyptypb3gEpHm6HfI",
                     password="odl36d5n",
                     user_agent="text scraper",
                     username="KesqiSePasse")

file = open("text.txt", "a")
punctuation = [".", ".", "'", '"', "?", "/", ")", "(", "!", ";", ":", "'"]

for submission in reddit.subreddit("prorevenge").stream.submissions():
    if submission.selftext != "":
        text = submission.selftext
        textarr = []
        try:
            if detect(text) != "en":
                continue
        except:
            continue

        text = text.replace("\n", "")

        for i in text:
            if i.isalnum() or i in punctuation or i == " ":
                textarr.append(i)

        text = ""
        for i in textarr:
            text = text + i

        file.write(text + " ")
        print("submission added")

    for comment in submission.comments:
        if comment.body != "":
            text = comment.body
            textarr = []
            try:
                if detect(text) != "en":
                    continue
            except:
                continue

            text = text.replace("\n", "")

            text = text.replace("\n", "")

            for i in text:
                if i.isalnum() or i in punctuation or i == " ":
                    textarr.append(i)

            text = ""
            for i in textarr:
                text = text + i
            file.write(text + " ")
            print("comment added")