import csv
import os
from time import sleep

import click
import pandas as pd
import praw
from tqdm import tqdm
from deep_fields import data_path
import subprocess


@click.command()
@click.option('--subreddits', '-s',  required=True, multiple=True, help='List of subreddits')
@click.option('--resume', is_flag=True, help='Continue Downloading')
def main(subreddits, resume):
    reddit = praw.Reddit(
        user_agent="WWW-Redit",
        client_id="i5KMLHdncVNq0g",
        client_secret="hC77k9pz9DbuaZ7sS5NJh8nJYF4Xzw",
    )

    subreddits = list(subreddits) if isinstance(
        subreddits, list) else subreddits

    for subreddit in subreddits:
        subs: pd.DataFrame = pd.read_csv(subreddit, index_col='id', usecols=['id'])

        base_path = os.path.dirname(subreddit)
        file_name = os.path.split(subreddit)[1]
        path = os.path.join(base_path, '../comments/')
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, file_name)
        file_mode = 'w'
        submissions_to_download = subs.index
        if resume:
            file_mode = 'a'
            last_submission_id = get_last_submission_id(file_path)
            submissions_to_download = subs.loc[last_submission_id:].index[1:]
        with open(file_path, file_mode, encoding="utf-8", newline='') as f:
            try:
                writer = csv.writer(f, delimiter=',')
                if not resume:
                    writer.writerow(["id", "submission_id", "body", "created_utc", "permalink", "score", "author"])
                pbar = tqdm(submissions_to_download, desc=f"Fetching comments for subbreddit {file_name}")
                for sub_id in pbar:
                    sub_comments = download_comments(reddit, sub_id, 0, pbar)
                    if not sub_comments:
                        sleep(2)
                        continue
                    writer.writerows(sub_comments)
                    sleep(2)
                if pbar.n % 50 == 0:
                    f.flush()
            except Exception as e:
                print(e)
            finally:
                f.flush()

            print('\n\n')


def get_last_submission_id(filename):
    line = subprocess.check_output(['tail', '-1', filename])
    return line.decode().split(',')[1]


def download_comments(reddit_api, sub_id, comments_cap, pbar: tqdm):
    """
    Download Comments

    Solution from https://praw.readthedocs.io/en/latest/tutorials/comments.html
    """

    out_comments = []
    try:
        submission_data = reddit_api.submission(id=sub_id)

        pbar.set_description_str(f'with comments:  {submission_data.num_comments}')
        if submission_data.num_comments == 0:
            return out_comments
        submission_data.comments.replace_more(limit=comments_cap)
        comments = submission_data.comments.list()
    except Exception:
        print(f"Submission not found: `{sub_id}`")
        return
    for comment in comments:
        if comment.author == 'AutoModerator':
            continue
        _data = [
            comment.id,
            sub_id,
            comment.body.replace('\n', '\\n'),
            int(comment.created_utc),
            comment.permalink,
            comment.score,
            comment.author
        ]
        out_comments.append(_data)
    return out_comments


if __name__ == '__main__':
    main()
