import click
from deep_fields import data_path
from psaw import PushshiftAPI
import datetime as dt


import pandas as pd
import os
from tqdm import tqdm
# import praw


@click.command()
@click.option('--subreddits', '-s',  required=True, multiple=True, help='List of subreddits')
@click.option('--start-date', nargs=2,  type=click.Tuple([int, int]), required=True, help='Start date <month> <year> (ex. 1 2019)')
@click.option('--end-date', nargs=2, type=click.Tuple([int, int]), required=True, help='End date <month> <year> (ex. 9 2019)')
def main(subreddits, start_date: click.Tuple([int, int]), end_date: click.Tuple([int, int])):
    api = PushshiftAPI()

    start_month, start_year = start_date
    end_month, end_year = end_date
    assert start_month < end_month
    assert start_year <= end_year
    assert start_month > 0 and start_month < 13
    assert end_month > 0 and end_month < 13

    years = list(range(start_year, end_year + 1))
    months = list(range(start_month, end_month + 1))
    ROOT = os.path.join(data_path, 'reddit')

    subreddits = list(subreddits) if isinstance(subreddits, list) else subreddits

    for subreddit in subreddits:
        for year in years:
            for month in months:
                download_submissions(api, ROOT, subreddit, year, month)

        print('\n\n')


def download_submissions(api, ROOT, subreddit, year, month):
    start_time_epoch = int(dt.datetime(year=year, month=month, day=1, hour=0, minute=0).timestamp())

    if month < 12:
        end_time_epoch = int(dt.datetime(year=year, month=month + 1, day=1).timestamp())
    else:
        end_time_epoch = int(dt.datetime(year=year, month=month, day=31, hour=23, minute=59).timestamp())

    results = api.search_submissions(before=end_time_epoch,
                                     after=start_time_epoch,
                                     subreddit=subreddit,
                                    #  filter=['id', 'author', 'title', 'subreddit', 'num_comments', 'score', 'author_is_blocked', ],
                                    #   limit=50,
                                     max_results_per_request=1000)

    N = api.metadata_.get('total_results')

    def c(gen):
        return [row.d_ for row in tqdm(gen, total=N, desc=f'{subreddit}-{year}-{month}')]
    df: pd.DataFrame = pd.DataFrame(c(results))
    path = os.path.join(ROOT, f'{subreddit}/submissions/')
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, f'{month:02d}_{year}.csv'), index=False)


if __name__ == '__main__':
    main()
