from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import argparse
from msedge.selenium_tools import Edge, EdgeOptions
import pandas as pd
import platform
import datetime
import pandas as pd
import multiprocessing as mp
from functools import partial
import sys
from glob import glob


class Tweet:
    """Wrapper class to contain scraped tweets data"""

    def __init__(self, username, timestamp, text, likes, retweets, replies, url):
        self.username = username
        self.timestamp = timestamp
        self.text = text
        self.likes = likes
        self.retweets = retweets
        self.url = url
        self.replies = replies


def get_data(card):
    """Extract data from tweet card"""
    try:
        handle = card.find_element_by_xpath(
            './/span[contains(text(), "@")]').text
    except:
        return

    try:
        postdate = card.find_element_by_xpath(
            './/time').get_attribute('datetime')
    except:
        return

    try:
        comment = card.find_element_by_xpath('.//div[2]/div[2]/div[1]').text
    except:
        comment = ""

    try:
        responding = card.find_element_by_xpath('.//div[2]/div[2]/div[2]').text
    except:
        responding = ""

    text = comment + responding

    try:
        reply_cnt = card.find_element_by_xpath(
            './/div[@data-testid="reply"]').text
    except:
        reply_cnt = 0

    try:
        retweet_cnt = card.find_element_by_xpath(
            './/div[@data-testid="retweet"]').text
    except:
        retweet_cnt = 0

    try:
        like_cnt = card.find_element_by_xpath(
            './/div[@data-testid="like"]').text
    except:
        like_cnt = 0

        # handle promoted tweets
    try:
        promoted = card.find_element_by_xpath(
            './/div[2]/div[2]/[last()]//span').text == "Promoted"
    except:
        promoted = False
    if promoted:
        return
    # tweet url
    try:
        element = card.find_element_by_xpath(
            './/a[contains(@href, "/status/")]')
        tweet_url = element.get_attribute('href')
    except:
        return
    like_cnt = 0 if not like_cnt else like_cnt
    retweet_cnt = 0 if not retweet_cnt else retweet_cnt
    reply_cnt = 0 if not reply_cnt else reply_cnt

    return Tweet(
        handle, postdate, text, like_cnt, retweet_cnt, reply_cnt, tweet_url
    )


def log_search_page(driver, start_date, end_date, lang, display_type, words):
    """Builds and searches for a query between start_date and end_date"""

    if words != None:
        words = str(words).split("//")
        words = "%20".join(words) + "%20"
    else:
        words = ""

    if lang != None:
        lang = 'lang%3A'+lang
    else:
        lang = ""

    end_date = "until%3A"+end_date+"%20"
    start_date = "since%3A"+start_date+"%20"

    query = f"https://twitter.com/search?q={words}"\
        f"{end_date}{start_date}{lang}&src=typed_query"
    driver.get(query)

    sleep(1)

    # navigate to historical 'Top' or 'Latest' tab
    try:
        driver.find_element_by_link_text(display_type).click()
    except:
        print("Latest Button doesnt exist.")


def init_driver(navig, headless, proxy):
    # create instance of web driver
    # path to the chromdrive.exe
    if navig == "chrome":
        browser_path = ''
        if platform.system() == 'Windows':
            print('Detected OS : Windows')
            browser_path = './drivers/chromedriver_win.exe'
        elif platform.system() == 'Linux':
            print('Detected OS : Linux')
            browser_path = './drivers/chromedriver_linux'
        elif platform.system() == 'Darwin':
            print('Detected OS : Mac')
            browser_path = './drivers/chromedriver_mac'
        else:
            raise OSError('Unknown OS Type')
        options = Options()
        if headless == True:
            options.headless = True
        else:
            options.headless = False
        options.add_argument('--disable-gpu')
        options.add_argument('log-level=3')
        if proxy != None:
            options.add_argument('--proxy-server=%s' % proxy)
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(
            options=options, executable_path=browser_path)
        driver.set_page_load_timeout(100)
        return driver
    elif navig == "edge":
        browser_path = 'drivers/msedgedriver.exe'
        options = EdgeOptions()
        if proxy != None:
            options.add_argument('--proxy-server=%s' % proxy)
        if headless == True:
            options.headless = True
            options.use_chromium = False
        else:
            options.headless = False
            options.use_chromium = True
        options.add_argument('log-level=3')
        driver = Edge(options=options, executable_path=browser_path)
        return driver


def keep_scroling(driver):
    """Keeps scrolling the Twitter feed and records tweets"""
    data = []
    scrolling = True
    last_position = driver.execute_script("return window.pageYOffset;")
    while scrolling:
        # get the card of tweets
        page_cards = driver.find_elements_by_xpath(
            '//div[@data-testid="tweet"]'
        )
        for card in page_cards:
            tweet = get_data(card)
            if tweet:
                data.append(tweet)
                last_date = tweet.timestamp
                print("Tweet made at: " + str(last_date)+" is found.")
        scroll_attempt = 0
        while True:
            driver.execute_script(
                'window.scrollTo(0, document.body.scrollHeight);')
            sleep(1)
            curr_position = driver.execute_script("return window.pageYOffset;")
            if last_position == curr_position:
                scroll_attempt += 1
                # end of scroll region
                if scroll_attempt > 1:
                    scrolling = False
                    break
                else:
                    sleep(1)  # attempt another scroll
            else:
                last_position = curr_position
                break
    return data


def make_output_path(words, init_date, max_date):
    """Builds an output path to save the tweets for given dates"""
    first_term = words.split("//")[0]
    first_date = str(init_date).split(" ")[0]
    first_end_date = str(max_date).split(" ")[0]
    return f"tweeter_output/{first_term}_{first_date}_{first_end_date}.csv"


def scrap(
    data_tuple,
    words,
    days_between,
    lang,
    display_type,
    driver,
    headless,
    proxy
):
    """
    scrap data from twitter using requests, starting from start_date until max_date. The bot make a search between each start_date and end_date
    (days_between) until it reaches the max_date.

    return:
    data : df containing all tweets scraped with the associated features.
    save a csv file containing all tweets scraped with the associated features.
    """
    start_date, end_date = data_tuple
    # initiate the driver
    driver = init_driver(driver, headless, proxy)

    # log search page between start_date and end_date
    log_search_page(driver=driver, words=words, start_date=datetime.datetime.strftime(start_date, '%Y-%m-%d'), end_date=datetime.datetime.strftime(
        end_date, '%Y-%m-%d'), lang=lang, display_type=display_type
    )

    print(f"looking for tweets between {start_date} and {end_date}...")
    count_failed = 0
    data = keep_scroling(driver)
    # retrying in case of timeouts of disconnections (up to 3 times)
    while len(data) == 0 and count_failed < 2:
        driver.close()
        driver = init_driver(driver, headless, proxy)
        log_search_page(driver=driver, words=words, start_date=datetime.datetime.strftime(start_date, '%Y-%m-%d'), end_date=datetime.datetime.strftime(
            end_date, '%Y-%m-%d'), lang=lang, display_type=display_type
        )
        data = keep_scroling(driver)
        count_failed += 1
    driver.close()
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrap tweets.')

    parser.add_argument('--words', type=str,
                        help='Queries. they should be devided by "//" : Cat//Dog.', default=None)
    parser.add_argument('--max_date', type=str,
                        help='Max date for search query. example : %%Y-%%m-%%d.', required=True)
    parser.add_argument('--start_date', type=str,
                        help='Start date for search query. example : %%Y-%%m-%%d.', required=True)
    parser.add_argument('--interval', type=int,
                        help='Interval days between each start date and end date for search queries. example : 5.', default=1)
    parser.add_argument('--navig', type=str,
                        help='Navigator to use : chrome or edge.', default="chrome")
    parser.add_argument('--lang', type=str,
                        help='Tweets language. example : "en" for english and "fr" for french.', default="en")
    parser.add_argument('--headless', type=bool,
                        help='Headless webdrives or not. True or False', default=True)
    parser.add_argument('--display_type', type=str,
                        help='Display type of twitter page : Latest or Top', default="Top")
    parser.add_argument('--proxy', type=str,
                        help='Proxy server', default=None)
    parser.add_argument('--custom_column_name', type=str,
                        help='A column to include for the words query', default="airline")

    args = parser.parse_args()

    words = args.words
    max_date = args.max_date
    start_date = args.start_date
    interval = args.interval
    navig = args.navig
    lang = args.lang
    driver = args.navig
    headless = args.headless
    display_type = args.display_type
    proxy = args.proxy
    custom_column_name = args.custom_column_name

    # monitor time
    start = datetime.datetime.now()
    # creating list of requirements
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(max_date, '%Y-%m-%d')
    start_dates = [start_date]
    while start_dates[-1] + datetime.timedelta(days=interval) < end_date:
        start_dates.append(start_dates[-1] +
                           datetime.timedelta(days=interval))
    end_dates = [start_dates[i]
                 for i in range(1, len(start_dates))] + [end_date]
    # launching one worker thread per interval of time to scrape
    pool = mp.Pool(mp.cpu_count())
    try:
        all_tweets = pool.map(
            partial(
                scrap,
                words=words,
                days_between=interval,
                lang=lang,
                display_type=display_type,
                driver=driver,
                headless=headless,
                proxy=proxy,
            ),
            zip(start_dates, end_dates)
        )
        # merge and save
        merged_tweets = [
            t for interval_tweets in all_tweets for t in interval_tweets
        ]
        # save file
        df = pd.DataFrame([
            {**vars(tweet), **{custom_column_name: "_".join(words.split("//"))}}
            for tweet in merged_tweets
        ])
        df.to_csv(make_output_path(words, start_date, end_date), index=False)
        end = datetime.datetime.now()
        print(f"Time for current execution: {end - start}")
        print(f"Found {len(df.index)} tweets.")
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.exit()
