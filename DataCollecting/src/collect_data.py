import asyncio
from statistics import median, stdev

import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import os
import logging
import re

# Define the directory for saving results
OUTPUT_DIR = 'results'

# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'script.log')
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Read the GitHub token from a file
TOKEN_FILE_PATH = '../private/token.txt'
if not os.path.exists(TOKEN_FILE_PATH):
    logging.critical(f"GitHub token file not found at {TOKEN_FILE_PATH}")
    raise FileNotFoundError(f"GitHub token file not found at {TOKEN_FILE_PATH}")

with open(TOKEN_FILE_PATH, 'r') as f:
    TOKEN = f.read().strip()

# Base URL for GitHub API
GITHUB_API_URL = "https://api.github.com"

# Headers for GitHub API requests
HEADERS = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
}

# Maximum number of concurrent HTTP requests
MAX_CONCURRENT_REQUESTS = 10

# Maximum number of search pages to prevent excessive API usage
MAX_SEARCH_PAGES = 50  # Can be adjusted as needed

# Log file for failed repositories
FAILED_REPOS_LOG = os.path.join(OUTPUT_DIR, 'failed_repositories.log')

# Clear the previous log file if it exists
if os.path.exists(FAILED_REPOS_LOG):
    os.remove(FAILED_REPOS_LOG)
    logging.info(f"Cleared previous log file: {FAILED_REPOS_LOG}")

# Initialize data storage
developer_metrics = []

def normalize_email(email):
    """
    Normalizes GitHub noreply emails by removing numerical prefixes.

    Args:
        email (str): The original email address.

    Returns:
        str: The normalized email address.
    """
    if email:
        # Match emails with format 'number+username@users.noreply.github.com'
        match = re.match(r'\d+\+(.+)', email)
        if match:
            return match.group(1)
    return email

async def fetch(session, url, params=None):
    """
    Asynchronously fetches JSON data from the specified URL.
    Handles rate limiting by checking response headers.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        url (str): URL to fetch data from.
        params (dict, optional): Query parameters.

    Returns:
        dict or list: JSON response.
    """
    try:
        async with session.get(url, headers=HEADERS, params=params) as response:
            if response.status == 200:
                data = await response.json()
                logging.debug(f"Successfully fetched data from {url}")
                return data
            elif response.status == 204:
                logging.debug(f"No content for {url}")
                return []  # No Content
            elif response.status == 403 and 'rate limit' in (await response.text()).lower():
                reset_timestamp = response.headers.get('X-RateLimit-Reset')
                if reset_timestamp:
                    reset_time = datetime.fromtimestamp(int(reset_timestamp))
                    sleep_seconds = (reset_time - datetime.now()).total_seconds() + 5  # Add buffer
                    logging.warning(f"Rate limit exceeded. Sleeping for {sleep_seconds:.0f} seconds until {reset_time}.")
                    await asyncio.sleep(sleep_seconds)
                    return await fetch(session, url, params)
                else:
                    logging.error(f"Rate limit exceeded and no reset time provided for {url}.")
                    raise Exception(f"Rate limit exceeded and no reset time provided for {url}.")
            else:
                text = await response.text()
                logging.error(f"Failed to fetch {url}: Status {response.status} - {text}")
                raise Exception(f"Failed to fetch {url}: {response.status} - {text}")
    except Exception as e:
        logging.error(f"Exception while fetching {url}: {e}")
        raise

async def fetch_contributors(session, repo_full_name):
    """
    Asynchronously fetches contributors for the specified repository.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repo_full_name (str): Full name of the repository (e.g., 'owner/repo').

    Returns:
        list: List of contributors or None if failed.
    """
    contributors_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/contributors"
    logging.info(f"Fetching contributors for repository: {repo_full_name}")
    try:
        contributors = await fetch(session, contributors_url, params={'per_page': 5})
        logging.info(f"Fetched {len(contributors)} contributors for {repo_full_name}")
        return contributors
    except Exception as e:
        logging.error(f"Failed to fetch contributors for repository {repo_full_name}: {e}")
        # Log failed repository
        with open(FAILED_REPOS_LOG, 'a') as f:
            f.write(f"{repo_full_name}: {e}\n")
        return None

async def search_repositories(session, page, per_page):
    """
    Asynchronously searches for repositories based on defined criteria.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        page (int): Page number for pagination.
        per_page (int): Number of repositories per page.

    Returns:
        list: List of repositories.
    """
    query = 'stars:10..50 created:>=' + (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    params = {
        'q': query,
        'sort': 'stars',
        'order': 'asc',
        'per_page': per_page,
        'page': page,
    }
    logging.info(f"Searching repositories: page {page}, {per_page} per page")
    try:
        data = await fetch(session, f"{GITHUB_API_URL}/search/repositories", params=params)
        items = data.get('items', [])
        logging.info(f"Found {len(items)} repositories on page {page}")
        return items
    except Exception as e:
        logging.error(f"Failed to search repositories on page {page}: {e}")
        return []

async def select_repositories(session, target_count=100):
    """
    Asynchronously selects the specified number of small GitHub repositories.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        target_count (int, optional): Number of repositories to select. Default is 100.

    Returns:
        list: List of full repository names (e.g., 'owner/repo').
    """
    repositories = []
    page = 1
    per_page = 50  # Number of repositories per search page
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    while len(repositories) < target_count and page <= MAX_SEARCH_PAGES:
        logging.info(f"Processing search page {page}")
        items = await search_repositories(session, page, per_page)
        if not items:
            logging.info("No more repositories found.")
            break  # No more repositories found

        tasks = []
        for repo in items:
            if len(repositories) >= target_count:
                break
            repo_full_name = repo['full_name']
            # Use a shared semaphore
            task = asyncio.create_task(process_repository(session, repo_full_name, repositories, semaphore))
            tasks.append(task)

        # Wait for all tasks on the current page to complete
        await asyncio.gather(*tasks)
        logging.info(f"Page {page} processed. Collected {len(repositories)} repositories.")
        print(f"Processed page {page}. Total repositories collected: {len(repositories)}")
        page += 1
        await asyncio.sleep(1)  # Sleep to respect rate limits

    logging.info(f"Selected {len(repositories)} repositories.")
    return repositories

async def process_repository(session, repo_full_name, repositories, semaphore):
    """
    Processes a single repository: fetches its contributors and adds it to the list if it meets criteria.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repo_full_name (str): Full name of the repository.
        repositories (list): List to add valid repositories.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.
    """
    async with semaphore:
        logging.info(f"Starting processing repository: {repo_full_name}")
        contributors = await fetch_contributors(session, repo_full_name)
        if contributors is not None and len(contributors) <= 5:
            repositories.append(repo_full_name)
            logging.info(f"Added repository: {repo_full_name}")
        else:
            logging.info(f"Rejected repository: {repo_full_name} (contributors: {len(contributors) if contributors else 'N/A'})")

async def collect_commits(session, repo_full_name):
    """
    Asynchronously collects commit data for the specified repository.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repo_full_name (str): Full name of the repository (e.g., 'owner/repo').

    Returns:
        list: List of commit data.
    """
    commits = []
    page = 1
    per_page = 100  # Maximum per page

    logging.info(f"Starting to collect commits for repository: {repo_full_name}")
    while True:
        params = {
            'per_page': per_page,
            'page': page,
        }
        commits_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/commits"
        try:
            commit_data = await fetch(session, commits_url, params=params)
            if not commit_data:
                logging.info(f"Finished collecting commits for {repo_full_name} on page {page}.")
                break
            commits.extend(commit_data)
            logging.info(f"Fetched {len(commit_data)} commits on page {page} for {repo_full_name}")
            if len(commit_data) < per_page:
                logging.info(f"Finished collecting commits for {repo_full_name}. Total commits: {len(commits)}")
                break
            page += 1
            await asyncio.sleep(0.5)  # Sleep to respect rate limits
        except Exception as e:
            logging.error(f"Failed to fetch commits for repository {repo_full_name} on page {page}: {e}")
            # Log failed repository
            with open(FAILED_REPOS_LOG, 'a') as f:
                f.write(f"{repo_full_name} (commits page {page}): {e}\n")
            break

    logging.info(f"Collected {len(commits)} commits for repository {repo_full_name}")
    return commits

async def fetch_commit_details(session, repo_full_name, commit_sha):
    """
    Asynchronously fetches detailed information about a single commit, including 'stats'.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repo_full_name (str): Full name of the repository.
        commit_sha (str): SHA of the commit.

    Returns:
        dict: Detailed commit data or None if failed.
    """
    commit_url = f"{GITHUB_API_URL}/repos/{repo_full_name}/commits/{commit_sha}"
    logging.debug(f"Fetching detailed information for commit {commit_sha} in {repo_full_name}")
    try:
        commit_detail = await fetch(session, commit_url)
        logging.debug(f"Fetched details for commit {commit_sha} in {repo_full_name}")
        return commit_detail
    except Exception as e:
        logging.error(f"Failed to fetch details for commit {commit_sha} in repository {repo_full_name}: {e}")
        return None

def compute_developer_metrics(commits, detailed_commits):
    """
    Computes metrics for each developer based on commit data.

    This function has been enhanced to:
      1) Calculate both median and standard deviation for the days between commits,
         providing deeper insights into the consistency of a developer's commit behavior.
      2) Provide clearer in-line comments on the mathematics involved.

    Args:
        commits (list): List of commit data (as returned by the GitHub API).
        detailed_commits (dict): Mapping from commit SHA to detailed commit data (includes 'stats').

    Returns:
        None: The results are appended to the global `developer_metrics` list.
    """
    logging.info("Starting to compute metrics for developers")
    developers = defaultdict(list)  # Mapping from developer email to list of their commits

    # STEP 1: Group commits by developer's email
    for commit in commits:
        # Some commits may not have author information (e.g., if the email is private)
        author = commit.get('commit', {}).get('author', {})
        if not author:
            logging.debug(f"Commit {commit.get('sha')} has no author information.")
            continue
        author_email = normalize_email(author.get('email'))
        if not author_email:
            logging.debug(f"Commit {commit.get('sha')} has no author email.")
            continue
        # Use email as a unique identifier for the developer
        developers[author_email].append(commit)

    # STEP 2: Calculate metrics per developer
    for email, dev_commits in developers.items():
        logging.debug(f"Processing metrics for developer: {email} with {len(dev_commits)} commits")
        metrics = {'email': email}
        commit_dates = []
        commit_times = []
        loc_modified_list = []
        time_diffs = []

        # Collect commit times and LOC changes
        for commit in dev_commits:
            commit_info = commit.get('commit', {})
            author_info = commit_info.get('author', {})
            date_str = author_info.get('date')
            if not date_str:
                logging.debug(f"Commit {commit.get('sha')} has no date.")
                continue

            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                logging.warning(f"Invalid date format '{date_str}' for commit {commit.get('sha')}")
                continue

            commit_dates.append(date_obj.date())
            commit_times.append(date_obj)

            # Collect LOC changes (additions + deletions)
            commit_sha = commit.get('sha')
            detailed_commit = detailed_commits.get(commit_sha)
            if detailed_commit:
                stats = detailed_commit.get('stats', {})
                if stats:
                    total_changes = stats.get('total', 0)
                else:
                    total_changes = 0
            else:
                total_changes = 0

            loc_modified_list.append(total_changes)

        if not commit_times:
            logging.debug(f"Developer {email} has no commits with valid dates.")
            continue

        # Sort commit times in ascending order
        commit_times.sort()

        # STEP 2A: Calculate time differences (in days) between consecutive commits.
        # We skip consecutive commits on the same day (diff == 0) to avoid skewing the median.
        for i in range(1, len(commit_times)):
            diff = (commit_times[i].date() - commit_times[i - 1].date()).days
            if diff > 0:
                time_diffs.append(diff)

        # 'period' metric: Number of days between the first and the last commit
        period_days = (commit_times[-1] - commit_times[0]).days
        metrics['period'] = period_days

        # 'days' metric: Number of distinct calendar days with at least one commit
        metrics['days'] = len(set(commit_dates))

        # 'weeks' metric: Number of distinct calendar weeks with at least one commit
        weeks = set()
        for d in commit_dates:
            year, week_num, _ = d.isocalendar()
            weeks.add((year, week_num))
        metrics['weeks'] = len(weeks)

        # 'timediff' metric: Median of the day differences between consecutive commits
        if time_diffs:
            median_timediff = median(time_diffs)
        else:
            median_timediff = 0
        metrics['timediff'] = median_timediff

        # NEW: 'timediff_std' metric (optional): Standard deviation of day differences
        # This measures how spread out the intervals between commits are.
        # A higher value indicates more erratic commit times.
        if len(time_diffs) > 1:
            # stdev requires at least two data points
            metrics['timediff_std'] = round(stdev(time_diffs), 2)
        else:
            metrics['timediff_std'] = 0

        # 'commits' metric: Number of commits authored by this developer
        metrics['commits'] = len(dev_commits)

        # 'loc_per_commit' metric: Median of total LOC changes per commit
        if loc_modified_list and all(isinstance(x, (int, float)) for x in loc_modified_list):
            median_loc = median(loc_modified_list)
        else:
            median_loc = 0
        metrics['loc_per_commit'] = median_loc

        # 'weekend' metric: Percentage of commits on Saturdays (5) or Sundays (6)
        weekend_commits = sum(1 for dt in commit_times if dt.weekday() >= 5)
        total_commits = len(commit_times)
        metrics['weekend'] = (weekend_commits / total_commits * 100) if total_commits else 0

        # Time-of-day metrics
        hour_counts = defaultdict(int)
        night_commits = 0
        morning_commits = 0
        afternoon_commits = 0
        evening_commits = 0
        office_commits = 0

        for dt in commit_times:
            hour = dt.hour
            hour_counts[hour] += 1
            if 0 <= hour < 6:
                night_commits += 1
            if 6 <= hour < 12:
                morning_commits += 1
            if 12 <= hour < 18:
                afternoon_commits += 1
            if 18 <= hour < 24:
                evening_commits += 1
            if 8 <= hour < 17:
                office_commits += 1

        if total_commits:
            metrics['night'] = f"{(night_commits / total_commits * 100):.2f}%"
            metrics['morning'] = f"{(morning_commits / total_commits * 100):.2f}%"
            metrics['afternoon'] = f"{(afternoon_commits / total_commits * 100):.2f}%"
            metrics['evening'] = f"{(evening_commits / total_commits * 100):.2f}%"
            metrics['office'] = f"{(office_commits / total_commits * 100):.2f}%"
        else:
            metrics['night'] = "0.00%"
            metrics['morning'] = "0.00%"
            metrics['afternoon'] = "0.00%"
            metrics['evening'] = "0.00%"
            metrics['office'] = "0.00%"

        # 'most_active_hour' metric: Hour of day with the highest commit frequency
        if hour_counts:
            most_active_hour = max(hour_counts, key=hour_counts.get)
            metrics['most_active_hour'] = f"{most_active_hour:02d}:00"
        else:
            metrics['most_active_hour'] = "N/A"

        # Working-hours metrics (weekday commits only)
        weekday_times = [dt for dt in commit_times if dt.weekday() < 5]
        if weekday_times:
            total_weekday_commits = len(weekday_times)
            # Define regular working hours
            regular_start = 8  # 8 AM
            regular_end = 17   # 5 PM
            regular_commits = sum(1 for dt in weekday_times if regular_start <= dt.hour < regular_end)

            # Percentage of commits during regular office hours (Mon-Fri, 8am-5pm)
            regular_percentage = (regular_commits / total_weekday_commits * 100) if total_weekday_commits else 0
            metrics['office_percentage'] = f"{regular_percentage:.2f}%"

            # Determine 'beginning_regular', 'end_regular', and 'length_regular'
            hours = [dt.hour + dt.minute / 60 for dt in weekday_times]
            start_hour = min(hours)
            end_hour = max(hours)
            length_regular = end_hour - start_hour

            # Convert to "HH:MM" format
            beginning_regular = f"{int(start_hour):02d}:{int((start_hour - int(start_hour)) * 60):02d}"
            end_regular = f"{int(end_hour):02d}:{int((end_hour - int(end_hour)) * 60):02d}"
            length_regular_hours = f"{int(length_regular)}h {int((length_regular - int(length_regular)) * 60)}m"

            metrics['beginning_regular'] = beginning_regular
            metrics['end_regular'] = end_regular
            metrics['length_regular'] = length_regular_hours
        else:
            metrics['office_percentage'] = "0.00%"
            metrics['beginning_regular'] = "N/A"
            metrics['end_regular'] = "N/A"
            metrics['length_regular'] = "N/A"

        developer_metrics.append(metrics)
        logging.info(f"Metrics for {email} added to developer_metrics list")

    logging.info("Completed computing metrics for developers")


async def fetch_commit_details_for_repo(session, repo_full_name, commits, semaphore):
    """
    Asynchronously fetches detailed commit data for a repository.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repo_full_name (str): Full name of the repository.
        commits (list): List of commit data.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.

    Returns:
        dict: Mapping from commit SHA to detailed commit data.
    """
    detailed_commits = {}
    tasks = []
    for commit in commits:
        commit_sha = commit.get('sha')
        if commit_sha:
            task = asyncio.create_task(fetch_commit_details_with_semaphore(session, repo_full_name, commit_sha, semaphore))
            tasks.append(task)

    detailed_results = await asyncio.gather(*tasks)
    for result in detailed_results:
        if result:
            detailed_commits[result['sha']] = result

    logging.info(f"Fetched detailed data for {len(detailed_commits)} commits in repository {repo_full_name}")
    return detailed_commits

async def fetch_commit_details_with_semaphore(session, repo_full_name, commit_sha, semaphore):
    """
    Wrapper to fetch commit details using a semaphore.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repo_full_name (str): Full name of the repository.
        commit_sha (str): SHA of the commit.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.

    Returns:
        dict: Detailed commit data or None.
    """
    async with semaphore:
        return await fetch_commit_details(session, repo_full_name, commit_sha)

async def collect_all_data(session, repositories):
    """
    Asynchronously collects all necessary data for each repository.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repositories (list): List of full repository names.

    Returns:
        None
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    for repo_full_name in repositories:
        task = asyncio.create_task(process_repository_data(session, repo_full_name, semaphore))
        tasks.append(task)

    await asyncio.gather(*tasks)
    logging.info("Completed data collection from all repositories")

async def process_repository_data(session, repo_full_name, semaphore):
    """
    Processes all data collection for a single repository.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        repo_full_name (str): Full name of the repository.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.

    Returns:
        None
    """
    logging.info(f"Starting data collection for repository: {repo_full_name}")
    commits = await collect_commits(session, repo_full_name)
    if commits:
        # Fetch detailed commit data to get 'stats'
        detailed_commits = await fetch_commit_details_for_repo(session, repo_full_name, commits, semaphore)
        # Compute metrics based on commits and detailed data
        compute_developer_metrics(commits, detailed_commits)
        logging.info(f"Data collection and processing for {repo_full_name} completed")
    else:
        logging.warning(f"No commits found for repository {repo_full_name}")

def validate_data(data, name):
    """
    Validates that the data list is not empty.

    Args:
        data (list): Data list to validate.
        name (str): Name of the data type for logging.

    Returns:
        bool: True if data is valid, False otherwise.
    """
    if not data:
        logging.warning(f"No data collected for {name}.")
        print(f"No data collected for {name}.")
        return False
    logging.info(f"Data validation for {name} passed successfully.")
    return True

def clean_and_sanitize(df, columns_to_truncate=None, columns_to_sanitize=None, max_length=1000):
    """
    Cleans and sanitizes the DataFrame by truncating and removing problematic characters.

    Args:
        df (pd.DataFrame): DataFrame to clean.
        columns_to_truncate (list, optional): List of columns to truncate.
        columns_to_sanitize (list, optional): List of columns to sanitize.
        max_length (int): Maximum allowed length for strings.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logging.info("Starting DataFrame cleaning and sanitization")
    if columns_to_truncate:
        for column in columns_to_truncate:
            if column in df.columns:
                df[column] = df[column].astype(str).str.slice(0, max_length)
                logging.debug(f"Truncated column '{column}' to {max_length} characters")
    if columns_to_sanitize:
        for column in columns_to_sanitize:
            if column in df.columns:
                # Remove non-printable characters
                df[column] = df[column].astype(str).str.replace(r'[\x00-\x1F\x7F]', '', regex=True)
                logging.debug(f"Sanitized column '{column}' by removing non-printable characters")
    logging.info("DataFrame cleaning and sanitization completed")
    return df

def save_developer_metrics(df, filename='developer_metrics.parquet', output_dir='results'):
    """
    Saves the developer metrics DataFrame to a Parquet file in the specified directory.

    Args:
        df (pd.DataFrame): DataFrame with developer metrics.
        filename (str): Name of the output Parquet file.
        output_dir (str): Directory where the Parquet file will be saved.

    Returns:
        None
    """
    try:
        logging.info(f"Starting to save developer metrics to file '{filename}'")
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Full path to the Parquet file
        filepath = os.path.join(output_dir, filename)

        # Save to Parquet
        df.to_parquet(filepath, index=False, engine='pyarrow')
        logging.info(f"Developer metrics successfully saved to '{filepath}'.")
        print(f"Saved to '{filepath}'.")
    except Exception as e:
        logging.error(f"Failed to save '{filename}' in '{output_dir}': {e}")
        print(f"Error saving '{filename}' in '{output_dir}': {e}")

async def main_async():
    """
    Main asynchronous function to perform data collection and metric computation.
    """
    logging.info("Script execution started")

    # Step 1: Select 100 small GitHub repositories
    print("Selecting 100 small GitHub repositories...")
    logging.info("Starting repository selection")
    async with aiohttp.ClientSession() as session:
        repositories = await select_repositories(session, target_count=100)
    print(f"Selected {len(repositories)} repositories.")
    logging.info(f"Selected {len(repositories)} repositories")

    if len(repositories) < 100:
        print(f"Proceeding with the available repositories: {len(repositories)} instead of 100.")
        logging.warning(f"Selected fewer repositories than expected: {len(repositories)} instead of 100")

    # Continue only if repositories are found
    if repositories:
        # Step 2: Collect all necessary data for each repository
        print(f"Collecting data from {len(repositories)} repositories...")
        logging.info(f"Starting data collection from {len(repositories)} repositories")
        async with aiohttp.ClientSession() as session:
            await collect_all_data(session, repositories)

        # Step 3: Compute developer metrics has already been done during data collection

        # Convert developer_metrics to DataFrame
        if developer_metrics:
            logging.info("Converting developer metrics to DataFrame")
            df_metrics = pd.DataFrame(developer_metrics)

            # Clean and sanitize data as needed
            # For example: truncate 'email' if it's too long
            df_metrics = clean_and_sanitize(
                df_metrics,
                columns_to_truncate=['email'],
                columns_to_sanitize=['email']
            )

            # Step 4: Save developer metrics to a Parquet file
            save_developer_metrics(df_metrics, 'developer_metrics.parquet')
        else:
            logging.warning("developer_metrics is empty. No metrics to save.")
            print("No developer metrics to save.")
    else:
        logging.warning("No repositories to process.")
        print("No repositories to process.")

    logging.info("Script execution completed")

def main():
    """
    Entry point for the script.
    """
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.critical(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
