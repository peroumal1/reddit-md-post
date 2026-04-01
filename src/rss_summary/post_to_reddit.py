import os
from datetime import datetime
from pathlib import Path

import click
import pyotp
import requests
from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright


def _get_auth_params():
    load_dotenv()
    missing = [
        v
        for v in ("REDDIT_OTP_SECRET", "REDDIT_LOGIN", "REDDIT_PASSWORD")
        if not os.getenv(v)
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    reddit_otp_secret = os.getenv("REDDIT_OTP_SECRET")
    totp = pyotp.TOTP(reddit_otp_secret)
    code = totp.now()
    return {
        "code": code,
        "login": os.getenv("REDDIT_LOGIN"),
        "password": os.getenv("REDDIT_PASSWORD"),
    }


def _fetch_body(feed_file, feed_url):
    if feed_url:
        r = requests.get(feed_url, timeout=15)
        if not r.ok:
            raise click.ClickException(f"Failed to fetch feed URL (HTTP {r.status_code}): {feed_url}")
        return r.text
    try:
        return Path(feed_file).read_text()
    except FileNotFoundError:
        raise click.ClickException(f"Feed file not found: {feed_file}")


def run(playwright: Playwright, body: str) -> None:
    params = _get_auth_params()
    today = datetime.now().strftime("%d/%m/%Y")
    browser = playwright.firefox.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    page.goto("https://www.reddit.com/")
    page.get_by_role("button", name="Accept All").click()
    page.get_by_role("link", name="Log In").click()
    page.get_by_role("textbox", name="Email or username").click()
    page.get_by_role("textbox", name="Email or username").fill(params["login"])
    page.get_by_role("textbox", name="Email or username").press("Tab")
    page.get_by_role("textbox", name="Password").fill(params["password"])
    page.get_by_role("button", name="Log In").click()
    page.locator(
        "#one-time-code-appOtp > label > .label-container > .input-boundary-box"
    ).click()
    page.get_by_role("textbox", name="Verification code").fill(params["code"])
    page.get_by_role("button", name="Check code").click()

    page.locator("#moderation_section").get_by_role("link", name="r/Guadeloupe").click()
    page.get_by_test_id("create-post").click()

    page.get_by_role("textbox", name="Titre").click()
    page.get_by_role("textbox", name="Titre").fill("Les infos quotidiennes - " + today)

    page.get_by_role("button", name="Ajouter un flair et des é").click()
    page.get_by_role("radio", name="News").click()
    page.get_by_role("button", name="Ajouter", exact=True).click()

    page.get_by_role("paragraph").click()
    page.get_by_role("button", name="Plus d\u2019options").click()
    page.get_by_role("menuitem", name="Passer à Markdown").click()

    # Type body content — press_sequentially() is required to trigger Lexical's
    # internal EditorState updates; fill() does not fire the beforeinput events
    # that Lexical listens on.
    body_field = page.get_by_role("textbox", name="Champ de texte du corps de la")
    body_field.click()
    body_field.press_sequentially(body, timeout=0)

    # Save as draft — change to "Publier" when ready to submit for real
    page.get_by_role("button", name="Enregistrer le brouillon").click()

    context.close()
    browser.close()


@click.command()
@click.option(
    "--feed-file",
    default="data/feed.md",
    show_default=True,
    help="Local path to the markdown feed file.",
)
@click.option(
    "--feed-url",
    default=None,
    help="URL to fetch the markdown feed from (e.g. raw GitHub URL). Takes precedence over --feed-file.",
)
def main(feed_file, feed_url):
    """Post today's feed digest to r/Guadeloupe via Playwright."""
    if feed_file != "data/feed.md" and feed_url:
        raise click.UsageError("--feed-file and --feed-url are mutually exclusive.")
    body = _fetch_body(feed_file, feed_url)
    with sync_playwright() as playwright:
        run(playwright, body)


if __name__ == "__main__":
    main()
