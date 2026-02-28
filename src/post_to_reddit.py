import os
from datetime import datetime

import pyotp
from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright


def _get_auth_params():
    load_dotenv()
    missing = [v for v in ("REDDIT_OTP_SECRET", "REDDIT_LOGIN", "REDDIT_PASSWORD") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    reddit_otp_secret = os.getenv("REDDIT_OTP_SECRET")
    totp = pyotp.TOTP(reddit_otp_secret)
    code = totp.now()
    return {
        "code": code,
        "login": os.getenv("REDDIT_LOGIN"),
        "password": os.getenv("REDDIT_PASSWORD"),
    }


def run(playwright: Playwright) -> None:
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
    page.get_by_role("button", name="Plus d'options").click()
    page.locator(
        "rpl-menu-item:nth-child(2) > #item > rpl-item > .text-container"
    ).first.click()
    page.get_by_role("textbox", name="Champ de texte du corps de la").fill(
        "Bel bonswa le sub! "
    )
    page.get_by_role("button", name="Enregistrer le brouillon").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
