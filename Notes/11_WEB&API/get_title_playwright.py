import asyncio
import sys
from playwright.async_api import async_playwright

async def get_title():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto('https://nplus1.ru/news/2021/10/11/econobel2021')
        await page.wait_for_load_state('networkidle')
        title = await page.title()
        await browser.close()
        return title

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        result = asyncio.run(get_title())
        print(result)   # обязательно выводим
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)