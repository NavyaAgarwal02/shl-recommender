import json
import time
import re
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.shl.com"

CATALOG_PAGES = [
    f"{BASE_URL}/solutions/products/product-catalog/?start={i}&type=1"
    for i in range(0, 380, 12)
]

def scrape_catalog():
    assessments = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        for page_url in CATALOG_PAGES:
            print(f"Scraping: {page_url}")
            try:
                page.goto(page_url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2000)

                # Get links AND surrounding text for description
                items = page.eval_on_selector_all(
                    "a[href*='/product-catalog/view/']",
                    """els => els.map(e => ({
                        href: e.href,
                        text: e.innerText.trim(),
                        context: e.closest('tr,li,div') ? e.closest('tr,li,div').innerText.trim() : ''
                    }))"""
                )

                new_found = 0
                seen_urls = {a["url"] for a in assessments}
                for item in items:
                    href = item["href"]
                    name = item["text"].strip()
                    context = item.get("context", "")
                    if href not in seen_urls and name and len(name) > 2:
                        seen_urls.add(href)
                        # Guess test type from context text
                        test_type = ""
                        ctx_lower = context.lower()
                        if any(w in ctx_lower for w in ["ability","cognitive","verbal","numerical","reasoning"]):
                            test_type = "A"
                        elif any(w in ctx_lower for w in ["personality","behaviour","opq"]):
                            test_type = "P"
                        elif any(w in ctx_lower for w in ["knowledge","technical","skill","coding"]):
                            test_type = "K"
                        elif any(w in ctx_lower for w in ["simulation","exercise","inbox"]):
                            test_type = "S"
                        elif any(w in ctx_lower for w in ["biodata","motivation"]):
                            test_type = "B"

                        assessments.append({
                            "name": name,
                            "url": href,
                            "test_type": test_type,
                            "description": context[:400],
                            "duration": "",
                        })
                        new_found += 1

                print(f"  Found {new_found} new (total: {len(assessments)})")
                if new_found == 0:
                    print("  Done — end of catalog")
                    break

            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(0.8)

        browser.close()

    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(assessments)} assessments to catalog.json")


if __name__ == "__main__":
    scrape_catalog()