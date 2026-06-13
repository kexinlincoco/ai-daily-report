import os
import re
import json
import datetime
import requests
import feedparser
from openai import OpenAI

RSS_FEEDS = {
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "OpenAI Blog": "https://openai.com/blog/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "MIT Technology Review": "https://www.technologyreview.com/feed/",
    "Ars Technica Tech": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "NVIDIA Blog": "https://blogs.nvidia.com/feed/",
}

USER_AGENT = "AI-Daily-Report/1.0"


def fetch_recent_articles(hours=24):
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
    articles = []

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
            for entry in feed.entries[:15]:
                published = None
                for attr in ("published_parsed", "updated_parsed"):
                    parsed = getattr(entry, attr, None)
                    if parsed:
                        try:
                            published = datetime.datetime(*parsed[:6], tzinfo=datetime.timezone.utc)
                        except Exception:
                            pass
                        break

                if published and published < cutoff:
                    continue

                summary = entry.get("summary", "") or entry.get("description", "")
                summary = re.sub(r"<[^>]+>", "", summary)[:600].strip()

                articles.append({
                    "source": source,
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", ""),
                    "summary": summary,
                    "published": published.strftime("%Y-%m-%d") if published else "未知",
                })
        except Exception as e:
            print(f"[WARNING] Failed to fetch {source}: {e}")

    print(f"Fetched {len(articles)} articles from {len(RSS_FEEDS)} sources")
    return articles


def build_prompt(articles, today_str):
    articles_text = ""
    for i, a in enumerate(articles, 1):
        articles_text += (
            f"\n{i}. [{a['source']}] {a['title']}\n"
            f"   链接: {a['link']}\n"
            f"   摘要: {a['summary']}\n"
        )

    return f"""你是一位专业的 AI 行业分析师，为 AI 产品经理撰写每日 AI 资讯日报。

今日（{today_str}）收集到的 AI 相关资讯如下：
{articles_text}

请根据以上资讯，用中文撰写一份简洁的日报，格式严格如下：

【AI日报｜{today_str}】

一、今日最值得关注的 5 条
（选取最重要的 5 条，每条格式：
标题：xxx
来源：xxx
链接：xxx
发生了什么：xxx（1句话）
为什么重要：xxx（1句话）
）

二、模型与大厂动态
（OpenAI / Anthropic / Google / Meta / Microsoft 等，无则省略）

三、开源与开发者生态
（GitHub / Hugging Face / 框架工具等，无则省略）

四、AI 产品经理视角
（2-3 条产品趋势或 AI PM 启发，结合今日资讯）

五、今日关键词
（5 个关键词，用「·」分隔）

要求：全程中文，适合手机阅读，保留原始链接，不编造未出现在资讯中的内容。"""


def summarize_with_llm(articles):
    today_str = datetime.date.today().strftime("%Y年%m月%d日")

    if not articles:
        return f"【AI日报｜{today_str}】\n\n今日暂无最新 AI 资讯，请明天再见。"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(articles, today_str)}],
        max_tokens=2500,
        temperature=0.3,
    )
    return response.choices[0].message.content


def send_to_feishu(message):
    webhook_url = os.environ["FEISHU_WEBHOOK_URL"]
    payload = {
        "msg_type": "text",
        "content": {"text": message},
    }
    resp = requests.post(
        webhook_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"Feishu API returned error: {result}")
    print("Successfully sent to Feishu!")


def main():
    print(f"=== AI Daily Report | {datetime.date.today()} ===")

    articles = fetch_recent_articles(hours=24)

    print("Generating daily report with LLM...")
    report = summarize_with_llm(articles)

    preview = report[:300] + "..." if len(report) > 300 else report
    print(f"\n--- Report Preview ---\n{preview}\n")

    print("Sending to Feishu...")
    send_to_feishu(report)
    print("Done!")


if __name__ == "__main__":
    main()
