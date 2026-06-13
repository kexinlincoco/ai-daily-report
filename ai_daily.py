import os
import re
import json
import datetime
import requests
import feedparser
from openai import OpenAI

RSS_FEEDS = {
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "OpenAI Blog": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Anthropic News": "https://www.anthropic.com/news/rss.xml",
    "Microsoft AI Blog": "https://blogs.microsoft.com/ai/feed/",
    "Meta AI Blog": "https://ai.meta.com/blog/rss/",
    "NVIDIA Blog": "https://blogs.nvidia.com/feed/",
    "LangChain Blog": "https://blog.langchain.com/rss/",
    "LlamaIndex Blog": "https://www.llamaindex.ai/blog/rss.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "MIT Technology Review": "https://www.technologyreview.com/feed/",
}

USER_AGENT = "AI-Daily-Report/1.0"


def fetch_recent_articles(hours=36):
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
            f"   发布时间: {a['published']}\n"
            f"   链接: {a['link']}\n"
            f"   摘要: {a['summary']}\n"
        )

    return f"""你是一位专业的 AI 行业分析师，同时具备 AI 产品经理视角。你的任务不是简单翻译新闻，而是从“产品机会、技术趋势、公司战略、开发者生态、人员流动”角度，为我整理每日 AI 资讯日报。

今日（{today_str}）收集到的 AI 相关资讯如下：
{articles_text}

请根据以上资讯，用中文撰写一份适合发到飞书、适合手机阅读的 AI 日报。

重要要求：
1. 只基于我提供的资讯总结，不要编造没有出现的信息。
2. 如果某个板块没有足够信息，可以写“今日暂无值得关注更新”。
3. 不要把链接堆在正文中间；每条新闻只在最后保留一行“原文：链接”。
4. 不要写成普通新闻摘要，要突出“为什么重要”和“对 AI PM 的启发”。
5. 内容要有信息密度，但不要太长，适合手机阅读。
6. 优先关注：模型发布、AI Agent、RAG、多模态、AI Coding、开源模型、AI 产品化、商业化、人员变动、组织调整、融资并购。

请严格按照下面格式输出：

【AI日报｜{today_str}】

一、今日必看 Top 5

1. 标题：xxx
一句话结论：xxx
What：用 3-5 句话说明发生了什么。
Why：用 2-3 句话说明为什么重要。
AI PM 视角：说明这件事对产品设计、用户需求、商业化、工作流或行业趋势有什么启发。
原文：xxx

2. 标题：xxx
一句话结论：xxx
What：xxx
Why：xxx
AI PM 视角：xxx
原文：xxx

二、模型与大厂动态
- 总结 OpenAI / Anthropic / Google / Meta / Microsoft / NVIDIA / xAI 等公司的模型、产品、平台或战略更新。
- 每条控制在 2-3 句话。
- 无则写：今日暂无值得关注更新。

三、开源与开发者生态
- 总结 GitHub、Hugging Face、LangChain、LlamaIndex、开源模型、开发者工具等动态。
- 重点说明它们对开发者和 AI 应用构建的影响。
- 无则写：今日暂无值得关注更新。

四、AI 产品与商业化
- 总结 AI 应用、Agent、办公自动化、搜索、AI Coding、企业服务、内容生成等产品动态。
- 重点说明这些动态反映了什么产品趋势。
- 无则写：今日暂无值得关注更新。

五、人员变动与组织调整
- 关注 AI 公司高管、核心研究员、创始人、模型负责人、产品负责人、团队重组等变化。
- 说明该变动可能代表的公司战略或行业信号。
- 无则写：今日暂无值得关注人员变动。

六、融资与行业趋势
- 总结融资、并购、合作、监管、市场竞争等内容。
- 重点说明行业方向变化。
- 无则写：今日暂无值得关注更新。

七、AI 产品经理视角
请输出 2-3 条高价值启发，每条格式：
- 启发 1：xxx
- 启发 2：xxx
- 启发 3：xxx

八、今日关键词
用「·」分隔，输出 5-8 个关键词。
"""

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
