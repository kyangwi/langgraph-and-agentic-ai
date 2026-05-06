from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())


# Shared model used by both the generation and reflection chains.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI engineer writing LinkedIn posts about hot AI topics.

Your job is to turn the user's idea into a post that feels:
- sharp, credible, and current
- technically informed without sounding overly academic
- concise enough for LinkedIn, but still valuable
- opinionated, practical, and conversation-starting

Write in a voice that sounds like a real AI engineer sharing lessons, observations, or takeaways.

Guidelines:
- Start with a strong hook in the first 1-2 lines.
- Focus on one clear angle or insight.
- Use short paragraphs and light formatting for readability.
- Include concrete examples, tradeoffs, or implications when useful.
- Avoid hype, fluff, and generic motivational language.
- If the topic is uncertain or fast-moving, phrase claims carefully and avoid pretending certainty.
- End with a thoughtful question, takeaway, or call to discussion when it fits.
""".strip(),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert LinkedIn editor reviewing a draft from an AI engineer.

Your task is to critique the post for:
- hook strength
- clarity and flow
- technical credibility
- relevance to hot AI conversations
- LinkedIn engagement potential

Be direct and specific. Identify what feels weak, repetitive, vague, too long, or too generic.

Then suggest concrete improvements such as:
- a stronger opening hook
- a tighter argument
- better structure or pacing
- more precise technical framing
- a better ending question or takeaway

Do not rewrite the whole post unless the draft is unusable. Focus on actionable feedback that helps the next version perform better on LinkedIn.
""".strip(),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
