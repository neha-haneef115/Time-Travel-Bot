from __future__ import annotations

import os
import asyncio
from typing import List, Dict, cast
import textwrap

import requests
from dotenv import load_dotenv
import chainlit as cl

try:
    from openai import AsyncOpenAI
except ImportError as e: 
    raise SystemExit("`openai` package missing — run `pip install openai`. ") from e

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "gemini-2.0-flash")

if not (GEMINI_KEY or OPENAI_KEY):
    raise RuntimeError("Set GEMINI_API_KEY or OPENAI_API_KEY in .env.")


def get_async_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client wired to OpenAI **or** Google Gemini."""
    if OPENAI_KEY:
        return AsyncOpenAI(api_key=OPENAI_KEY)  
    return AsyncOpenAI(
        api_key=GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


async def fetch_bio(name: str) -> str:
    """Grab the first paragraph of a Wikipedia summary (non‑blocking)."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '%20')}"

    def _request() -> str:
        try:
            r = requests.get(url, timeout=6)
            r.raise_for_status()
            data: Dict = r.json()
            return data.get("extract", "")
        except Exception:
            return ""

    return await asyncio.to_thread(_request)


def build_system_prompt(name: str, bio: str) -> str:
    bio_clause = f"Here is a concise biography to ground you: {bio}" if bio else ""
    return textwrap.dedent(f"""
        You are {name}, speaking in first person. {bio_clause}
        Adopt {name}'s typical tone, vocabulary, and worldview. Stay in character unless explicitly asked to drop the act. If you are unsure of something, make a best‑guess based on {name}'s known life and era.
    """)



@cl.on_chat_start
async def on_chat_start():
    """Greet the user and set up session storage."""
    await cl.Message(
        """**Welcome to Time‑Travel Bot!**\n\n"""
        "Tell me the name of any historical or contemporary public figure, and I'll let you chat with them in their own voice.\n"
        "Example: *Albert Einstein*, *Cleopatra*, *Allama Iqbal*, *Elon Musk*.\n\n"
        "Type `switch <another name>` at any time to change persona."
    ).send()

    cl.user_session.set("persona", None)      
    cl.user_session.set("prompt", "")         
    cl.user_session.set("history", [])         
    cl.user_session.set("model", MODEL_NAME)
    cl.user_session.set("client", get_async_client())


@cl.on_message
async def on_message(msg: cl.Message):
    user_text = msg.content.strip()

    if user_text.lower().startswith("switch"):
        parts = user_text.split(" ", 1)
        if len(parts) == 1 or not parts[1].strip():
            await cl.Message("Usage: `switch Marie Curie`", author="System").send()
            return
        cl.user_session.set("persona", None) 
        cl.user_session.set("history", [])
        await cl.Message(f"🔄 Okay, who would you like to speak with now?").send()
        return

    persona: str | None = cast(str | None, cl.user_session.get("persona"))
    if persona is None:
        persona = user_text.title()
        await cl.Message(f"🕰️ Summoning **{persona}**… one moment.").send()

        bio = await fetch_bio(persona)
        system_prompt = build_system_prompt(persona, bio)

        cl.user_session.set("persona", persona)
        cl.user_session.set("prompt", system_prompt)
        cl.user_session.set("history", [{"role": "system", "content": system_prompt}])
        await cl.Message("Great! Ask your first question.").send()
        return  
    
    history: List[Dict[str, str]] = cast(List[Dict[str, str]], cl.user_session.get("history"))
    history.append({"role": "user", "content": user_text})

    placeholder = await cl.Message("Thinking…").send()

    client: AsyncOpenAI = cast(AsyncOpenAI, cl.user_session.get("client"))
    model: str = cast(str, cl.user_session.get("model"))

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=history,
        )
        assistant_text: str = cast(str, response.choices[0].message.content)
    except Exception as exc:
        placeholder.content = f"❗ Error from model: {exc}"
        await placeholder.update()
        return

    history.append({"role": "assistant", "content": assistant_text})
    cl.user_session.set("history", history)

    placeholder.content = assistant_text
    await placeholder.update()