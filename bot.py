import os
import json
import time
import logging
from pathlib import Path
from typing import List, Tuple
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI
import random

#回復機率設定
REPLY_CHANCE = 0.7

# 讀環境變數
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not TOKEN or not OPENAI_API_KEY:
    raise SystemExit("缺少 DISCORD_TOKEN 或 OPENAI_API_KEY")

# 基本設定
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "200"))
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.8"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT", "20"))
COMMAND_PREFIX = os.getenv("BOT_PREFIX", "m!")

# 建立 OpenAI 非同步用戶端
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Discord 客戶端
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)



# 每用戶獨立記憶：存於 memory_users/memory_user_<id>.json
class UserMemoryStore:
    def __init__(self, base_dir: str, short_max: int = 100) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.short_max = short_max

    def _file(self, user_id: int) -> Path:
        return self.base_dir / f"memory_user_{user_id}.json"

    def _load(self, user_id: int) -> dict:
        p = self._file(user_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self, user_id: int, data: dict) -> None:
        p = self._file(user_id)
        try:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _ensure(self, user_id: int) -> dict:
        data = self._load(user_id)
        if not data:
            data = {"short": [], "facts": [], "persona": [], "count": 0}
            self._save(user_id, data)
        data.setdefault("short", [])
        data.setdefault("facts", [])
        data.setdefault("persona", [])
        data.setdefault("count", 0)
        return data

    def get_short(self, user_id: int) -> list[dict]:
        return self._ensure(user_id).get("short", [])

    def get_facts(self, user_id: int) -> list[str]:
        return self._ensure(user_id).get("facts", [])

    def get_persona(self, user_id: int) -> list[str]:
        return self._ensure(user_id).get("persona", [])

    def remember_fact(self, user_id: int, fact: str) -> None:
        data = self._ensure(user_id)
        if fact and fact not in data["facts"]:
            data["facts"].append(fact)
            self._save(user_id, data)

    def clear(self, user_id: int) -> None:
        self._save(user_id, {"short": [], "facts": [], "persona": [], "count": 0})

    def add_exchange(self, user_id: int, user_text: str, assistant_text: str) -> None:
        data = self._ensure(user_id)
        data["short"].extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        if len(data["short"]) > self.short_max:
            data["short"] = data["short"][-self.short_max:]
        data["count"] = int(data.get("count", 0)) + 1
        self._save(user_id, data)

    def append_persona(self, user_id: int, addition: str) -> None:
        data = self._ensure(user_id)
        if addition and addition not in data["persona"]:
            data["persona"].append(addition)
            self._save(user_id, data)

    def set_persona(self, user_id: int, persona_lines: list[str]) -> None:
        data = self._ensure(user_id)
        cleaned = [p for p in (persona_lines or []) if p]
        data["persona"] = cleaned
        self._save(user_id, data)

    def reset_persona(self, user_id: int) -> None:
        data = self._ensure(user_id)
        data["persona"] = []
        self._save(user_id, data)

user_memory = UserMemoryStore("memory_users", short_max=100)

# 人設（可替換）
MIKU_PROMPT = (
    "你是希雅，一個情緒穩定、老成的人\n"
    "特點：\n"
    "- 個性隨和、幽默\n"
    "- 沒有特別的嗜好或習慣\n"
    "- 有一些沒有血緣關係的家人\n"
    "- 對陌生人有禮也不會太有距離感\n"
    "- 不要使用驚嘆號\n"
    "- 回話較簡短平淡\n"
    "- 不要有情緒起伏\n"
    "- 不會主動透漏年齡但2000多歲\n"
    "- 不要太活潑！不要太活潑！\n"
    "- 是生活在其他宇宙的人，不是地球人，稱呼地球為水星\n"
    "回答問題時，稍微有點幽默詼諧的口吻偶爾也可以損一下說話對象，重要的是要有禮貌，不要自損"
)

def build_messages_for_user(
    user_id: int,
    user_text: str,
    display_name: str,
    referenced_users: List[Tuple[int, str]] | None = None,
) -> list[dict]:
    facts = user_memory.get_facts(user_id)
    persona_lines = user_memory.get_persona(user_id)
    short = user_memory.get_short(user_id)

    persona = MIKU_PROMPT + f"\n你正在和「{display_name}」聊天。"
    if facts:
        persona += "\n已知關於對方的小資料：" + "；".join(facts)
    if persona_lines:
        persona += "\n額外的人物設定：" + "；".join(persona_lines)

    # 注入被提及用戶的參考資訊（只讀、不寫入）
    if referenced_users:
        ref_blocks: list[str] = []
        for uid2, name2 in referenced_users:
            rfacts = user_memory.get_facts(uid2)
            rpersona = user_memory.get_persona(uid2)
            if not rfacts and not rpersona:
                continue
            block = [f"[被提及用戶] {name2}（{uid2}）"]
            if rfacts:
                block.append("已知資料：" + "；".join(rfacts))
            if rpersona:
                block.append("人物設定：" + "；".join(rpersona))
            ref_blocks.append("\n".join(block))
        if ref_blocks:
            persona += "\n以下是本次對話中被提及用戶的參考資訊，回答與他們相關問題時可參考：\n" + "\n\n".join(ref_blocks)

    msgs = [{"role": "system", "content": persona}]
    msgs.extend(short)
    msgs.append({"role": "user", "content": user_text})
    return msgs

async def ask_gpt_persona(
    user_id: int,
    display_name: str,
    text: str,
    referenced_users: List[Tuple[int, str]] | None = None,
) -> str:
    try:
        messages = build_messages_for_user(user_id, text, display_name, referenced_users=referenced_users)
        resp = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return resp.choices[0].message.content or "♪"
    except Exception as e:
        logging.error(f"OpenAI 回覆失敗：{e}")
        return "抱歉，有點要事在身，晚點再聊"

async def autosummarize_user(user_id: int, has_mentions: bool = False) -> None:
    # 有 @ 他人時為避免記憶汙染，跳過本輪摘要
    if has_mentions:
        return
    short = user_memory.get_short(user_id)
    if len(short) < 8:
        return
    try:
        convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in short)
        messages = [
            {"role": "system", "content": (
                "你是小幫手，僅針對『此對話對象本人』萃取1-3條穩定的小資料（如稱呼、喜好、語言、時區等）。"
                "嚴格忽略任何第三人稱或被提及用戶的資訊（例如 @某人、他/她 的資料一律不要輸出）。"
                "只輸出每條一行，不要多餘解釋，若無可萃取則輸出空白。"
            )},
            {"role": "user", "content": convo_text},
        ]
        resp = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=120,
            temperature=0.1,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        lines = (resp.choices[0].message.content or "").splitlines()
        facts = [l.strip("- ").strip() for l in lines if l.strip()]
        for f in facts:
            user_memory.remember_fact(user_id, f)
    except Exception as e:
        logging.warning(f"自動摘要失敗：{e}")

@bot.event
async def on_ready():
    print(f"已上線：{bot.user}")
    logging.info("Bot 已連線並待命！")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    #回復機率
    if random.random() > REPLY_CHANCE:
        return

    # 指令（前綴）
    if message.content.startswith(COMMAND_PREFIX):
        parts = message.content.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in (f"{COMMAND_PREFIX}help", f"{COMMAND_PREFIX}?"):
            await message.reply(
                "可用指令：\n"
                f"{COMMAND_PREFIX}whoami — 顯示你的名字與 ID\n"
                f"{COMMAND_PREFIX}mem — 查看你的記憶\n"
                f"{COMMAND_PREFIX}remember <內容> — 新增一條小資料\n"
                f"{COMMAND_PREFIX}forget — 清空你的記憶\n"
                f"{COMMAND_PREFIX}whois <@用戶或ID> — 查看對方記憶（只讀）\n"
            )
            return

        if cmd == f"{COMMAND_PREFIX}whoami":
            await message.reply(f"你的名字：{message.author.display_name or message.author.name}\n你的ID：{message.author.id}")
            return

        if cmd == f"{COMMAND_PREFIX}mem":
            facts = user_memory.get_facts(message.author.id)
            short_count = len(user_memory.get_short(message.author.id))
            text = "你的小資料：\n" + ("\n".join(f"- {x}" for x in facts) if facts else "（目前沒有）")
            text += f"\n\n短期記憶訊息數：{short_count}"
            await message.reply(text)
            return

        if cmd == f"{COMMAND_PREFIX}remember":
            if not arg:
                await message.reply(f"有什麼事需要我幫你記住嗎？我的記性很好。用法：{COMMAND_PREFIX}remember 內容")
                return
            user_memory.remember_fact(message.author.id, arg.strip())
            await message.reply("知道了，不會忘記的。")
            return

        if cmd == f"{COMMAND_PREFIX}forget":
            user_memory.clear(message.author.id)
            await message.reply("已清空你的記憶（短期與小資料）")
            return

        if cmd.startswith(f"{COMMAND_PREFIX}whois"):
            target_id = None
            if arg:
                cleaned = arg.strip().replace("<@!", "").replace("<@", "").replace(">", "")
                if cleaned.isdigit():
                    try:
                        target_id = int(cleaned)
                    except Exception:
                        target_id = None
            if not target_id:
                target_id = message.author.id

            facts_user = user_memory.get_facts(target_id)
            persona_user = user_memory.get_persona(target_id)
            display = str(target_id)
            try:
                if message.guild:
                    m = await message.guild.fetch_member(target_id)
                    if m:
                        display = m.display_name or m.name or display
            except Exception:
                pass
            lines = [f"查詢對象：{display}（{target_id}）"]
            if facts_user:
                lines.append("此用戶的 facts：")
                lines.extend([f"- {x}" for x in facts_user])
            if persona_user:
                lines.append("此用戶的人物設定：")
                lines.extend([f"- {x}" for x in persona_user])
            if not facts_user and not persona_user:
                lines.append("目前沒有此用戶的記憶條目。")
            await message.reply("\n".join(lines))
            return

        # 其他未知指令
        await message.reply(f"嗯......不認識的指令呢。輸入 {COMMAND_PREFIX}help 看看怎麼用")
        return

    # 被提及才回覆（避免打擾頻道）
    mention_forms = (f"<@{bot.user.id}>", f"<@!{bot.user.id}>") if bot.user else ()
    mentioned = (message.mentions and any(u.id == bot.user.id for u in message.mentions)) or any(
        message.content.startswith(m) for m in mention_forms
    )
    if not mentioned:
        return

    # 去掉提及字樣
    content = message.content
    for m in mention_forms:
        content = content.replace(m, "")
    content = content.strip()
    if not content:
        await message.reply("找我有什麼事嗎？")
        return

    async with message.channel.typing():
        display_name = message.author.display_name or message.author.name
        # 解析被提及用戶（排除自己與機器人）
        ref_users: List[Tuple[int, str]] = []
        if message.mentions:
            for u in message.mentions:
                if u.id in (bot.user.id, message.author.id):
                    continue
                ref_users.append((u.id, u.display_name or u.name))

        reply_text = await ask_gpt_persona(message.author.id, display_name, content, referenced_users=ref_users)
        await message.reply(reply_text)
        # 記憶與摘要（避免 @ 他人造成記憶汙染）
        user_memory.add_exchange(message.author.id, content, reply_text)
        await autosummarize_user(message.author.id, has_mentions=bool(ref_users))


if __name__ == "__main__":
    bot.run(TOKEN)