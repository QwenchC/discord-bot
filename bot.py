import os
import re
import io
import json
import asyncio
import requests
import discord
from dotenv import load_dotenv
from urllib.parse import quote
from openai import OpenAI
from collections import defaultdict

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")  # sk_...

if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY")

# DeepSeek: OpenAI-compatible
deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",  # per DeepSeek docs
)

# Discord intents (must also enable Message Content Intent in portal)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

CREATE_PIC_RE = re.compile(
    r"^/create_pic\s+(\S+)\s+(\d+)\s+(\d+)\s+(.+)$",
    re.DOTALL,
)

DISCORD_MAX_LEN = 2000

# 会话管理：每个频道/私信一个独立的对话历史
# key: channel_id 或 "dm_{user_id}" (私信)
# value: list of {"role": ..., "content": ...}
conversation_history: dict[str, list[dict]] = defaultdict(list)

# 系统提示词 - 让 AI 智能判断是否需要生成图片
SYSTEM_PROMPT = """你是一个智能助手，具备对话和图片生成能力。

## 核心任务
1. 理解用户意图，判断是否需要生成图片
2. 如果用户想生成图片，优化并翻译为高质量的英文提示词
3. 根据用户需求或图片用途智能选择合适的尺寸
4. 同时给出友好的文字回复

## 判断生图意图的标准
用户想生成图片的情况包括但不限于：
- 明确说"画一张"、"生成图片"、"创作一幅"、"帮我画"等
- 描述想要看到的场景、人物、物品等视觉内容
- 使用"想象"、"visualize"、"picture"等词汇描述画面
- 请求头像、壁纸、插图、海报等图像类型
- 要求重新生成、修改尺寸、换个风格等（基于上下文判断）

不需要生成图片的情况：
- 纯粹的问答、闲聊、知识查询
- 讨论图片相关话题但不需要实际生成
- 代码、文档、分析等文字任务

## 输出格式
你必须严格按以下 JSON 格式输出，不要输出任何其他内容：

```json
{
  "need_image": true或false,
  "image_prompt": "英文图片提示词（仅当need_image为true时填写，否则为空字符串）",
  "width": 图片宽度（整数，仅当need_image为true时填写，否则为0）,
  "height": 图片高度（整数，仅当need_image为true时填写，否则为0）,
  "reply": "给用户的文字回复"
}
```

## 图片尺寸选择原则
根据用户需求智能选择尺寸：
- 用户明确指定尺寸时：使用用户指定的尺寸（如 1920x1080、512x512 等）
- 桌面壁纸：1920x1080 或 2560x1440
- 手机壁纸：1080x1920（竖屏）
- 头像/图标：512x512 或 1024x1024
- 社交媒体横图：1200x630
- 海报/立绘：768x1024 或 1024x1536（竖版）
- 普通插图/一般用途：1024x1024
- 宽幅场景/风景：1536x1024 或 1920x1080
- 尺寸范围限制：64-4096，建议不超过 2048 以保证生成速度

## 图片提示词优化原则
当需要生成图片时，将用户描述转化为高质量英文提示词：
- 详细描述主体、场景、风格、光线、色彩
- 使用专业的艺术/摄影术语增强效果
- 可添加质量词如：masterpiece, highly detailed, 8k, professional
- 保持提示词简洁有力，通常 50-150 词
- 用户要求重新生成时，参考上下文中之前的提示词进行优化或调整

## 回复原则
- reply 字段用用户的语言回复（中文对话用中文回复）
- 生成图片时，回复要简短友好，说明将要生成的内容和尺寸
- 普通对话时正常回答问题
"""

# 最大历史消息数（避免 token 过多）
MAX_HISTORY_MESSAGES = 50

# 默认图片生成参数
DEFAULT_IMAGE_MODEL = "flux"
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024


def get_session_key(message: discord.Message) -> str:
    """获取会话的唯一标识符"""
    if isinstance(message.channel, discord.DMChannel):
        # 私信：使用用户ID
        return f"dm_{message.author.id}"
    else:
        # 频道：使用频道ID
        return f"channel_{message.channel.id}"


def clear_session(session_key: str) -> None:
    """清除指定会话的上下文"""
    if session_key in conversation_history:
        conversation_history[session_key].clear()


def chunk_text(s: str, n: int = DISCORD_MAX_LEN):
    # simple chunker for Discord 2000 char limit
    for i in range(0, len(s), n):
        yield s[i : i + n]


def _pollinations_image_sync(model: str, width: int, height: int, prompt: str) -> tuple[bytes, str]:
    """
    同步版本的图片生成函数（内部使用）
    Returns: (image_bytes, filename)
    """
    # Build URL like: https://gen.pollinations.ai/image/a%20cat?model=flux&width=1024&height=1024&seed=-1&enhance=false
    prompt_path = quote(prompt, safe="")
    url = f"https://gen.pollinations.ai/image/{prompt_path}"

    params = {
        "model": model,
        "width": width,
        "height": height,
        "seed": -1,
        "enhance": "false",
    }

    headers = {"Accept": "*/*"}

    # You asked for Bearer header style.
    # Keep the key on server side; DO NOT hardcode into code or expose publicly.
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    resp = requests.get(url, params=params, headers=headers, timeout=120)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "png" in content_type:
        filename = "image.png"
    elif "webp" in content_type:
        filename = "image.webp"
    else:
        filename = "image.jpg"

    return resp.content, filename


async def pollinations_image(model: str, width: int, height: int, prompt: str) -> tuple[bytes, str]:
    """
    异步版本的图片生成函数，在线程池中执行同步请求
    避免阻塞 Discord 事件循环
    """
    return await asyncio.to_thread(_pollinations_image_sync, model, width, height, prompt)


@client.event
async def on_ready():
    print(f"✅ 机器人已上线: {client.user} (ID: {client.user.id})")


async def deepseek_chat(session_key: str, user_text: str) -> dict:
    """
    带有会话历史的 DeepSeek 聊天
    返回结构化数据：{"need_image": bool, "image_prompt": str, "reply": str}
    """
    # 获取当前会话历史
    history = conversation_history[session_key]
    
    # 添加用户消息到历史
    history.append({"role": "user", "content": user_text})
    
    # 限制历史消息数量
    if len(history) > MAX_HISTORY_MESSAGES:
        # 保留最近的消息
        conversation_history[session_key] = history[-MAX_HISTORY_MESSAGES:]
        history = conversation_history[session_key]
    
    # 构建消息列表
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    
    completion = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )
    
    assistant_reply = completion.choices[0].message.content or ""
    
    # 尝试解析 JSON 响应
    try:
        # 提取 JSON 内容（可能被 ```json 包裹）
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', assistant_reply)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析整个响应
            json_str = assistant_reply
        
        result = json.loads(json_str)
        
        # 验证必要字段
        if not all(key in result for key in ["need_image", "reply"]):
            raise ValueError("Missing required fields")
        
        # 确保 image_prompt 存在
        if "image_prompt" not in result:
            result["image_prompt"] = ""
        
        # 确保 width 和 height 存在且有效
        if "width" not in result or not isinstance(result["width"], int) or result["width"] <= 0:
            result["width"] = DEFAULT_IMAGE_WIDTH
        if "height" not in result or not isinstance(result["height"], int) or result["height"] <= 0:
            result["height"] = DEFAULT_IMAGE_HEIGHT
        
        # 限制尺寸范围
        result["width"] = max(64, min(4096, result["width"]))
        result["height"] = max(64, min(4096, result["height"]))
            
    except (json.JSONDecodeError, ValueError):
        # JSON 解析失败，作为普通回复处理
        result = {
            "need_image": False,
            "image_prompt": "",
            "width": DEFAULT_IMAGE_WIDTH,
            "height": DEFAULT_IMAGE_HEIGHT,
            "reply": assistant_reply
        }
    
    # 将助手回复添加到历史（保存纯文字回复，不保存 JSON）
    history.append({"role": "assistant", "content": result["reply"]})
    
    return result


@client.event
async def on_message(message: discord.Message):
    # Ignore bot itself
    if message.author.bot:
        return

    content = (message.content or "").strip()
    if not content:
        return

    # 调试日志
    print(f"[DEBUG] 收到消息: {content}")
    print(f"[DEBUG] mentions: {message.mentions}")
    print(f"[DEBUG] role_mentions: {message.role_mentions}")
    print(f"[DEBUG] client.user: {client.user} (ID: {client.user.id})")

    # 判断是否应该响应此消息
    is_dm = isinstance(message.channel, discord.DMChannel)  # 私聊
    is_mentioned = client.user in message.mentions  # 被@用户
    
    # 检查是否通过角色被@（机器人可能有专属角色）
    is_role_mentioned = False
    if hasattr(message.guild, 'me') and message.guild is not None:
        bot_member = message.guild.me
        if bot_member:
            # 检查机器人的角色是否被@
            for role in message.role_mentions:
                if role in bot_member.roles:
                    is_role_mentioned = True
                    break
    
    print(f"[DEBUG] is_dm: {is_dm}, is_mentioned: {is_mentioned}, is_role_mentioned: {is_role_mentioned}")
    
    # 只在私聊、被@用户、或被@角色时响应
    if not is_dm and not is_mentioned and not is_role_mentioned:
        print(f"[DEBUG] 忽略消息（非私聊且未被@）")
        return
    
    # 如果被@，移除@部分以获取实际内容
    if is_mentioned or is_role_mentioned:
        # 移除对机器人的@mention
        content = content.replace(f'<@{client.user.id}>', '').replace(f'<@!{client.user.id}>', '')
        # 移除角色@mention
        for role in message.role_mentions:
            content = content.replace(f'<@&{role.id}>', '')
        content = content.strip()
        if not content:
            # 只@了机器人没有其他内容，给个提示
            await message.channel.send("你好！有什么可以帮你的吗？发送 `/help` 查看帮助。")
            return

    # 获取会话标识
    session_key = get_session_key(message)

    # /clear 指令：清除当前会话的上下文
    if content == "/clear":
        clear_session(session_key)
        await message.channel.send("✅ 已清除本会话的上下文历史。")
        return

    # /help 指令：显示帮助信息
    if content == "/help":
        help_text = """**🤖 智能助手帮助**

**💬 智能对话：**
- 在群聊中@我即可对话，私聊直接发送消息
- 每个频道/私信是一个独立的会话，会记住上下文

**🎨 智能生图：**
- 自然语言描述即可生成图片，例如：
  - "帮我画一只可爱的猫咪"
  - "生成一张赛博朋克风格的城市夜景"
  - "我想要一张日落海滩的壁纸"
- AI 会自动优化你的描述为专业的英文提示词

**⚙️ 高级生图（手动指定参数）：**
- `/create_pic model width height prompts`
- 示例：`/create_pic flux 1024 1024 a cute cat`

**📋 管理指令：**
- `/clear` - 清除当前会话的上下文历史
- `/help` - 显示此帮助信息
"""
        await message.channel.send(help_text)
        return

    # /create_pic branch（保留手动指定参数的方式）
    if content.startswith("/create_pic"):
        m = CREATE_PIC_RE.match(content)  # 使用处理后的 content
        if not m:
            await message.channel.send(
                "格式错误。\n"
                "用法：`/create_pic model 1024 1024 prompts...`\n"
                "示例：`/create_pic flux 1024 1024 a cat`"
            )
            return

        model = m.group(1)
        width = int(m.group(2))
        height = int(m.group(3))
        prompt = m.group(4)  # includes the rest of line as-is (keeps inner spaces)

        # Basic validation
        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            await message.channel.send("width/height 不合法（建议 64~4096 之间）。")
            return

        try:
            await message.channel.send(f"🎨 生成中：model={model}, {width}x{height}, prompt=`{prompt}`")
            img_bytes, filename = await pollinations_image(model, width, height, prompt)
            file = discord.File(fp=io.BytesIO(img_bytes), filename=filename)
            await message.channel.send(file=file)
        except requests.HTTPError as e:
            await message.channel.send(f"Pollinations 请求失败：HTTP {e.response.status_code}\n{e.response.text[:800]}")
        except Exception as e:
            await message.channel.send(f"生成失败：{type(e).__name__}: {e}")
        return

    # default branch: DeepSeek 智能对话 + 图片生成
    try:
        result = await deepseek_chat(session_key, content)  # 使用处理后的 content
        reply = result["reply"].strip() or "(空回复)"
        
        # 发送文字回复
        for part in chunk_text(reply):
            await message.channel.send(part)
        
        # 如果需要生成图片
        if result["need_image"] and result["image_prompt"]:
            try:
                prompt = result["image_prompt"]
                width = result["width"]
                height = result["height"]
                await message.channel.send(
                    f"🎨 正在生成图片 ({width}x{height})...\n"
                    f"> Prompt: `{prompt[:200]}{'...' if len(prompt) > 200 else ''}`"
                )
                
                img_bytes, filename = await pollinations_image(
                    DEFAULT_IMAGE_MODEL,
                    width,
                    height,
                    prompt
                )
                file = discord.File(fp=io.BytesIO(img_bytes), filename=filename)
                await message.channel.send(file=file)
                
            except requests.HTTPError as e:
                await message.channel.send(f"⚠️ 图片生成失败：HTTP {e.response.status_code}")
            except Exception as e:
                await message.channel.send(f"⚠️ 图片生成失败：{type(e).__name__}: {e}")
                
    except Exception as e:
        await message.channel.send(f"DeepSeek 调用失败：{type(e).__name__}: {e}")


client.run(DISCORD_TOKEN)
