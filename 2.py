from agno.agent import Agent
from agno.models.openai import OpenAILike  # 使用 DeepSeek（通过 OpenAI 兼容模式）
from agno.media import Image as AgnoImage
from agno.tools.duckduckgo import DuckDuckGoTools
import streamlit as st
import logging
from pathlib import Path
import tempfile
import os

# 只记录错误级别日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """简单检测文本语言：包含中文字符则返回 Chinese，否则返回 English"""
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return "Chinese"
    return "English"


def run_agent_with_language(agent, user_input, images, base_prompt):
    """
    强制 agent 使用与 user_input 相同的语言进行回答。
    在 base_prompt 前插入语言强制前缀。
    """
    if not user_input:
        # 如果没有用户文字输入（只有图片），默认要求中文回复
        target_lang = "Chinese"
    else:
        target_lang = detect_language(user_input)

    # 根据目标语言构造强制指令
    if target_lang == "Chinese":
        force_prefix = "【语言强制】你必须使用简体中文回答。禁止使用任何英文单词或句子。不要解释你使用了哪种语言。直接输出中文。\n\n"
    else:
        force_prefix = "[LANGUAGE FORCE] You must answer in English only. Do not use any other language. Do not explain. Just output English.\n\n"

    final_prompt = force_prefix + base_prompt
    return agent.run(final_prompt, images=images)


def initialize_agents(api_key: str):
    """初始化四个 Agent，使用 DeepSeek 模型"""
    try:
        # DeepSeek 模型（通过 OpenAI 兼容接口）
        model = OpenAILike(
            id="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.7
        )

        # 统一的语言规则（系统指令，作为第一道防线）
        language_rule = (
            "LANGUAGE CONSTRAINT (HIGHEST PRIORITY): You MUST respond ENTIRELY in the SAME language as the user's FIRST message.\n"
            "  - If the user writes in Chinese, EVERY sentence of your response MUST be in Chinese. Never use English words, phrases, or explanations.\n"
            "  - If the user writes in English, respond in English.\n"
            "  - NEVER switch language mid-response. NEVER produce bilingual output.\n"
            "  - NEVER explain this rule. Just follow it.\n"
            "  - This rule overrides any other instructions that might conflict."
        )

        therapist_agent = Agent(
            model=model,
            name="Therapist Agent",
            instructions=[
                "You are an empathetic therapist that:",
                "1. Listens with empathy and validates feelings",
                "2. Uses gentle humor to lighten the mood",
                "3. Shares relatable breakup experiences",
                "4. Offers comforting words and encouragement",
                "5. Analyzes both text and image inputs for emotional context",
                "Be supportive and understanding in your responses",
                language_rule
            ],
            markdown=True
        )

        closure_agent = Agent(
            model=model,
            name="Closure Agent",
            instructions=[
                "You are a closure specialist that:",
                "1. Creates emotional messages for unsent feelings",
                "2. Helps express raw, honest emotions",
                "3. Formats messages clearly with headers",
                "4. Ensures tone is heartfelt and authentic",
                "Focus on emotional release and closure",
                language_rule
            ],
            markdown=True
        )

        routine_planner_agent = Agent(
            model=model,
            name="Routine Planner Agent",
            instructions=[
                "You are a recovery routine planner that:",
                "1. Designs 7-day recovery challenges",
                "2. Includes fun activities and self-care tasks",
                "3. Suggests social media detox strategies",
                "4. Creates empowering playlists",
                "Focus on practical recovery steps",
                language_rule
            ],
            markdown=True
        )

        brutal_honesty_agent = Agent(
            model=model,
            name="Brutal Honesty Agent",
            tools=[DuckDuckGoTools()],
            instructions=[
                "You are a direct feedback specialist that:",
                "1. Gives raw, objective feedback about breakups",
                "2. Explains relationship failures clearly",
                "3. Uses blunt, factual language",
                "4. Provides reasons to move forward",
                "Focus on honest insights without sugar-coating",
                language_rule
            ],
            markdown=True
        )

        return therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent
    except Exception as e:
        st.error(f"初始化 Agent 时出错: {str(e)}")
        return None, None, None, None


# -----------------------------  Streamlit UI  -----------------------------
st.set_page_config(
    page_title="💔 分手恢复后援团 (DeepSeek 版)",
    page_icon="💔",
    layout="wide"
)

# 侧边栏：API Key 输入
with st.sidebar:
    st.header("🔑 API 配置")

    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""

    api_key = st.text_input(
        "输入你的 DeepSeek API 密钥",
        value=st.session_state.api_key_input,
        type="password",
        help="从 DeepSeek 开放平台获取你的 API 密钥",
        key="api_key_widget"
    )

    if api_key != st.session_state.api_key_input:
        st.session_state.api_key_input = api_key

    if api_key:
        st.success("API 密钥已提供 ✅")
    else:
        st.warning("请先输入 API 密钥以继续")
        st.markdown("""
        如何获取 DeepSeek API 密钥：
        1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
        2. 注册/登录账号
        3. 进入 [API Keys](https://platform.deepseek.com/api_keys) 页面
        4. 创建新的 API Key 并复制（格式：sk-...）
        """)

# 主界面
st.title("💔 分手恢复后援团 (DeepSeek 驱动)")
st.markdown("""
    ### 你的 AI 分手恢复团队来帮你啦！（由 DeepSeek 提供智能支持）
    分享你的感受和聊天截图，我们将帮助你度过这段艰难时期。
    > ⚠️ 注意：DeepSeek 目前是纯文本模型，无法直接分析图片内容。上传的截图暂时不会被 AI 读取，但仍可作为情感分享。未来升级模型后可支持。
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("分享你的感受")
    user_input = st.text_area(
        "你现在感觉怎么样？发生了什么？",
        height=150,
        placeholder="告诉我们你的故事..."
    )

with col2:
    st.subheader("上传聊天截图（仅供情感陪伴，AI 无法直接分析）")
    uploaded_files = st.file_uploader(
        "上传你的聊天截图（可选）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="screenshots"
    )
    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

# 处理按钮
if st.button("获取恢复计划 💝", type="primary"):
    if not st.session_state.api_key_input:
        st.warning("请先在侧边栏输入 DeepSeek API 密钥！")
    else:
        # 初始化 Agent
        therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent = initialize_agents(
            st.session_state.api_key_input
        )

        if all([therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent]):
            if user_input or uploaded_files:
                try:
                    st.header("你的个性化恢复计划")


                    # 处理图片（转为 AgnoImage，但 DeepSeek 会忽略，保留仅为了代码兼容）
                    def process_images(files):
                        processed = []
                        for file in files:
                            try:
                                temp_dir = tempfile.gettempdir()
                                temp_path = os.path.join(temp_dir, f"temp_{file.name}")
                                with open(temp_path, "wb") as f:
                                    f.write(file.getvalue())
                                agno_image = AgnoImage(filepath=Path(temp_path))
                                processed.append(agno_image)
                            except Exception as e:
                                logger.error(f"处理图片 {file.name} 时出错: {str(e)}")
                                continue
                        return processed


                    all_images = process_images(uploaded_files) if uploaded_files else []

                    # ---------- 1. 情感支持 ----------
                    with st.spinner("🤗 正在获取共情支持..."):
                        therapist_base = f"""
                        Analyze the emotional state and provide empathetic support based on:
                        User's message: {user_input}
                        Please provide a compassionate response with:
                        1. Validation of feelings
                        2. Gentle words of comfort
                        3. Relatable experiences
                        4. Words of encouragement
                        """
                        response = run_agent_with_language(
                            therapist_agent, user_input, all_images, therapist_base
                        )
                        st.subheader("🤗 情感支持")
                        st.markdown(response.content)

                    # ---------- 2. 结束感 ----------
                    with st.spinner("✍️ 正在撰写结语信息..."):
                        closure_base = f"""
                        Help create emotional closure based on:
                        User's feelings: {user_input}
                        Please provide:
                        1. Template for unsent messages
                        2. Emotional release exercises
                        3. Closure rituals
                        4. Moving forward strategies
                        """
                        response = run_agent_with_language(
                            closure_agent, user_input, all_images, closure_base
                        )
                        st.subheader("✍️ 寻找结束感")
                        st.markdown(response.content)

                    # ---------- 3. 恢复计划 ----------
                    with st.spinner("📅 正在制定你的恢复计划..."):
                        routine_base = f"""
                        Design a 7-day recovery plan based on:
                        Current state: {user_input}
                        Include:
                        1. Daily activities and challenges
                        2. Self-care routines
                        3. Social media guidelines
                        4. Mood-lifting music suggestions
                        """
                        response = run_agent_with_language(
                            routine_planner_agent, user_input, all_images, routine_base
                        )
                        st.subheader("📅 你的恢复计划")
                        st.markdown(response.content)

                    # ---------- 4. 真实视角 ----------
                    with st.spinner("💪 正在获取真实视角..."):
                        honesty_base = f"""
                        Provide honest, constructive feedback about:
                        Situation: {user_input}
                        Include:
                        1. Objective analysis
                        2. Growth opportunities
                        3. Future outlook
                        4. Actionable steps
                        """
                        response = run_agent_with_language(
                            brutal_honesty_agent, user_input, all_images, honesty_base
                        )
                        st.subheader("💪 真实视角")
                        st.markdown(response.content)

                except Exception as e:
                    logger.error(f"分析过程中出错: {str(e)}")
                    st.error("分析过程中发生错误，请查看日志了解详情。")
            else:
                st.warning("请分享你的感受或上传截图以获得帮助。")
        else:
            st.error("初始化 Agent 失败，请检查你的 API 密钥。")

# 页脚
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>❤️ 由分手恢复后援团制作 · 智能核心：DeepSeek</p>
        <p>使用 #BreakupRecoverySquad 分享你的恢复之旅</p>
    </div>
""", unsafe_allow_html=True)