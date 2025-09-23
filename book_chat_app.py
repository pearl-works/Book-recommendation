# -*- coding: utf-8 -*-
"""
KU 전공 맞춤 도서추천 (RAG + BM25 + HyDE + Embedding Re-rank)
필요 패키지: streamlit, pandas, numpy, sqlalchemy, rank_bm25, openai, psycopg2-binary(선택)
"""

import os, re, json, hashlib
import os, re, sqlite3, pandas as pd, numpy as np
from typing import List, Dict, Tuple
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from collections import defaultdict
import html
from openai import OpenAI
from google.cloud import bigquery
from google.oauth2 import service_account


# ========= 스타일 (Korea University 톤) =========
KU_CRIMSON = "#A6192E"
KU_BEIGE   = "#F7F3EE"
KU_GRAY    = "#4B4B4B"

with open("assets/korea_university.png", "rb") as f:
    st.set_page_config(page_title="도서 추천 챗봇", page_icon=f.read(), layout="wide")


st.markdown(f"""
<style>
.stApp {{ background: #fff; }}
h1,h2,h3,h4 {{ color: {KU_CRIMSON} !important; }}
[data-testid="stSidebar"] > div:first-child {{
  background: {KU_BEIGE}; border-right: 1px solid rgba(0,0,0,0.06);
}}
.stButton>button, .stForm button[kind="primary"] {{
  background: {KU_CRIMSON} !important; color: #fff !important; border:none !important;
  border-radius: 10px !important; padding: 0.5rem 0.9rem !important;
}}
.rec-card {{ background:#fff; border:1px solid rgba(0,0,0,0.06); border-radius:16px;
             padding:14px 16px; box-shadow:0 8px 24px rgba(0,0,0,0.06); }}
.card-meta {{ color:{KU_GRAY}; font-size:0.9rem; }}
.card-reason-label {{ color:{KU_CRIMSON}; font-weight:700; }}
.chip {{
  display:inline-block; background:{KU_BEIGE}; border:1px solid rgba(0,0,0,0.06);
  border-radius:999px; padding:2px 10px; margin:2px 6px 2px 0; font-size:0.85rem; color:{KU_GRAY};
}}
hr.soft {{ border:none; border-top:1px solid rgba(0,0,0,.06); margin:8px 0 4px; }}
</style>
""", unsafe_allow_html=True)


#Secrets 읽기
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()
PROJECT_ID = st.secrets.get("PROJECT_ID", "").strip()
GCP_SA_INFO = st.secrets.get("gcp_service_account", None)

assert OPENAI_API_KEY.strip(), "환경변수 OPENAI_API_KEY를 먼저 설정하세요."
if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID가 설정되지 않았어요 (.env 또는 환경변수 확인).")

client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-5-mini"


def get_bq_client(project_id: str, sa_info: dict) -> bigquery.Client:
    if not sa_info:
        st.error("gcp_service_account가 Secrets에 없습니다.")
        st.stop()
    creds = service_account.Credentials.from_service_account_info(sa_info)
    return bigquery.Client(project=project_id or sa_info.get("project_id"), credentials=creds)

bq_client = get_bq_client(PROJECT_ID, GCP_SA_INFO)



TABLE = "korea_univ.dm_book_detail"
COL_TITLE  = "book"
COL_AUTHOR = "author"
COL_PUB    = "publisher"
COL_TEXT   = "openai"  

@st.cache_data(show_spinner=False)
def fn_query(sql: str) -> pd.DataFrame:
    return bq_client.query(sql).result().to_dataframe()


@st.cache_data(show_spinner=False)
def load_major_map() -> tuple[dict[str, list[str]], list[str]]:
    # DB에서 전공 목록 읽어서 {단과대:[학과...]} 맵 구성
    df_major = fn_query("""
        SELECT college, department
        FROM korea_univ.dm_major
    """)
    by_college: dict[str, list[str]] = defaultdict(list)
    for _, row in df_major.iterrows():
        by_college[row["college"]].append(row["department"])
    # 정렬/중복제거
    for k in by_college.keys():
        by_college[k] = sorted(set(by_college[k]))
    college_list = sorted(by_college.keys())
    return by_college, college_list


df_books = fn_query('''SELECT * FROM korea_univ.dm_book_detail where author != '' ''')   
by_college, college_list = load_major_map()

# ========= 유틸: 텍스트/파싱 =========
def simple_tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    return [w for w in s.split() if w.strip()]

def extract_section(text: str, n: int, title_kw: str) -> str:
    if not text:
        return ""
    
    text = text.replace("\\n", "\n")
    pat = rf"{n}\.\s*{title_kw}.*?\n([\s\S]+?)(?:\n\s*\d+\.\s|$)"
    m = re.search(pat, text)
    return (m.group(1).strip() if m else "").strip()

def extract_genre_from_gpt(text: str) -> str:
    # "4. 장르 및 핵심 키워드" 섹션 첫 줄을 장르로 간주 (없으면 키워드 전체를 한 줄 요약)
    block = extract_section(text, 4, "장르")
    print('###################################3',block)
    if not block:
        return ""
    first_line = block.splitlines()[0].strip()
    # 콤마/슬래시 기준 정리
    first_line = re.sub(r"\s*[,/]\s*", ", ", first_line)
    return first_line

def extract_summary_from_gpt(text: str) -> str:
    block = extract_section(text, 5, "내용 요약")
    return block if block else text


def detect_intent(q: str) -> str:
    if not q: 
        return "unknown"
    qs = q.strip().lower()
    # 이름 질문
    if any(k in qs for k in ["이름", "who are you", "what is your name", "너 이름", '안녕']):
        return "ask_name"
    # 공연 추천/검색 의도 키워드
    rec_kw = ["추천", "찾아줘", "책", "도서", "전공", "읽", "알려줘", "관심", "어떤"]
    if any(k in q for k in rec_kw):
        return "recommend"
    return "other"

# ========= BM25 인덱스 =========
@st.cache_resource(show_spinner=False)
def build_bm25_index(df: pd.DataFrame):
    docs = []
    for _, r in df.iterrows():
        summary = extract_summary_from_gpt(r[COL_TEXT])
        docs.append(simple_tokenize(f"{r[COL_TITLE]} {summary}"))
    return BM25Okapi(docs)

bm25 = build_bm25_index(df_books)

# ========= 임베딩 & Re-rank =========
@st.cache_data(show_spinner=False)
def embed_text(texts: List[str]) -> np.ndarray:
    if not client:
        # OPENAI_API_KEY 없으면 모두 0으로 리턴(→ BM25만 사용)
        return np.zeros((len(texts), 1536), dtype=np.float32)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T

def hyde_expand(query: str) -> str:
    if not client:
        return query
    sys = "너는 한국어 도서 추천 비서다. 사용자의 요청을 잘 반영한 한 단락의 가상 답변을 만들어 검색 성능을 높여라."
    msg = [{"role":"system","content":sys},{"role":"user","content":query}]
    out = client.chat.completions.create(model=CHAT_MODEL, messages=msg)
    return (out.choices[0].message.content or "").strip()

def build_query_with_major(user_query: str) -> str:
    majors = st.session_state.get("selected_departments", []) or []
    if majors:
        return f"{user_query}\n\n[전공키워드] " + " / ".join(majors)
    return user_query

def rerank(query: str, df_cand: pd.DataFrame, top_k: int = 8, use_hyde: bool = True) -> List[Dict]:
    # HyDE 쿼리 확장
    q = hyde_expand(query) if (use_hyde and client) else query
    # 후보 텍스트 구성(제목 + 요약)
    texts = []
    rows = []
    for _, r in df_cand.iterrows():
        summary = extract_summary_from_gpt(r[COL_TEXT])
        texts.append(f"{r[COL_TITLE]}\n{summary}")
        rows.append(r)
    # 임베딩
    q_vec = embed_text([q])           # (1,d)
    d_vecs = embed_text(texts)        # (n,d)
    sims = cosine_sim(q_vec, d_vecs)[0]  # (n,)
    order = np.argsort(-sims)[:top_k]
    out = []
    for idx in order:
        r = rows[idx]
        out.append({
            "title": r[COL_TITLE],
            "author": r[COL_AUTHOR],
            "publisher": r[COL_PUB],
            "text": r[COL_TEXT],
            "score": float(sims[idx]),
        })
    return out

# ========= 추천 이유 생성 =========
def make_reasons(query: str, items: List[Dict]) -> List[str]:
    if not items:
        return []
    # 장르(키워드) 파싱을 먼저 수행
    for it in items:
        it["genre"] = extract_genre_from_gpt(it.get("text",""))

    # OPENAI_API_KEY 없으면 규칙 기반 한 줄 요약
    if not client:
        reasons = []
        for it in items:
            g = it.get("genre","")
            reasons.append(f"요청과 '{g}'(으)로 묘사된 주제가 맞닿아 있습니다.")
        return reasons

    sys = "너는 한국어 도서 추천 비서다. 각 책이 사용자의 요청·전공 맥락에 왜 맞는지 1~2문장으로 간결히 써라."
    # 모델 입력 축약
    book_blocks = []
    for i, it in enumerate(items, 1):
        desc = extract_summary_from_gpt(it.get("text",""))[:450]
        book_blocks.append({
            "idx": i,
            "title": it.get("title",""),
            "author": it.get("author",""),
            "publisher": it.get("publisher",""),
            "genre": it.get("genre",""),
            "detail": desc
        })
    usr = {
        "user_query": query,
        "format": "JSON list of objects with keys: idx, reason (Korean, 1~2 sentences).",
        "books": book_blocks
    }
    out = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":json.dumps(usr, ensure_ascii=False)}],
    )
    txt = out.choices[0].message.content.strip()
    reasons = [""] * len(items)
    try:
        data = json.loads(txt)
        for obj in data:
            i = int(obj.get("idx", 0)) - 1
            if 0 <= i < len(items):
                reasons[i] = str(obj.get("reason","")).strip()
    except Exception:
        for i, it in enumerate(items):
            g = it.get("genre","")
            reasons[i] = f"요청과 '{g}' 관련성이 높습니다."
    return reasons

def short_acknowledge(user_q: str) -> str:
    """
    GPT-5-mini를 사용해 짧은(1문장) 응답 생성
    """
    sys = "너는 사용자의 요청을 공감하며 1문장으로 짧게 답하는 비서야. 너가 책 추천할 필요는 없어. 공감만해. 반드시 한국어로 존댓말로 응답해."
    msg = [
        {"role":"system","content":sys},
        {"role":"user","content":f"사용자가 '{user_q}' 라고 질문했어. 1문장으로 공감하는 문장을 짧게 응답해줘."}
    ]
    try:
        out = client.chat.completions.create(model="gpt-5-mini", messages=msg)
        return out.choices[0].message.content.strip()
    except Exception:
        # 실패 시 기본 응답
        return "네, 책 추천을 준비해드릴게요!"

# ========= 카드 렌더 =========
def html_escape(text: str) -> str:
    return html.escape(text or "")


def render_book_card(item: Dict[str, Any], rank: int, reason: str = ""):
    """
    item: {
      title, author, publisher, genre, reason, score(optional)
    }
    """
    title_html  = html_escape(str(item.get("title", "")))
    author_html = html_escape(str(item.get("author", "")))
    pub_html    = html_escape(str(item.get("publisher", "")))
    genre_html  = html_escape(str(item.get("genre", "")))
    reason_html = html_escape(str(item.get("reason", ""))) or "이 책이 적합합니다!"

    score_val   = float(item.get("score", 0.0) or 0.0)
    score_help_html = (
        "<div style=\"font-size:12px;color:#64748b;display:flex;align-items:center;gap:6px;\">"
        f"<span>semantic score: {score_val:.4f}</span>"
        "<span class=\"tooltip\" "
        "data-tooltip=\"사용자 질의와 책 설명의 의미 유사도를 0~1로 표시합니다.\n"
        "1에 가까울수록 더 잘 맞아요.\">ⓘ help</span></div>"
    )


    st.markdown(
        f"""
<div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:10px 0;background:#ffffff;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:18px;font-weight:700;">#{rank} {title_html}</div>
    {score_help_html}
  </div>

  <div style="margin-top:6px;color:#111827;display:flex;flex-wrap:wrap;gap:10px;">
    <span>✍️ <b> 저자</b> -{author_html}</span>
    <span>🏢 <b> 출판사</b> -{pub_html}</span>
    <span>📚 <b> 장르</b> -{genre_html}</span>
  </div>

  <div style="margin-top:10px;line-height:1.5;">
    🧠 <b>추천이유</b> — {reason_html}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

# ========= 사이드바: 전공 선택 (교차 누적 + 일괄 삭제 + 빠른 렌더) =========
with st.sidebar:
    st.header("학과 선택")

    # 상태 초기화
    if "selected_map" not in st.session_state:
        st.session_state["selected_map"] = {c: [] for c in college_list}
    else:
        for c in college_list:
            st.session_state["selected_map"].setdefault(c, [])

 
    selected_college = st.selectbox("단과대학", college_list, key="college_select")
    current_departments = by_college.get(selected_college, [])

    with st.form("dept_pick_form", clear_on_submit=False):
        selected_this_college = st.multiselect(
            "학과(복수 선택 가능)",
            options=current_departments,
            default=st.session_state["selected_map"].get(selected_college, []),
            placeholder="학과를 선택하세요",
            key=f"dept_multi_{selected_college}",
        )
        apply_pick = st.form_submit_button("선택 적용")

    if apply_pick:
        st.session_state["selected_map"][selected_college] = selected_this_college
        st.rerun()

    # 전체 집계
    all_pairs = [(c, d) for c in college_list for d in st.session_state["selected_map"][c]]
    st.session_state["selected_pairs"] = all_pairs
    st.session_state["selected_departments"] = [d for _, d in all_pairs]

    # 요약/삭제
    if all_pairs:
        st.markdown(f"**선택된 학과 ({len(all_pairs)}개)**")

        import hashlib
        for c, d in all_pairs:
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                st.write(f"{c} · **{d}**")
            with cols[1]:
                key = "del_" + hashlib.md5(f"{c}|{d}".encode("utf-8")).hexdigest()
                if st.button("✕", key=key, help="이 항목을 선택 해제"):
                    st.session_state["selected_map"][c] = [
                        x for x in st.session_state["selected_map"][c] if x != d
                    ]
                    st.rerun()
    else:
        st.caption("아직 선택된 학과가 없습니다.")

    with st.expander("단과대학별 전체 학과 보기", expanded=False):
        for c in college_list:  # 원래 순서 유지
            st.markdown(f"**{c}**  \n" + " · ".join(by_college[c]))

    st.markdown(
    """
    <div class="sidebar-footer">
    <hr/>
    <div>Copyright. 2025 대학혁신지원사업 연구과제 서진주, 전재현</div>
    </div>
    """,
    unsafe_allow_html=True
        )


# ========= 본문: 질의 → RAG 추천 =========
st.title("KU 전공 맞춤 도서추천")
st.caption("전공(학과) 선택 후, 원하는 주제/톤/난이도를 입력하면 전공에 맞춘 도서를 추천해 드립니다.")

st.markdown("##### ")
tab_reco, tab_help = st.tabs(["✨ 추천", "❓도움말"])

# 대화 히스토리 준비
st.session_state.setdefault("chat", [])

# ============ 탭: 추천 ============
with tab_reco:

    st.markdown("""
    <style>
    /* 1) 채팅 입력창: 화면 하단 고정 + 메인영역만 차지 */
    div[data-testid="stChatInput"]{
    position: fixed;
    z-index: 999;
    bottom: 20px;                 /* 바닥에서 띄우기 */
    right: 24px;                  /* 우측 여백 */
    left: 360px;                  /* 사이드바 폭(대략)만큼 띄우기: 필요시 조정 */
    max-width: 1100px;            /* 너무 길어지지 않도록 상한 */
    margin: 0 auto;               /* 중앙 정렬 느낌 */
    background: white;
    padding: 0.75rem 1rem;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }

    /* 2) 반응형: 화면이 좁을 땐 좌우 여백만 두고 꽉 차게 */
    @media (max-width: 1100px){
    div[data-testid="stChatInput"]{
        left: 16px;
        right: 16px;
        bottom: 12px;
    }
    }

    /* 3) 내용이 chat_input 뒤에 가리지 않도록 메인 영역에 하단 패딩 추가 */
    section.main, div[data-testid="stAppViewContainer"] section{
    padding-bottom: 140px; /* chat_input 높이 + 여유 */
    }
                
    /* ==== Tooltip (semantic score help) - 페이지 밖 넘침 방지 포함 ==== */
    .tooltip {
        position: relative !important;
        display: inline-flex !important;
        align-items: center !important;
        cursor: help !important;
        border-bottom: 1px dotted #94a3b8 !important;
    }
    .tooltip::after {
        content: attr(data-tooltip) !important;
        position: absolute !important;
        right: 0 !important; left: auto !important;
        bottom: calc(100% + 10px) !important;
        transform: none !important;

        display: block !important;
        background: #111827 !important; color: #fff !important; text-align: left !important;
        font-size: 12px !important; line-height: 1.5 !important;
        padding: 8px 10px !important; border-radius: 8px !important;
        box-shadow: 0 8px 20px rgba(0,0,0,.15) !important;

        white-space: pre-line !important; overflow-wrap: anywhere !important; word-break: break-word !important;
        width: max-content !important; max-width: min(320px, calc(100vw - 32px)) !important;

        opacity: 0 !important; pointer-events: none !important; transition: opacity .15s ease !important;
        z-index: 10000 !important;
    }
    .tooltip::before {
        content: "" !important;
        position: absolute !important;
        right: 8px !important; left: auto !important;
        bottom: calc(100% + 4px) !important;
        border-width: 6px !important; border-style: solid !important; border-color: #111827 transparent transparent transparent !important;
        opacity: 0 !important; transition: opacity .15s ease !important; z-index: 10000 !important;
    }
    .tooltip:hover::after, .tooltip:hover::before { opacity: 1 !important; }
    @media (max-width: 420px){
        .tooltip::after { left:50% !important; right:auto !important; transform: translateX(-50%) !important; }
        .tooltip::before { left:50% !important; right:auto !important; transform: translateX(-50%) !important; }
    }

    </style>
    """, unsafe_allow_html=True)

    n_bm25  = st.session_state.get("n_bm25", 80)
    top_k   = st.session_state.get("top_k", 5)
    use_hyde = st.session_state.get("use_hyde", True)

    # --- 유틸: 대화 append ---
    def append_chat(role: str, content: str, cards: list | None = None):
        msg = {"role": role, "content": content}
        if cards:
            msg["cards"] = cards
        st.session_state["chat"].append(msg)

    # --- 히스토리 렌더 ---
    for turn in st.session_state["chat"]:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn.get("content", ""))
            cards = turn.get("cards") if isinstance(turn, dict) else None
            if cards:
                for i, it in enumerate(cards, 1):
                    render_book_card(it, i, it.get("reason", ""))

    # --- 입력창 ---
    selected_cnt = len(all_pairs)
    can_chat = (selected_cnt > 0)
    query = st.chat_input("전공 기반으로 추천받을 주제를 입력하세요", disabled=not can_chat)

    if not can_chat:
        st.info("전공(학과)을 먼저 선택해주세요!")
        st.stop()

    # --- 입력 처리 ---
    if query:
        # 1) 유저 질문 먼저 기록 + 화면 노출
        append_chat("user", query)
        with st.chat_message("user"):
            st.markdown(query)

        # 2) 로딩 → 검색/랭킹
        with st.chat_message("assistant"):
            with st.spinner("전공에 맞는 책을 찾는 중입니다..."):
                intent = detect_intent(query)

                if intent == "ask_name":
                    # 결과 저장
                    append_chat("assistant", "저는 KU센스입니다. 도서추천을 도와드릴게요")
                elif intent != "recommend":
                    append_chat(
                        "assistant",
                        "현재는 **전공기반 책 추천/검색** 질문에 최적화되어 있어요.\n\n"
                        "예시) `AI 윤리의식에 관련된 책 추천해줘`, "
                        "`내 전공은 무슨 책이 유명해?`"
                    )
                else:
                    try:
                        # 전공 반영 쿼리 확장
                        q_aug = build_query_with_major(query)

                        # BM25 후보
                        tokenized_query = simple_tokenize(q_aug)
                        corpus = []
                        for _, r in df_books.iterrows():
                            summary = extract_summary_from_gpt(r[COL_TEXT])
                            corpus.append(simple_tokenize(f"{r[COL_TITLE]} {summary}"))
                        bm25_local = BM25Okapi(corpus)
                        scores = bm25_local.get_scores(tokenized_query)
                        order = np.argsort(-np.asarray(scores))[:int(n_bm25)]
                        cand = df_books.iloc[order].reset_index(drop=True)

                        # 임베딩 재랭킹(+옵션 HyDE)
                        ranked = rerank(q_aug, cand, top_k=int(top_k), use_hyde=use_hyde)

                        # 추천 이유 생성
                        reasons = make_reasons(q_aug, ranked)

                        # 카드 데이터 생성
                        cards = []
                        for i, r in enumerate(ranked):
                            cards.append({
                                "title": r["title"],
                                "author": r["author"],
                                "publisher": r["publisher"],
                                "genre": extract_genre_from_gpt(r.get("text","")),
                                "reason": reasons[i] if i < len(reasons) else "",
                                "score": r.get("score", 0.0)
                            })

                        if cards:
                            ack = short_acknowledge(query)
                            st.markdown(ack)

                            # append_chat("assistant", "### ✅ 추천 결과", cards=cards)
                            append_chat("assistant", ack, cards=cards)
                        else:
                            append_chat("assistant", "조건에 맞는 책을 찾지 못했어요. 전공/주제를 조금 넓혀볼까요?")
                    except Exception as e:
                        append_chat("assistant", f"오류가 발생했어요: {e}")

        # 3) rerun → 위에서 히스토리 순서대로 다시 렌더
        st.rerun()

    # --- 빈 상태 박스 ---
    if not st.session_state["chat"]:
        st.markdown(
            """
            <div class="emptystate" style="border:1px dashed rgba(0,0,0,.15);border-radius:16px;padding:16px;background:#fff;">
              <h4 style="margin:0 0 8px;color:#A6192E;">아직 추천을 시작하지 않았어요</h4>
              <div>아래 예시처럼 물어보세요:</div>
              <ul style="margin-top:6px;">
                <li>조직 커뮤니케이션 초보자용 책 추천해줘</li>
                <li>내 전공은 무슨 책이 적합해?</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
