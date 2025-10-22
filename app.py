# -*- coding: utf-8 -*-
import os, io, time, json, math, random, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

# ============== Opsiyonel bağımlılıkları güvenli içe aktarma ==============
PYPDF_OK = True
try:
    from pypdf import PdfReader
except Exception:
    PYPDF_OK = False

YF_OK = True
try:
    import yfinance as yf
except Exception:
    YF_OK = False

import numpy as np

# RAG
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# LLM’ler
try:
    import google.generativeai as genai  # Gemini
except Exception:
    genai = None
try:
    from openai import OpenAI           # OpenAI
except Exception:
    OpenAI = None

# ===================== 0) Genel Ayarlar =====================
load_dotenv()
st.set_page_config(page_title="Akıllı Finans Asistanı", page_icon="💬", layout="wide")

DATA_DIR = Path("data")
CHROMA_DIR = Path(".chroma_store")
DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"

# ============== Tema/CSS ==============
st.markdown("""
<style>
.main .block-container{max-width:1120px; padding-top:1rem; padding-bottom:2rem;}
h1,h2,h3{letter-spacing:.2px}
p,li{line-height:1.6}
.card{background:var(--secondary-background-color);border:1px solid rgba(255,255,255,.06);
  border-radius:14px;padding:16px 18px}
.source-chip{display:inline-block;margin:4px 6px 0 0;padding:4px 10px;border-radius:999px;
  border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);font-size:.86rem}
.stButton>button{border-radius:10px !important;padding:.6rem 1.1rem !important;font-weight:600}
::-webkit-scrollbar{height:10px;width:10px} ::-webkit-scrollbar-thumb{background:#334155;border-radius:10px}
</style>
""", unsafe_allow_html=True)

# ===================== Yardımcı / RAG Fonksiyonları =====================
def get_chroma():
    CHROMA_DIR.mkdir(exist_ok=True)
    return chromadb.Client(Settings(
        anonymized_telemetry=False,
        persist_directory=str(CHROMA_DIR)
    ))

@st.cache_resource(show_spinner=False)
def get_embedder(name: str = DEFAULT_EMBED):
    return SentenceTransformer(name)

def embed_texts(texts: List[str], model: SentenceTransformer):
    return model.encode(texts, show_progress_bar=False).tolist()

def list_json_csv_files() -> List[Path]:
    DATA_DIR.mkdir(exist_ok=True)
    files = list(DATA_DIR.glob("*.json")) + list(DATA_DIR.glob("*.csv"))
    return sorted(files)

def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def ensure_collection(client, name: str):
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

def index_dataset(file_path: Path, embedder: SentenceTransformer) -> str:
    """Dosya adına göre koleksiyon oluşturur ve indeksler. Geriye koleksiyon adını döner."""
    rows = load_rows(file_path)
    coll_name = file_path.stem
    client = get_chroma()
    coll = ensure_collection(client, coll_name)

    # Zaten varsa tekrar ekleme
    try:
        if coll.count() > 0:
            return coll_name
    except Exception:
        pass

    if not rows:
        st.warning(f"⚠️ {file_path.name} boş görünüyor.")
        return coll_name

    with st.spinner(f"🔄 {file_path.name} indeksleniyor…"):
        B = 128
        for i in range(0, len(rows), B):
            batch = rows[i:i+B]
            docs = [str(rec.get("text", "")) for rec in batch]
            ids = [str(rec.get("chunk_id", f"{file_path.stem}-{i+idx}")) for idx, rec in enumerate(batch)]
            metas = [{
                "source": rec.get("source_file", file_path.name),
                "page": rec.get("page", None),
                "code": rec.get("source_code", "")
            } for rec in batch]
            embs = embed_texts(docs, get_embedder(DEFAULT_EMBED))
            coll.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        try:
            client.persist()
        except Exception:
            pass
    return coll_name

def query_collection(coll_name: str, query: str, top_k: int) -> Tuple[List[str], List[Dict[str,Any]]]:
    client = get_chroma()
    coll = ensure_collection(client, coll_name)
    res = coll.query(query_texts=[query], n_results=top_k, include=["documents","metadatas"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas

def multi_query(collection_names: List[str], q: str, k: int) -> Tuple[List[str], List[Dict[str,Any]]]:
    """Birden çok koleksiyondan aynı anda sonuç getir (basit birleştirme)."""
    client = get_chroma()
    agg_docs, agg_metas = [], []
    per = max(2, math.ceil(k / max(1, len(collection_names))))
    for name in collection_names:
        coll = ensure_collection(client, name)
        try:
            res = coll.query(query_texts=[q], n_results=per, include=["documents","metadatas"])
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            for d, m in zip(docs, metas):
                agg_docs.append(d); agg_metas.append(m)
        except Exception:
            continue
    return agg_docs[:k], agg_metas[:k]

def build_prompt(query: str, docs: List[str]) -> str:
    context = "\n\n".join([f"[Kaynak {i+1}]\n{d[:1200]}" for i,d in enumerate(docs)])
    return f"""Aşağıdaki bağlama sadık kalarak Türkçe, kısa ve net yanıt ver.
Kaynak numaralarını ([Kaynak 1] gibi) belirt. Yatırım tavsiyesi verme.

[SORU]
{query}

[BAĞLAM]
{context}
""".strip()

def is_financial_advice(q: str) -> bool:
    ql = q.lower()
    flags = ["yatırım tavsiyesi","hangi hisse","hangi coin","almalı mıyım","satmalı mıyım","garanti getiri"]
    return any(f in ql for f in flags)

# =================== LLM Yardımcıları (liste, retry, çağrı) ===================
def list_gemini_models() -> List[str]:
    available = []
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY","")
    if not (genai and key):
        return available
    try:
        genai.configure(api_key=key)
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available.append(m.name)
    except Exception:
        pass
    return available

def _sleep_backoff(attempt: int, base: float=8.0, jitter: float=0.4):
    delay = base * (1.3 ** attempt) + random.uniform(0, jitter)
    time.sleep(min(delay, 40))

def call_gemini(prompt: str, model_name: str, temperature: float=0.2, stream: bool=True, retries: int=2) -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY","")
    if not (genai and key):
        return "Gemini API yapılandırılmadı."
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name=model_name)
    for attempt in range(retries+1):
        try:
            if stream:
                chunks = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature},
                    stream=True
                )
                return st.write_stream((c.text for c in chunks if hasattr(c,"text") and c.text)) or ""
            resp = model.generate_content(prompt, generation_config={"temperature": temperature})
            return getattr(resp, "text", "") or "Yanıt üretilemedi."
        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower() or "429" in msg:
                if attempt < retries:
                    st.info("🕐 Gemini limiti nedeniyle bekleniyor… tekrar denenecek.")
                    _sleep_backoff(attempt)
                    continue
            return f"Gemini hatası: {e}"
    return "Gemini hatası: yeniden denemeler başarısız."

def call_openai(prompt: str, model: str="gpt-4o-mini", temperature: float=0.2, stream: bool=True, retries: int=2) -> str:
    key = os.getenv("OPENAI_API_KEY","")
    if not (OpenAI and key):
        return "OpenAI API yapılandırılmadı."
    client = OpenAI(api_key=key)
    for attempt in range(retries+1):
        try:
            if stream:
                gen = client.chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role":"user","content":prompt}], stream=True
                )
                def iter_text():
                    for ev in gen:
                        delta = getattr(ev.choices[0], "delta", None)
                        if delta and getattr(delta, "content", None):
                            yield delta.content
                return st.write_stream(iter_text()) or ""
            resp = client.chat.completions.create(
                model=model, temperature=temperature, messages=[{"role":"user","content":prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "429" in msg:
                if attempt < retries:
                    st.info("🕐 OpenAI limiti nedeniyle bekleniyor… tekrar denenecek.")
                    _sleep_backoff(attempt)
                    continue
            return f"OpenAI hatası: {e}"
    return "OpenAI hatası: yeniden denemeler başarısız."

def extractive_fallback(docs: List[str]) -> str:
    take = docs[:3]
    lines = ["• " + d.replace("\n"," ").strip()[:400] + ("..." if len(d)>400 else "") for d in take]
    return "LLM devre dışı. İlgili kaynak özetleri:\n" + "\n".join(lines)

# ====================== PDF -> Chunk Ingestion ======================
def pdf_to_chunks(file_bytes: bytes, filename: str, chunk_words: int=180, overlap: int=80) -> List[Dict[str,Any]]:
    def chunkify(text: str):
        words = text.split()
        out, i = [], 0
        step = max(1, chunk_words-overlap)
        while i < len(words):
            out.append(" ".join(words[i:i+chunk_words]))
            i += step
        return out
    reader = PdfReader(io.BytesIO(file_bytes))
    rows = []
    for pno, page in enumerate(reader.pages, start=1):
        try: txt = page.extract_text() or ""
        except Exception: txt = ""
        chunks = chunkify(txt) if txt.strip() else []
        for ci, ch in enumerate(chunks):
            rows.append({
                "chunk_id": f"{filename}::p{pno:03d}::c{ci:02d}",
                "source_file": filename,
                "source_code": "GENEL",
                "title": Path(filename).stem,
                "page": pno,
                "chunk_index_in_page": ci,
                "text": ch
            })
    return rows

# ====================== Akıllı fiyat/kur niyet algılama ======================
def detect_price_intent(q: str):
    """
    'AAPL fiyatı nedir', 'GARAN.IS kaç TL', 'USD/TRY kaç' gibi soruları algılar.
    Döner:
      {"type":"equity","symbol":"AAPL"}  veya {"type":"fx","base":"USD","quote":"TRY"}
    """
    ql = q.lower()
    # döviz çifti (USD/TRY, EURTRY, usd try)
    fx = re.search(r"\b([A-Z]{3})[\/\s]?([A-Z]{3})\b", q.upper())
    if fx and any(k in ql for k in ["kur","fiyat","kaç","kac","ne","parite","çevir","cevir"]):
        base, quote = fx.group(1), fx.group(2)
        return {"type":"fx","base":base,"quote":quote}
    # hisse sembolü benzeri (AAPL, MSFT, THYAO.IS, GARAN.IS)
    if any(k in ql for k in ["fiyat","kaç","kac","ne","quote","son fiyat"]):
        m = re.search(r"\b([A-Z]{1,5}(?:\.IS)?)\b", q.upper())
        if m: return {"type":"equity","symbol":m.group(1)}
    return None

def fetch_equity_price(symbol: str):
    if not YF_OK: return None, "Bu özellik için `pip install yfinance` gerekli."
    try:
        data = yf.Ticker(symbol).history(period="1d", interval="1m")
        if data.empty: return None, "Veri bulunamadı."
        return float(data["Close"].iloc[-1]), None
    except Exception as e:
        return None, f"Hata: {e}"

def fetch_fx_rate(base: str, quote: str):
    if not YF_OK: return None, "Bu özellik için `pip install yfinance` gerekli."
    try:
        pair = f"{base}{quote}=X"
        data = yf.Ticker(pair).history(period="1d", interval="1m")
        if data.empty: return None, "Kur verisi bulunamadı."
        return float(data["Close"].iloc[-1]), None
    except Exception as e:
        return None, f"Hata: {e}"

# ====================== UI: Başlık & Üst Yerleşim ======================
hero = st.container()
ui_top = st.container()
below = st.container()

with hero:
    st.title("💬 Akıllı Finans Asistanı")
    st.caption("Belgeleri anlayan, veriyi açıklayan yapay zekâ destekli finans rehberi. Sor, finansın yanıt versin.")
    st.caption("Eğitim amaçlıdır; yatırım tavsiyesi vermez.")

if "history" not in st.session_state:
    st.session_state.history = []

# =========================== Sidebar ============================
with st.sidebar:
    st.header("⚙️ Ayarlar")
    mode = st.radio("Mod", ["Genel Finans Taraması", "Belge Tabanlı Tarama"], index=0)
    top_k = st.slider("🔍 Kullanılacak belge parçası sayısı", 2, 10, 5,
                      help="LLM'e bağlam olarak kaç metin parçası gönderileceğini belirler.")
    temperature = st.slider("🎨 Yaratıcılık düzeyi", 0.0, 1.0, 0.2, 0.1,
                            help="Düşük değer = net/teknik. Yüksek değer = daha yaratıcı.")

    st.divider()
    with st.expander("⚙️ Gelişmiş Ayarlar"):
        stream_out = st.toggle("🌊 Akış Modu", value=True,
                               help="Açık olduğunda cevap kelime kelime akar.")
        backend = st.radio("🧠 LLM Motoru", ["Gemini", "OpenAI"], index=0, horizontal=True)

        gem_models = list_gemini_models()
        default_gem = "models/gemini-2.5-flash" if "models/gemini-2.5-flash" in gem_models else (
            gem_models[0] if gem_models else "models/gemini-2.5-flash"
        )
        gem_model_name = st.selectbox("🧩 Gemini Modeli",
                                      options=[default_gem]+[m for m in gem_models if m != default_gem])
        openai_model = st.selectbox("🔷 OpenAI Modeli",
                                    ["gpt-4o-mini","gpt-4o","gpt-4o-mini-translate"], index=0)

    st.divider()
    with st.expander("📂 Veri Kaynakları (RAG)", expanded=False):
        files = list_json_csv_files()
        file_names = [f.name for f in files]
        selected_files = st.multiselect(
            "Koleksiyon(lar) (dosya adı = koleksiyon)",
            options=file_names,
            default=file_names[:1] if file_names else [],
            help="Birden fazla dosya seçersen sorguda bağlam birleştirilir."
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔁 Yeniden indeksle", use_container_width=True, disabled=not selected_files):
                client = get_chroma()
                for fname in selected_files:
                    try: client.delete_collection(Path(fname).stem)
                    except Exception: pass
                st.success("Seçili koleksiyonlar temizlendi. Sorgudan önce otomatik yeniden indekslenecek.")
        with c2:
            if st.button("🗑️ Hepsini Sil", use_container_width=True, disabled=not file_names):
                client = get_chroma()
                for f in file_names:
                    try: client.delete_collection(Path(f).stem)
                    except Exception: pass
                st.success("Tüm koleksiyonlar kaldırıldı.")

        if selected_files:
            client = get_chroma()
            st.caption("İndeks Durumu")
            for fname in selected_files:
                coll = ensure_collection(client, Path(fname).stem)
                try:
                    st.write(f"• **{fname}** — parça sayısı: `{coll.count()}`")
                except Exception:
                    st.write(f"• **{fname}** — henüz indekslenmedi")

        with st.expander("🔍 Örnek Parça Önizleme"):
            try:
                if selected_files:
                    sample = DATA_DIR / selected_files[0]
                    rows = load_rows(sample)
                    st.code(rows[0].get("text", "")[:800] + ("..." if len(rows[0].get("text",""))>800 else ""))
                else:
                    st.info("Önizleme için bir dosya seç.")
            except Exception as e:
                st.warning(f"Önizleme yapılamadı: {e}")

        st.caption("Yeni kaynak ekle:")
        up_tab1, up_tab2 = st.tabs(["📄 PDF", "📑 JSON/CSV"])
        with up_tab1:
            up_pdf = st.file_uploader("PDF yükle (otomatik JSON’a çevrilir)", type=["pdf"], key="pdf_up")
            if up_pdf is not None:
                if not PYPDF_OK:
                    st.error("PDF işlemek için: pip install pypdf")
                else:
                    rows = pdf_to_chunks(up_pdf.read(), up_pdf.name)
                    out = DATA_DIR / f"{Path(up_pdf.name).stem}.json"
                    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.success(f"{out.name} kaydedildi ({len(rows)} parça). Menüden seçebilirsin.")
        with up_tab2:
            up_other = st.file_uploader("JSON/CSV yükle", type=["json","csv"], key="jsoncsv")
            if up_other is not None:
                out = DATA_DIR / up_other.name
                out.write_bytes(up_other.read())
                st.success(f"{up_other.name} kaydedildi. Menüden seçebilirsin.")

# ============== RAG: seçilen dosyaları indeksle (gerekirse) ==============
@st.cache_resource(show_spinner=True)
def prepare_collections(file_list: List[str], embed_name: str):
    embedder = get_embedder(embed_name)
    names = []
    for f in file_list:
        path = DATA_DIR / f
        if path.exists():
            names.append(index_dataset(path, embedder))
    return names

collection_names: List[str] = []
if 'selected_files' in locals() and selected_files and os.path.exists(DATA_DIR.as_posix()):
    collection_names = prepare_collections(selected_files, DEFAULT_EMBED)

# =========================== SEKME + SORU (EN ÜSTTE) ===========================
with ui_top:
    tab_chat, tab_sources, tab_tools = st.tabs(["💬 Sohbet","📚 Kaynaklar","🧰 Finans Araçları"])

    with tab_chat:
        st.subheader("Soru Sor")
        # Form: Enter ile gönder
        with st.form("ask_form", clear_on_submit=False):
            query = st.text_input(
                "Örnek: 'Vadeli mevduat nasıl işler?' veya 'AAPL fiyatı nedir?'",
                placeholder="Sorunuzu yazın…"
            )
            c1, c2, c3 = st.columns([1,1,6])
            with c1:
                ask = st.form_submit_button("Sor", use_container_width=True, type="primary")
            with c2:
                st.form_submit_button("Temizle", use_container_width=True)

        # Geçmiş
        if st.session_state.history:
            with st.expander("🧵 Sohbet Geçmişi", expanded=False):
                for m in st.session_state.history[-10:]:
                    role = "👤" if m["role"]=="user" else "🤖"
                    st.markdown(f"{role} {m['content']}")

    with tab_sources:
        st.subheader("RAG Kaynak Dosyaları")
        files_all = list_json_csv_files()
        if files_all:
            for f in files_all:
                st.write(f"• `{f.name}`  —  {(f.stat().st_size/1024):.1f} KB")
        else:
            st.info("`data/` klasörüne .json/.csv koyduğunda burada listelenir.")

    with tab_tools:
        st.subheader("Hızlı Araçlar")
        tool = st.selectbox("Araç seç:", ["Hisse Fiyatı (yfinance)","Kur Çevirici (yfinance)","Kredi/Loan Ödemesi (PMT)"])

        if tool == "Hisse Fiyatı (yfinance)":
            if not YF_OK:
                st.error("Bu aracın çalışması için: pip install yfinance")
            else:
                ticker = st.text_input("Sembol (örn: AAPL, MSFT, GARAN.IS)", "AAPL")
                if st.button("Fiyatı Getir"):
                    try:
                        data = yf.Ticker(ticker).history(period="1d", interval="1m")
                        if data.empty:
                            st.warning("Veri bulunamadı.")
                        else:
                            last = float(data["Close"].iloc[-1])
                            st.metric(f"{ticker} Son Fiyat", f"{last:.2f}")
                            st.line_chart(data["Close"])
                    except Exception as e:
                        st.error(f"Hata: {e}")

        elif tool == "Kur Çevirici (yfinance)":
            if not YF_OK:
                st.error("Bu aracın çalışması için: pip install yfinance")
            else:
                base = st.text_input("Baz (örn: USD)", "USD").upper()
                quote = st.text_input("Karşı (örn: TRY)", "TRY").upper()
                amount = st.number_input("Tutar", min_value=0.0, value=100.0, step=10.0)
                if st.button("Çevir"):
                    try:
                        pair = f"{base}{quote}=X"
                        data = yf.Ticker(pair).history(period="1d", interval="1m")
                        if data.empty:
                            st.warning("Kur verisi bulunamadı.")
                        else:
                            rate = float(data["Close"].iloc[-1])
                            st.metric("Kur", f"1 {base} = {rate:.4f} {quote}")
                            st.write(f"≈ {amount*rate:.2f} {quote}")
                    except Exception as e:
                        st.error(f"Hata: {e}")

        else:  # Loan PMT
            st.caption("Aylık taksit hesabı: PMT = r*PV / (1-(1+r)^-n)")
            pv = st.number_input("Kredi Tutarı (PV)", min_value=0.0, value=100000.0, step=1000.0)
            annual = st.number_input("Yıllık Faiz (%)", min_value=0.0, value=36.0, step=1.0)
            months = st.number_input("Vade (ay)", min_value=1, value=12, step=1)
            if st.button("Hesapla"):
                r = (annual/100.0)/12.0
                pmt = pv/months if r == 0 else (r*pv)/(1 - (1+r)**(-months))
                st.metric("Aylık Taksit", f"{pmt:,.2f}")
                total = pmt*months
                st.write(f"Toplam Ödeme ≈ {total:,.2f}  |  Toplam Faiz ≈ {total-pv:,.2f}")

# ======================== CEVABI EN ÜSTE YAZDIR (below) ========================
if 'ask' in locals() and ask and 'query' in locals() and query.strip():
    with below:
        start = time.time()
        st.session_state.history.append({"role":"user","content":query})

        # RAG/Model metrikleri
        use_rag = mode.lower().startswith("belge") and bool(collection_names)
        colA, colB, colC = st.columns(3)
        colA.metric("RAG", "AKTİF" if use_rag else "PASİF")
        colB.metric("Top-K", str(top_k))
        colC.metric("Model", backend)

        st.markdown("### 💬 Yanıt")

        # ❶ Akıllı yönlendirme: fiyat/kur isteği mi?
        intent = detect_price_intent(query)
        if intent:
            if intent["type"] == "equity":
                price, err = fetch_equity_price(intent["symbol"])
                if err: st.warning(err)
                else:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric(f"{intent['symbol']} Son Fiyat", f"{price:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption("Bu yanıt canlı piyasa verisinden getirildi (RAG değil).")
                st.caption(f"⏱ Süre: {time.time()-start:.2f}s")
                st.stop()
            elif intent["type"] == "fx":
                rate, err = fetch_fx_rate(intent["base"], intent["quote"])
                if err: st.warning(err)
                else:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Kur", f"1 {intent['base']} = {rate:.4f} {intent['quote']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption("Bu yanıt canlı kur verisinden getirildi (RAG değil).")
                st.caption(f"⏱ Süre: {time.time()-start:.2f}s")
                st.stop()

        # ❷ RAG akışı
        if use_rag:
            if is_financial_advice(query):
                st.warning("Bu uygulama yatırım/alımsatım tavsiyesi vermez. Ancak dokümanlara dayalı genel bilgi sunar.")
            docs, metas = multi_query(collection_names, query, top_k)
            if not docs:
                st.info("Uygun bağlam bulunamadı. Daha genel sor veya başka dosya seç.")
            else:
                prompt = build_prompt(query, docs)
                if backend == "Gemini":
                    answer = call_gemini(prompt, gem_model_name, temperature, stream_out)
                else:
                    answer = call_openai(prompt, openai_model, temperature, stream_out)

                if not answer or "429" in str(answer):
                    st.info(extractive_fallback(docs))
                else:
                    st.session_state.history.append({"role":"assistant","content":str(answer)})

                # Yanıt + kaynaklar (card + chips)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write(answer if 'answer' in locals() else "")
                st.markdown('</div>', unsafe_allow_html=True)

                if metas:
                    st.markdown("##### 🔍 Kaynaklar")
                    chips = []
                    for i, m in enumerate(metas, start=1):
                        label = f"[Kaynak {i}] {m.get('source','-')} s.{m.get('page','-')}"
                        chips.append(f'<span class="source-chip">{label}</span>')
                    st.markdown("".join(chips), unsafe_allow_html=True)

        # ❸ Saf LLM akışı
        else:
            sys_prompt = ("Türkçe konuş. Finansal bilgi ver ama yatırım tavsiyesi verme. "
                          "Gerektiğinde formül ve kısa örnek kullan. Kısa ve net ol.")
            final = f"{sys_prompt}\n\nKullanıcı sorusu: {query}"
            if backend == "Gemini":
                answer = call_gemini(final, gem_model_name, temperature, stream_out)
            else:
                answer = call_openai(final, openai_model, temperature, stream_out)

            if answer:
                st.session_state.history.append({"role":"assistant","content":str(answer)})
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(answer if 'answer' in locals() else "")
            st.markdown('</div>', unsafe_allow_html=True)

        st.caption(f"⏱ Süre: {time.time()-start:.2f}s")
