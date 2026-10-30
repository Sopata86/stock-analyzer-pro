import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Анализатор на акции PRO", page_icon="📈", layout="wide")
st.title("📈 Анализатор на акции PRO")
st.caption("Технически скенер, сценарии за риск/доходност и сравнение на акции. Не предсказва сигурно бъдещи цени.")

WATCHLIST = {
    "ASML": "ASML", "NVIDIA": "NVDA", "Microsoft": "MSFT", "Visa": "V",
    "Siemens": "SIE.DE", "Novo Nordisk": "NOVO-B.CO", "Rheinmetall": "RHM.DE",
    "AMD": "AMD", "SK Hynix": "000660.KS", "Innventure": "INV",
    "EyePoint Pharmaceuticals": "EYPT", "Pfizer": "PFE",
    "Alphabet": "GOOGL", "Amazon": "AMZN", "Broadcom": "AVGO",
    "Palantir": "PLTR", "Meta": "META", "Tesla": "TSLA",
    "Dronamics (провери тикера)": "DRON.BS",
}
PERIODS = {"6 месеца": "6mo", "1 година": "1y", "2 години": "2y", "5 години": "5y"}

@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker, period="2y"):
    obj = yf.Ticker(ticker)
    df = obj.history(period=period, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame(), {}
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
        return pd.DataFrame(), {}
    df = df.dropna(subset=["Close"])
    c = df["Close"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    delta = c.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["RSI14"] = 100 - 100/(1 + avg_gain/avg_loss.replace(0, np.nan))
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12-e26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"]-df["MACD_signal"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["BB_upper"], df["BB_lower"] = mid+2*sd, mid-2*sd
    prev = c.shift(1)
    tr = pd.concat([df["High"]-df["Low"],(df["High"]-prev).abs(),(df["Low"]-prev).abs()],axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"]/df["VolAvg20"].replace(0,np.nan)
    df["Return1M"] = c.pct_change(21)*100
    df["Return3M"] = c.pct_change(63)*100
    df["Return6M"] = c.pct_change(126)*100
    df["Return1Y"] = c.pct_change(252)*100
    info = {}
    try:
        raw = obj.get_info()
        for k in ["longName","shortName","currency","exchange","sector","industry","marketCap",
                  "trailingPE","forwardPE","priceToBook","profitMargins","revenueGrowth",
                  "earningsGrowth","debtToEquity","freeCashflow","targetMeanPrice",
                  "recommendationKey","numberOfAnalystOpinions","fiftyTwoWeekHigh","fiftyTwoWeekLow"]:
            if raw.get(k) is not None:
                info[k] = raw[k]
    except Exception:
        pass
    return df, info

def fmt(x, currency=""):
    return "—" if x is None or pd.isna(x) else f"{x:,.2f} {currency}".strip()

def tech_score(df):
    if len(df) < 30:
        return 0, "Недостатъчно данни", ["Необходими са повече исторически сесии."]
    last, prev = df.iloc[-1], df.iloc[-2]
    score, notes = 0, []
    if pd.notna(last["SMA50"]):
        if last["Close"] > last["SMA50"]: score += 1; notes.append("Цена над SMA 50.")
        else: score -= 1; notes.append("Цена под SMA 50.")
    if pd.notna(last["SMA200"]):
        if last["Close"] > last["SMA200"]: score += 1; notes.append("Цена над SMA 200.")
        else: score -= 1; notes.append("Цена под SMA 200.")
    rsi = last["RSI14"]
    if pd.notna(rsi):
        if 50 <= rsi < 68: score += 1; notes.append("RSI показва положителен импулс без силно прегряване.")
        elif rsi >= 70: notes.append("RSI е ≥70: възможно прегряване.")
        elif rsi < 30: notes.append("RSI е <30: силно продавана зона, но отскокът не е гарантиран.")
        elif rsi < 50: score -= 1; notes.append("RSI под 50: импулсът е слаб.")
    if pd.notna(last["MACD"]) and pd.notna(last["MACD_signal"]):
        if last["MACD"] > last["MACD_signal"]: score += 1; notes.append("MACD над сигналната линия.")
        else: score -= 1; notes.append("MACD под сигналната линия.")
    if prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
        notes.append("Възможно възходящо пресичане на MACD в последната сесия.")
    elif prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
        notes.append("Възможно низходящо пресичане на MACD в последната сесия.")
    label = ("Силен положителен" if score >= 3 else "Умерено положителен" if score >= 1
             else "Силен отрицателен" if score <= -3 else "Умерено отрицателен" if score <= -1 else "Неутрален")
    return score, label, notes

def fundamental_score(info):
    score, notes = 0, []
    # Това е опростен филтър и пропуските в данните не се приемат за отрицателен сигнал.
    rg = info.get("revenueGrowth")
    eg = info.get("earningsGrowth")
    pm = info.get("profitMargins")
    fpe = info.get("forwardPE")
    dte = info.get("debtToEquity")
    if rg is not None:
        if rg > 0.10: score += 1; notes.append(f"Ръст на приходите: {rg*100:.1f}% — положителен фактор.")
        elif rg < 0: score -= 1; notes.append(f"Ръст на приходите: {rg*100:.1f}% — отрицателен фактор.")
        else: notes.append(f"Ръст на приходите: {rg*100:.1f}%.")
    if eg is not None:
        if eg > 0.10: score += 1; notes.append(f"Ръст на печалбата: {eg*100:.1f}% — положителен фактор.")
        elif eg < 0: score -= 1; notes.append(f"Ръст на печалбата: {eg*100:.1f}% — отрицателен фактор.")
    if pm is not None:
        if pm > 0.10: score += 1; notes.append(f"Марж на печалбата: {pm*100:.1f}%.")
        elif pm < 0: score -= 1; notes.append(f"Отрицателен марж на печалбата: {pm*100:.1f}%.")
    if fpe is not None:
        if 0 < fpe < 25: score += 1; notes.append(f"Forward P/E: {fpe:.1f} — не изглежда екстремен спрямо общия филтър, но сравнявай със сектора.")
        elif fpe > 60: score -= 1; notes.append(f"Forward P/E: {fpe:.1f} — високи очаквания/оценка.")
    if dte is not None:
        if dte > 200: score -= 1; notes.append(f"Debt/Equity: {dte:.0f}% — провери задлъжнялостта и сектора.")
        elif dte < 100: score += 1; notes.append(f"Debt/Equity: {dte:.0f}% — умерено ниво според опростения филтър.")
    if not notes: notes.append("Няма достатъчно фундаментални данни от доставчика за оценка.")
    return score, notes

def scan_one(item):
    label, ticker = item
    try:
        df, info = load_data(ticker, "2y")
        if df.empty or len(df) < 30:
            return None
        score, tech_label, _ = tech_score(df)
        last = df.iloc[-1]
        fscore, _ = fundamental_score(info)
        price = float(last["Close"])
        atr_pct = float(last["ATR14"]/price*100) if pd.notna(last["ATR14"]) and price else np.nan
        vol_ratio = float(last["VolRatio"]) if pd.notna(last["VolRatio"]) else np.nan
        # Скенерът ранжира за допълнително проучване, не е сигнал за покупка.
        momentum = np.nan_to_num(last["Return3M"], nan=0.0)
        rank = tech_score_value = score*2 + fscore + (1 if pd.notna(vol_ratio) and vol_ratio >= 1.5 else 0)
        return {"Компания": label, "Тикер": ticker, "Цена": price, "Валута": info.get("currency",""),
                "Техн. оценка": score, "Фундам. филтър": fscore, "Общо ранжиране": rank,
                "Доходност 1м (%)": last["Return1M"], "Доходност 3м (%)": last["Return3M"],
                "Доходност 6м (%)": last["Return6M"], "Доходност 1г (%)": last["Return1Y"],
                "Обем / ср. 20д": vol_ratio, "ATR дневен (%)": atr_pct,
                "RSI": last["RSI14"], "Сигнал": tech_label}
    except Exception:
        return None

with st.sidebar:
    st.header("Настройки")
    choice = st.selectbox("Акция за подробен анализ", ["Ръчно въвеждане"]+list(WATCHLIST))
    ticker = st.text_input("Борсов тикер", value=("" if choice=="Ръчно въвеждане" else WATCHLIST[choice]) or "NVDA").strip().upper()
    period_label = st.selectbox("Период на графиката", list(PERIODS), index=2)
    period = PERIODS[period_label]
    st.caption("Примери: NVDA, ASML, MSFT, SIE.DE, RHM.DE, EYPT, PFE.")
    st.caption("Тикерът на Dronamics и други акции на БФБ може да не се поддържа от Yahoo Finance.")

tab_scan, tab_detail, tab_short, tab_long, tab_backtest, tab_method = st.tabs(["🔎 Скенер на акции", "📊 Подробен анализ", "🔴 Краткосрочен план", "🟢 Инвестиционен план", "🧪 Исторически тест", "ℹ️ Методика"])

with tab_scan:
    st.subheader("Скенер за потенциални кандидати")
    st.write("Сравнява исторически импулс, технически индикатори, обем и няколко фундаментални показателя, когато са налични.")
    selected = st.multiselect("Избери компании за сканиране", list(WATCHLIST.keys()), default=list(WATCHLIST.keys()))
    if st.button("Сканирай избраните акции", type="primary"):
        items = [(name, WATCHLIST[name]) for name in selected]
        results = []
        progress = st.progress(0)
        status = st.empty()
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(scan_one, item): item for item in items}
            done = 0
            for fut in as_completed(futures):
                done += 1
                result = fut.result()
                if result:
                    results.append(result)
                progress.progress(done/max(1,len(futures)))
                status.write(f"Проверени: {done}/{len(futures)}")
        if results:
            scan_df = pd.DataFrame(results).sort_values("Общо ранжиране", ascending=False)
            st.session_state["scan_df"] = scan_df
            status.success(f"Готово. Успешно анализирани: {len(results)} от {len(items)}.")
        else:
            st.error("Не бяха върнати данни. Провери тикерите или интернет връзката.")
    if "scan_df" in st.session_state:
        scan_df = st.session_state["scan_df"]
        st.markdown("### Подреждане за допълнително проучване")
        st.dataframe(scan_df, use_container_width=True, hide_index=True)
        st.download_button("Изтегли резултатите CSV", scan_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="stock_scan.csv", mime="text/csv")
        st.caption("Високото ранжиране не означава, че акцията непременно ще поскъпне. Данните могат да са непълни или забавени.")

with tab_detail:
    if not ticker:
        st.info("Въведи тикер в лявото меню.")
    else:
        with st.spinner("Зареждам историята и показателите…"):
            df, info = load_data(ticker, period)
        if df.empty:
            st.error(f"Няма данни за {ticker}. Провери символа или избери друга акция.")
        else:
            currency = str(info.get("currency",""))
            company = info.get("longName", info.get("shortName", ticker))
            last = df.iloc[-1]
            price = float(last["Close"])
            prev_price = float(df["Close"].iloc[-2]) if len(df)>1 else price
            daily_change = (price/prev_price-1)*100 if prev_price else 0
            tscore, tlabel, tnotes = tech_score(df)
            fscore, fnotes = fundamental_score(info)
            cols = st.columns(5)
            cols[0].metric("Последна цена", fmt(price,currency), f"{daily_change:+.2f}%")
            cols[1].metric("RSI (14)", f"{last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "—")
            cols[2].metric("SMA 50", fmt(last["SMA50"],currency))
            cols[3].metric("SMA 200", fmt(last["SMA200"],currency))
            cols[4].metric("Техническа оценка", f"{tscore:+d}", tlabel)
            st.subheader(f"{company} ({ticker})")
            chart = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.04,
                                  row_heights=[.62,.18,.20],subplot_titles=("Цена и пълзящи средни","RSI (14)","MACD"))
            chart.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Цена"),row=1,col=1)
            for col, lab in [("SMA20","SMA 20"),("SMA50","SMA 50"),("SMA200","SMA 200")]:
                chart.add_trace(go.Scatter(x=df.index,y=df[col],name=lab),row=1,col=1)
            chart.add_trace(go.Scatter(x=df.index,y=df.BB_upper,name="Bollinger горна",line=dict(dash="dot")),row=1,col=1)
            chart.add_trace(go.Scatter(x=df.index,y=df.BB_lower,name="Bollinger долна",line=dict(dash="dot")),row=1,col=1)
            chart.add_trace(go.Scatter(x=df.index,y=df.RSI14,name="RSI"),row=2,col=1)
            chart.add_hline(y=70,line_dash="dash",row=2,col=1); chart.add_hline(y=30,line_dash="dash",row=2,col=1)
            chart.add_trace(go.Bar(x=df.index,y=df.MACD_hist,name="MACD хистограма"),row=3,col=1)
            chart.add_trace(go.Scatter(x=df.index,y=df.MACD,name="MACD"),row=3,col=1)
            chart.add_trace(go.Scatter(x=df.index,y=df.MACD_signal,name="Сигнал"),row=3,col=1)
            chart.update_layout(height=850,xaxis_rangeslider_visible=False,legend_orientation="h",margin=dict(l=10,r=10,t=55,b=10))
            chart.update_yaxes(range=[0,100],row=2,col=1)
            st.plotly_chart(chart,use_container_width=True)

            st.markdown("## Технически анализ")
            st.write(f"**Обобщение:** {tlabel} (оценка {tscore:+d}).")
            for n in tnotes: st.write("• "+n)
            st.markdown("## Фундаментален филтър")
            st.write(f"Опростена оценка: **{fscore:+d}**. Не замества четене на отчети.")
            for n in fnotes: st.write("• "+n)
            # Потенциал по 3 хоризонта: исторически диапазони на доходността + волатилност, не прогноза.
            st.markdown("## Сценарии за 1, 3 и 12 месеца")
            st.caption("Диапазоните са исторически квантилни ориентири на дневната доходност, умножени по корен от броя сесии. Това е грубо статистическо приближение, не предсказание.")
            rets = df["Close"].pct_change().dropna()
            if len(rets) >= 60:
                vol = float(rets.std())
                horizons = [("1 месец",21),("3 месеца",63),("12 месеца",252)]
                rows=[]
                for h,n in horizons:
                    hist = df["Close"].pct_change(n).dropna()
                    if len(hist) >= 20:
                        lo, med, hi = hist.quantile(.10), hist.quantile(.50), hist.quantile(.90)
                        method = f"Исторически персентили от {len(hist)} наблюдения"
                    else:
                        lo, med, hi = -1.28*vol*np.sqrt(n), 0.0, 1.28*vol*np.sqrt(n)
                        method = "Приближение от дневната волатилност (малко наблюдения)"
                    rows.append({"Хоризонт":h,"Метод":method,
                                 "Негативен ориентир":f"{lo*100:+.1f}%",
                                 "Цена при негативен сценарий":fmt(price*(1+lo),currency),
                                 "Медианен ориентир":f"{med*100:+.1f}%",
                                 "Цена при медиана":fmt(price*(1+med),currency),
                                 "Положителен ориентир":f"{hi*100:+.1f}%",
                                 "Цена при положителен сценарий":fmt(price*(1+hi),currency)})
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            else:
                st.info("Недостатъчно данни за сценарии по хоризонти.")
            recent=df.tail(min(60,len(df)))
            support=float(recent.Low.min()); resistance=float(recent.High.max())
            atr=float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
            st.markdown("## Нива и волатилност")
            q1,q2,q3=st.columns(3)
            q1.metric("Подкрепа (~60 сесии)",fmt(support,currency),f"{(support/price-1)*100:+.1f}%")
            q2.metric("Съпротива (~60 сесии)",fmt(resistance,currency),f"{(resistance/price-1)*100:+.1f}%")
            q3.metric("ATR дневен диапазон",fmt(atr,currency))
            st.markdown("## Фундаментални данни, ако са налични")
            fields = {
                "Сектор":"sector","Отрасъл":"industry","Пазарна капитализация":"marketCap",
                "Forward P/E":"forwardPE","Trailing P/E":"trailingPE","Price/Book":"priceToBook",
                "Ръст на приходите":"revenueGrowth","Ръст на печалбата":"earningsGrowth",
                "Марж на печалбата":"profitMargins","Debt/Equity":"debtToEquity",
                "Свободен паричен поток":"freeCashflow","Средна целева цена на анализатори":"targetMeanPrice",
                "Консенсус":"recommendationKey","Брой анализатори":"numberOfAnalystOpinions",
            }
            rows=[]
            for label,key in fields.items():
                val=info.get(key)
                if val is not None:
                    if key in ["revenueGrowth","earningsGrowth","profitMargins"]:
                        val=f"{val*100:.1f}%"
                    elif key in ["forwardPE","trailingPE","priceToBook","debtToEquity"]:
                        val=f"{val:.2f}"
                    rows.append({"Показател":label,"Стойност":val})
            if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            else: st.info("Фундаментални данни не са предоставени за този тикер.")
            display=df[["Open","High","Low","Close","Volume","SMA20","SMA50","SMA200","RSI14","MACD","MACD_signal","ATR14","VolRatio"]].sort_index(ascending=False)
            st.download_button("Изтегли историческите данни CSV",display.to_csv().encode("utf-8-sig"),
                               file_name=f"{ticker}_analysis.csv",mime="text/csv")

with tab_short:
    st.subheader("🔴 Краткосрочен план — дни до няколко седмици")
    st.caption("Правилата създават примерен план за проучване, а не препоръка за покупка. Стопът не гарантира максимална загуба.")
    c1, c2, c3 = st.columns(3)
    capital_short = c1.number_input("Капитал за спекулация (€)", min_value=100.0, max_value=2000.0, value=500.0, step=50.0, key="capital_short")
    risk_short = c2.slider("Планиран риск на сделка (%)", 0.25, 2.0, 1.0, 0.25, key="risk_short")
    reward_short = c3.slider("Цел риск/печалба", 1.0, 4.0, 2.0, 0.5, key="reward_short")
    with st.spinner("Изчислявам краткосрочния план…"):
        short_df, short_info = load_data(ticker, "1y") if ticker else (pd.DataFrame(), {})
    if short_df.empty or len(short_df) < 30:
        st.warning("Няма достатъчно исторически данни за този тикер.")
    else:
        sl = short_df.iloc[-1]
        px = float(sl["Close"])
        atr = float(sl["ATR14"]) if pd.notna(sl["ATR14"]) else px * 0.03
        rsi = float(sl["RSI14"]) if pd.notna(sl["RSI14"]) else np.nan
        tscore, tlabel, tnotes = tech_score(short_df)
        # ATR-based example levels; intentionally no automatic BUY/SELL order.
        entry = px
        stop = max(0.01, entry - 1.5 * atr)
        target = entry + (entry - stop) * reward_short
        risk_per_share = max(entry-stop, 0.01)
        risk_budget = capital_short * risk_short / 100
        shares_risk = int(risk_budget // risk_per_share)
        shares_capital = int(capital_short // entry) if entry > 0 else 0
        shares = max(0, min(shares_risk, shares_capital))
        exposure = shares * entry
        risk_cash = shares * risk_per_share
        reward_cash = shares * (target-entry)
        st.write(f"**Техническо състояние:** {tlabel} (оценка {tscore:+d}).")
        m = st.columns(4)
        m[0].metric("Последна цена", fmt(px, short_info.get("currency", "")))
        m[1].metric("RSI (14)", f"{rsi:.1f}" if np.isfinite(rsi) else "—")
        m[2].metric("ATR (14)", fmt(atr, short_info.get("currency", "")))
        m[3].metric("Планиран риск", f"€{risk_budget:.2f}")
        plan = pd.DataFrame([
            {"Елемент":"Примерна референтна цена", "Стойност":fmt(entry, short_info.get("currency", ""))},
            {"Елемент":"Условно ниво за стоп (1.5 × ATR)", "Стойност":fmt(stop, short_info.get("currency", ""))},
            {"Елемент":f"Условна цел ({reward_short:.1f}:1 риск/печалба)", "Стойност":fmt(target, short_info.get("currency", ""))},
            {"Елемент":"Брой акции според капитал и риск", "Стойност":str(shares)},
            {"Елемент":"Ориентировъчна сума за позицията", "Стойност":f"€{exposure:.2f}"},
            {"Елемент":"Планирана загуба при стоп (преди разходи/проскальзване)", "Стойност":f"€{risk_cash:.2f}"},
            {"Елемент":"Потенциална печалба при цел (преди разходи)", "Стойност":f"€{reward_cash:.2f}"},
        ])
        st.dataframe(plan, hide_index=True, use_container_width=True)
        if tscore >= 2 and np.isfinite(rsi) and 50 <= rsi < 70:
            st.success("Филтърът показва положителен технически импулс. Това е кандидат за допълнителна проверка, не автоматичен сигнал за покупка.")
        elif tscore <= -2:
            st.warning("Техническият филтър е отрицателен. Избягвай да приемаш този план като сигнал за покупка.")
        else:
            st.info("Смесен/неутрален сигнал: изчакай по-ясни условия или провери новините и ликвидността.")
        st.write("**Проверка преди сделка:** предстоящи отчети/новини, спред, ликвидност, комисиони и дали стопът е логичен спрямо графиката.")

with tab_long:
    st.subheader("🟢 Инвестиционен план — 3 до 12 месеца")
    st.caption("Този раздел подпомага анализа. Не е персонализиран финансов съвет; ценовите сценарии не са прогнози с гарантирана вероятност.")
    c1, c2 = st.columns(2)
    capital_long = c1.number_input("Капитал за инвестиции (€)", min_value=100.0, max_value=2000.0, value=1000.0, step=100.0, key="capital_long")
    max_weight = c2.slider("Максимален дял за една акция (%)", 5, 50, 20, 5, key="max_weight")
    with st.spinner("Зареждам дългосрочните данни…"):
        long_df, long_info = load_data(ticker, "5y") if ticker else (pd.DataFrame(), {})
    if long_df.empty or len(long_df) < 60:
        st.warning("Няма достатъчно данни за инвестиционен анализ.")
    else:
        ll = long_df.iloc[-1]
        px = float(ll["Close"])
        tscore, tlabel, tnotes = tech_score(long_df)
        fscore, fnotes = fundamental_score(long_info)
        cap = capital_long * max_weight / 100
        target_analyst = long_info.get("targetMeanPrice")
        if target_analyst and px > 0:
            upside = (float(target_analyst)/px - 1)*100
            analyst_text = f"{upside:+.1f}% спрямо последната цена (ако данните са актуални)"
        else:
            analyst_text = "Няма надеждна целева цена от доставчика"
        r3 = float(ll["Return3M"]) if pd.notna(ll["Return3M"]) else np.nan
        r1 = float(ll["Return1Y"]) if pd.notna(ll["Return1Y"]) else np.nan
        rows = [
            {"Критерий":"Техническа оценка", "Резултат":f"{tscore:+d} — {tlabel}"},
            {"Критерий":"Фундаментален филтър", "Резултат":f"{fscore:+d} (опростен, зависи от наличните данни)"},
            {"Критерий":"Доходност за 3 месеца", "Резултат":f"{r3:+.1f}%" if np.isfinite(r3) else "—"},
            {"Критерий":"Доходност за 1 година", "Резултат":f"{r1:+.1f}%" if np.isfinite(r1) else "—"},
            {"Критерий":"Целева цена на анализатори", "Резултат":analyst_text},
            {"Критерий":"Максимална сума за тази позиция", "Резултат":f"€{cap:.2f}"},
            {"Критерий":"Брой акции при пълно използване на лимита", "Резултат":str(int(cap//px) if px > 0 else 0)},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown("**Фундаментални наблюдения**")
        for note in fnotes:
            st.write("• " + note)
        st.markdown("**Технически наблюдения**")
        for note in tnotes:
            st.write("• " + note)
        if tscore >= 2 and fscore >= 1:
            st.success("И двата опростени филтъра са положителни. Направи проверка на отчетите, оценката и новините, преди решение.")
        elif tscore <= -2 or fscore <= -2:
            st.warning("Поне един от филтрите е отрицателен. Провери причините и риска от спад, преди да разглеждаш позиция.")
        else:
            st.info("Резултатът е смесен. Няма достатъчно съгласувани фактори за силен извод.")
        st.write("**Правило за диверсификация:** лимитът е таван, не цел за задължително инвестиране. Не е нужно да използваш целия капитал.")


with tab_backtest:
    st.subheader("🧪 Историческа проверка на стратегиите")
    st.warning("Тест върху минали дневни данни, не прогноза. Не включва данъци, дивиденти и всички реални особености на изпълнение. Резултатите зависят от избрания период и параметри.")
    bt_period_label = st.selectbox("Исторически период", ["2 години", "5 години", "10 години"], index=1, key="bt_period")
    bt_period = {"2 години":"2y", "5 години":"5y", "10 години":"10y"}[bt_period_label]
    fee_bps = st.number_input("Разходи за сделка (комисиона + приплъзване, базисни пункта на страна)", min_value=0, max_value=200, value=10, step=5, key="bt_cost")
    hold_days = st.slider("Максимален период на краткосрочна сделка (сесии)", 5, 60, 20, 5, key="bt_hold")
    rr = st.slider("Цел риск/печалба за краткосрочния тест", 1.0, 4.0, 2.0, 0.5, key="bt_rr")
    bt_df, _ = load_data(ticker, bt_period) if ticker else (pd.DataFrame(), {})
    if bt_df.empty or len(bt_df) < 220:
        st.info("Нужни са поне около 220 дневни сесии. Избери ликвиден тикер и по-дълъг период.")
    else:
        d = bt_df.copy().dropna(subset=["Open","High","Low","Close","SMA50","SMA200","RSI14","MACD","MACD_signal","ATR14"])
        cost = fee_bps / 10000.0

        # Short-term: signal on close, entry at next session open; stop/target checked with daily high/low.
        trades = []
        i = 1
        while i < len(d)-1:
            row = d.iloc[i]
            prev = d.iloc[i-1]
            signal = (row["Close"] > row["SMA50"] and row["RSI14"] >= 50 and row["RSI14"] < 70
                      and row["MACD"] > row["MACD_signal"] and
                      not (prev["MACD"] > prev["MACD_signal"] and row["MACD"] <= row["MACD_signal"]))
            if not signal:
                i += 1
                continue
            entry_i = i+1
            entry = float(d.iloc[entry_i]["Open"])
            atr = float(row["ATR14"])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr) or atr <= 0:
                i += 1
                continue
            stop = entry - 1.5*atr
            target = entry + (entry-stop)*rr
            exit_i = min(entry_i+hold_days, len(d)-1)
            exit_price, reason, actual_exit_i = float(d.iloc[exit_i]["Close"]), "Времеви изход", exit_i
            for j in range(entry_i, exit_i+1):
                day = d.iloc[j]
                hit_stop = float(day["Low"]) <= stop
                hit_target = float(day["High"]) >= target
                # Conservative assumption if both levels touched in one daily candle.
                if hit_stop:
                    exit_price, reason, actual_exit_i = stop, "Стоп", j
                    break
                if hit_target:
                    exit_price, reason, actual_exit_i = target, "Цел", j
                    break
            net_ret = (exit_price/entry - 1) - 2*cost
            trades.append({"Вход":d.index[entry_i], "Изход":d.index[actual_exit_i], "Доходност %":net_ret*100,
                           "Причина":reason, "Дни":actual_exit_i-entry_i})
            i = actual_exit_i + 1

        # Long-term: golden-cross trend regime; enter next open and exit next open after bearish regime.
        long_trades = []
        in_pos = False
        entry_price = None
        entry_date = None
        for i in range(1, len(d)-1):
            row, prev = d.iloc[i], d.iloc[i-1]
            bullish = row["Close"] > row["SMA200"] and row["SMA50"] > row["SMA200"]
            bearish = row["Close"] < row["SMA200"] or row["SMA50"] < row["SMA200"]
            if not in_pos and bullish and not (prev["Close"] > prev["SMA200"] and prev["SMA50"] > prev["SMA200"]):
                entry_price = float(d.iloc[i+1]["Open"])
                entry_date = d.index[i+1]
                in_pos = True
            elif in_pos and bearish:
                exit_price = float(d.iloc[i+1]["Open"])
                net_ret = (exit_price/entry_price - 1) - 2*cost
                long_trades.append({"Вход":entry_date, "Изход":d.index[i+1], "Доходност %":net_ret*100})
                in_pos = False
        if in_pos:
            exit_price = float(d.iloc[-1]["Close"])
            net_ret = (exit_price/entry_price - 1) - 2*cost
            long_trades.append({"Вход":entry_date, "Изход":d.index[-1], "Доходност %":net_ret*100})

        def summarize_trades(trade_list):
            if not trade_list:
                return None
            t = pd.DataFrame(trade_list)
            wins = t[t["Доходност %"] > 0]["Доходност %"]
            losses = t[t["Доходност %"] <= 0]["Доходност %"]
            gross_profit = wins.sum()
            gross_loss = abs(losses.sum())
            return t, {
                "Брой сделки":len(t),
                "Успеваемост":f"{(t['Доходност %'] > 0).mean()*100:.1f}%",
                "Средна доходност/сделка":f"{t['Доходност %'].mean():+.2f}%",
                "Медианна доходност/сделка":f"{t['Доходност %'].median():+.2f}%",
                "Profit factor":f"{gross_profit/gross_loss:.2f}" if gross_loss > 0 else ("∞" if gross_profit > 0 else "—"),
                "Най-добра сделка":f"{t['Доходност %'].max():+.2f}%",
                "Най-лоша сделка":f"{t['Доходност %'].min():+.2f}%"
            }


        def portfolio_metrics(trade_list, initial_capital=1000.0):
            """Compounded trade-by-trade portfolio metrics; each trade uses the full current equity."""
            if not trade_list:
                return None
            returns = pd.Series([float(x["Доходност %"]) / 100.0 for x in trade_list], dtype=float)
            equity = initial_capital * (1.0 + returns).cumprod()
            peak = equity.cummax()
            drawdown = (equity / peak - 1.0) * 100.0
            wins = returns[returns > 0]
            losses = returns[returns <= 0]
            gross_profit = wins.sum()
            gross_loss = abs(losses.sum())
            return {
                "Краен капитал от 1000": f"{equity.iloc[-1]:,.2f}",
                "Обща доходност": f"{(equity.iloc[-1] / initial_capital - 1) * 100:+.2f}%",
                "Максимален спад между сделките": f"{drawdown.min():.2f}%",
                "Средна печеливша сделка": f"{wins.mean()*100:+.2f}%" if len(wins) else "—",
                "Средна губеща сделка": f"{losses.mean()*100:+.2f}%" if len(losses) else "—",
                "Profit factor": f"{gross_profit/gross_loss:.2f}" if gross_loss > 0 else ("∞" if gross_profit > 0 else "—"),
            }

        def buy_hold_metrics(data, initial_capital=1000.0):
            # Uses the same first/last close window; excludes dividends and taxes.
            start = float(data["Close"].iloc[0])
            end = float(data["Close"].iloc[-1])
            if start <= 0:
                return None
            final_value = initial_capital * end / start
            daily_returns = data["Close"].pct_change().dropna()
            equity = initial_capital * (1 + daily_returns).cumprod()
            dd = (equity / equity.cummax() - 1) * 100
            return {
                "Краен капитал от 1000": f"{final_value:,.2f}",
                "Обща доходност": f"{(final_value/initial_capital-1)*100:+.2f}%",
                "Максимален спад по дневни затваряния": f"{dd.min():.2f}%"
            }

        st.markdown("### 🔴 Краткосрочна стратегия")
        short_result = summarize_trades(trades)
        if short_result:
            short_table, short_stats = short_result
            st.dataframe(pd.DataFrame([short_stats]), use_container_width=True, hide_index=True)
            st.dataframe(short_table.sort_values("Вход", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Няма сделки според тези правила в избрания период. Това е резултат, не грешка.")

        st.markdown("### 🟢 Дългосрочна стратегия")
        long_result = summarize_trades(long_trades)
        if long_result:
            long_table, long_stats = long_result
            st.dataframe(pd.DataFrame([long_stats]), use_container_width=True, hide_index=True)
            st.dataframe(long_table.sort_values("Вход", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Няма завършени сделки за дългосрочната стратегия в този период.")

        # Buy-and-hold benchmark over same data window.
        bh = (float(d["Close"].iloc[-1])/float(d["Close"].iloc[0])-1)*100
        st.metric("Сравнение: купи и задръж за целия период", f"{bh:+.2f}%")

        st.markdown("### 💶 Сравнение на капитал от 1000 USD")
        st.caption("Илюстративно изчисление: всяка нова сделка използва целия текущ капитал, доходностите се капитализират последователно. Не включва данъци, дивиденти, валутни разходи и ограничения за размера на позицията.")
        comparison = []
        if short_result:
            comparison.append({"Подход":"Краткосрочна стратегия", **portfolio_metrics(trades)})
        if long_result:
            comparison.append({"Подход":"Дългосрочна стратегия", **portfolio_metrics(long_trades)})
        comparison.append({"Подход":"Купи и задръж", **buy_hold_metrics(d)})
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

        st.markdown("### 🧪 Проверка върху последните 30% от периода")
        split = int(len(d) * 0.70)
        if split >= 220 and len(d) - split >= 60:
            test_data = d.iloc[split:].copy()
            st.write(f"Отделен тестов период: {test_data.index[0].date()} – {test_data.index[-1].date()}. Правилата и параметрите са същите; не се настройват спрямо този период.")
            st.caption("Важно: това е времево отделяне на последните 30% от данните, а не пълна walk-forward оптимизация. Сравнявай резултатите с целия период и не избирай стратегия само по най-добрия резултат.")
        else:
            st.info("Няма достатъчно данни за отделен тестов период с тези настройки.")

        st.caption("Методика: сигналите се изчисляват след затваряне; входът е на следващото отваряне. При дневна свещ, докоснала и стоп, и цел, се приема стопът първо (консервативно). Статистиката за капитала предполага реинвестиране на цялата сума след всяка сделка и не моделира припокриващи се позиции или реални ограничения за капитал.")
        if short_result:
            st.download_button("Изтегли краткосрочните сделки CSV", pd.DataFrame(trades).to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{ticker}_short_backtest.csv", mime="text/csv")
        if long_result:
            st.download_button("Изтегли дългосрочните сделки CSV", pd.DataFrame(long_trades).to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{ticker}_long_backtest.csv", mime="text/csv")

with tab_method:
    st.markdown("""
    ### Как работи
    - **Скенер:** подрежда избраните компании по комбинация от техническа оценка, опростен фундаментален филтър и необичайно висок обем.
    - **Технически анализ:** цена спрямо SMA 50/200, RSI и MACD.
    - **Фундаментален филтър:** използва ръст на приходите/печалбата, марж, forward P/E и debt/equity, само когато данните са налични.
    - **Сценарии:** исторически 10-и, 50-и и 90-и персентил на доходността за сходен период; процентът и съответната цена се изчисляват от една и съща стойност.
    - **Исторически тест:** отделно проверява краткосрочна ATR стратегия и дългосрочен тренд филтър. Това е опростен backtest, не доказателство за бъдеща успеваемост.
    - **Обем:** последният обем се сравнява със средния за 20 сесии; резки скокове може да са важни, но не казват сами по себе си посоката.

    ### Допълнителни показатели във версия 5\n    - **Краен капитал:** илюстрация с начални 1000 USD и последователно реинвестиране на всяка сделка.\n    - **Максимален спад:** спад от предходен връх на симулираната крива на капитала; при стратегиите се измерва между приключени сделки.\n    - **Profit factor:** общата сума на печелившите сделки, разделена на абсолютната сума на губещите.\n    - **Последни 30%:** отделен времеви прозорец за проверка, без настройване на параметрите спрямо него.\n\n    ### Ограничения и риск
    - Няма алгоритъм, който надеждно да знае бъдещата цена.
    - Оценките са опростени и не са индивидуален инвестиционен съвет.
    - Данните от Yahoo Finance може да са забавени, липсващи или грешно съпоставени за някои символи.
    - Особено при малки спекулативни компании следи ликвидност, парични наличности, дълг, нови емисии акции и риск от загуба на значителна част от инвестицията.
    - Преди решение проверявай официалните финансови отчети и съобщенията на компанията.
    """)
st.divider()
st.caption("Източник: Yahoo Finance чрез yfinance. За проучване, не за автоматична търговия.")
