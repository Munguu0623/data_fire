"""
АЛХАМ 6 — Сарын PDF тайлан үүсгэх
=================================
forecast_data.json + panel_scored.parquet -> reports/monthly_fire_risk_YYYY_MM.pdf

PDF бүтэц:
  1. Cover, ерөнхий статистик
  2. TOP-20 хороо
  3. Дүүргийн нэгтгэл
  4. Өмнөх оны ижил сартай харьцуулсан өөрчлөлт
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

PAGE_W, PAGE_H = A4
MARGIN = 42

FONT_REGULAR = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"

CAUSES = [
    ("risk_ilgal", "Ил гал", colors.HexColor("#e6513d")),
    ("risk_tsakhilgaan", "Цахилгаан", colors.HexColor("#f3a21b")),
    ("risk_yandan", "Яндан/цонолт", colors.HexColor("#2f7d46")),
]

MN_MONTHS = [
    "1-р сар",
    "2-р сар",
    "3-р сар",
    "4-р сар",
    "5-р сар",
    "6-р сар",
    "7-р сар",
    "8-р сар",
    "9-р сар",
    "10-р сар",
    "11-р сар",
    "12-р сар",
]

NAVY = colors.HexColor("#14213d")
TEXT = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#6b7280")
LIGHT_BG = colors.HexColor("#f5f7fb")
GRID = colors.HexColor("#d8dee9")
RED = colors.HexColor("#c62828")
GREEN = colors.HexColor("#2e7d32")
GRAY = colors.HexColor("#7a869a")


def register_fonts() -> None:
    regular = fm.findfont("DejaVu Sans")
    bold = fm.findfont("DejaVu Sans:style=normal:weight=bold")
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, regular))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, bold))
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def pct(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def risk_level(value: float) -> str:
    if value >= 0.70:
        return "Онцгой"
    if value >= 0.50:
        return "Өндөр"
    if value >= 0.30:
        return "Дунд"
    return "Бага"


def mpl_color(color: colors.Color) -> str:
    return color.hexval().replace("0x", "#")


def trend_symbol(delta: float) -> tuple[str, colors.Color]:
    if pd.isna(delta):
        return "→", GRAY
    if delta > 0.03:
        return "↑", RED
    if delta < -0.03:
        return "↓", GREEN
    return "→", GRAY


def draw_header(c: canvas.Canvas, title: str, subtitle: str | None = None) -> None:
    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 16)
    c.drawString(MARGIN, PAGE_H - 45, title)
    if subtitle:
        c.setFillColor(MUTED)
        c.setFont(FONT_REGULAR, 9)
        c.drawString(MARGIN, PAGE_H - 62, subtitle)
    c.setStrokeColor(GRID)
    c.setLineWidth(0.7)
    c.line(MARGIN, PAGE_H - 74, PAGE_W - MARGIN, PAGE_H - 74)


def draw_footer(c: canvas.Canvas, page_no: int, period_label: str) -> None:
    c.setStrokeColor(GRID)
    c.line(MARGIN, 30, PAGE_W - MARGIN, 30)
    c.setFillColor(MUTED)
    c.setFont(FONT_REGULAR, 8)
    c.drawString(MARGIN, 18, f"Гал түймрийн эрсдэлийн сарын тайлан · {period_label}")
    c.drawRightString(PAGE_W - MARGIN, 18, f"{page_no}/4")


def draw_obeg_mark(c: canvas.Canvas, x: float, y: float, size: float = 58) -> None:
    logo_paths = [
        BASE_DIR / "assets" / "obeg_logo.png",
        BASE_DIR / "assets" / "logo.png",
        BASE_DIR.parent / "obeg_logo.png",
        BASE_DIR.parent / "logo.png",
    ]
    for path in logo_paths:
        if path.exists():
            c.drawImage(str(path), x, y, width=size, height=size, preserveAspectRatio=True, mask="auto")
            return

    c.setFillColor(colors.HexColor("#d72631"))
    c.circle(x + size / 2, y + size / 2, size / 2, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(FONT_BOLD, 15)
    c.drawCentredString(x + size / 2, y + size / 2 - 5, "ОБЕГ")


def draw_stat_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str, value: str, sub: str) -> None:
    c.setFillColor(colors.white)
    c.setStrokeColor(GRID)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    c.setFillColor(MUTED)
    c.setFont(FONT_REGULAR, 8.5)
    c.drawString(x + 14, y + h - 22, label)
    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 22)
    c.drawString(x + 14, y + h - 51, value)
    c.setFillColor(MUTED)
    c.setFont(FONT_REGULAR, 8)
    c.drawString(x + 14, y + 13, sub)


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, width: float, font: str, size: float, leading: float) -> float:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    c.setFont(font, size)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def load_report_data() -> tuple[pd.DataFrame, dict, int, int]:
    with open(DATA_DIR / "forecast_data.json", encoding="utf-8") as f:
        forecast = json.load(f)

    meta = forecast.get("meta", {})
    year = int(meta.get("forecast_year") or meta.get("year"))
    month = int(meta.get("forecast_month") or meta.get("month"))

    df = pd.DataFrame(forecast["records"])
    risk_cols = [key for key, _, _ in CAUSES]
    df["score"] = df[risk_cols].max(axis=1)
    df["level"] = df["score"].map(risk_level)

    panel = pd.read_parquet(DATA_DIR / "panel_scored.parquet")
    prev = panel[(panel["year"] == year - 1) & (panel["month"] == month)].copy()
    if not prev.empty:
        prev["prev_score"] = prev[risk_cols].max(axis=1)
        prev = prev[["district", "khoroo", "prev_score", *risk_cols]].rename(
            columns={col: f"prev_{col}" for col, _, _ in CAUSES}
        )
        df = df.merge(prev, on=["district", "khoroo"], how="left")
    else:
        df["prev_score"] = np.nan
        for col, _, _ in CAUSES:
            df[f"prev_{col}"] = np.nan

    df["delta"] = df["score"] - df["prev_score"]
    df = df.sort_values(["score", "district", "khoroo"], ascending=[False, True, True]).reset_index(drop=True)
    return df, meta, year, month


def district_chart(df: pd.DataFrame) -> ImageReader:
    summary = df.groupby("district")[[key for key, _, _ in CAUSES]].mean().sort_index()
    districts = summary.index.tolist()
    x = np.arange(len(districts))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.6, 5.1), dpi=160)
    for idx, (key, label, color) in enumerate(CAUSES):
        values = summary[key].values * 100
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=mpl_color(color))

    ax.set_ylabel("Эрсдэлийн дундаж (%)")
    ax.set_ylim(0, max(75, float(summary.max().max() * 115)))
    ax.set_xticks(x)
    ax.set_xticklabels(districts, rotation=32, ha="right")
    ax.grid(axis="y", color="#d8dee9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncols=3)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d8dee9")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def change_chart(rows: pd.DataFrame, title: str, color: str) -> ImageReader:
    labels = [f"{r.district} {int(r.khoroo)}" for r in rows.itertuples()]
    values = rows["delta"].values * 100

    fig, ax = plt.subplots(figsize=(5.1, 4.15), dpi=160)
    ax.barh(np.arange(len(rows)), values, color=color)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#7a869a", linewidth=0.8)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Өөрчлөлт, пункт")
    ax.grid(axis="x", color="#d8dee9", linewidth=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d8dee9")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def page_cover(c: canvas.Canvas, df: pd.DataFrame, meta: dict, year: int, month: int) -> None:
    period_label = f"{year} оны {MN_MONTHS[month - 1]}"
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 185, PAGE_W, 185, fill=1, stroke=0)
    draw_obeg_mark(c, MARGIN, PAGE_H - 112, 62)

    c.setFillColor(colors.white)
    c.setFont(FONT_BOLD, 26)
    c.drawString(MARGIN, PAGE_H - 145, "Гал түймрийн эрсдэлийн прогноз")
    c.setFont(FONT_REGULAR, 13)
    c.drawString(MARGIN, PAGE_H - 166, f"Сарын PDF тайлан · {period_label}")

    c.setFillColor(colors.HexColor("#eef2f7"))
    c.setFont(FONT_REGULAR, 9)
    generated = meta.get("generated_date", "-")
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 58, f"Үүсгэсэн огноо: {generated}")

    high_count = int((df["score"] >= 0.50).sum())
    critical_count = int((df["score"] >= 0.70).sum())
    avg_score = df["score"].mean()

    c.setFillColor(LIGHT_BG)
    c.rect(0, 0, PAGE_W, PAGE_H - 185, fill=1, stroke=0)
    card_y = PAGE_H - 300
    card_w = (PAGE_W - MARGIN * 2 - 20) / 3
    draw_stat_card(c, MARGIN, card_y, card_w, 92, "Нийт хороо", f"{len(df)}", "Таамаглалд орсон хороо")
    draw_stat_card(c, MARGIN + card_w + 10, card_y, card_w, 92, "Өндөр эрсдэл", f"{high_count}", "50%-иас дээш оноотой")
    draw_stat_card(c, MARGIN + (card_w + 10) * 2, card_y, card_w, 92, "Дундаж оноо", pct(avg_score), "3 шалтгааны хамгийн их утга")

    c.setFillColor(colors.white)
    c.setStrokeColor(GRID)
    c.roundRect(MARGIN, 188, PAGE_W - MARGIN * 2, 210, 7, fill=1, stroke=1)
    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 14)
    c.drawString(MARGIN + 18, 365, "Тайлангийн агуулга")
    c.setFillColor(TEXT)
    c.setFont(FONT_REGULAR, 10)
    lines = [
        "1. Cover — ОБЕГ тэмдэглэгээ, сар/он, ерөнхий статистик",
        "2. TOP-20 хороо — 3 шалтгааны оноо ба хандлага",
        "3. Дүүргийн нэгтгэл — 9 дүүргийн grouped bar chart",
        "4. Өөрчлөлт — өмнөх оны ижил сартай харьцуулсан өсөлт/бууралт",
    ]
    y = 337
    for line in lines:
        c.drawString(MARGIN + 26, y, line)
        y -= 28

    c.setFillColor(MUTED)
    draw_wrapped(
        c,
        "Тайлбар: TOP эрэмбэ болон хандлагыг ил гал, цахилгаан, яндан/цонолтын эрсдэлийн хамгийн их оноогоор тооцсон. "
        "Trend: delta > 0.03 бол өссөн, delta < -0.03 бол буурсан, бусад үед тогтвортой гэж үзэв.",
        MARGIN + 18,
        226,
        PAGE_W - MARGIN * 2 - 36,
        FONT_REGULAR,
        8.5,
        12,
    )
    draw_footer(c, 1, period_label)


def page_top20(c: canvas.Canvas, df: pd.DataFrame, year: int, month: int) -> None:
    period_label = f"{year} оны {MN_MONTHS[month - 1]}"
    draw_header(c, "TOP-20 эрсдэлтэй хороо", "3 шалтгааны оноо болон өмнөх оны ижил сартай харьцуулсан хандлага")

    top = df.head(20).copy()
    x0 = MARGIN
    y = PAGE_H - 100
    row_h = 26
    widths = [30, 100, 45, 60, 78, 84, 64, 56]
    headers = ["#", "Дүүрэг", "Хороо", "Ил гал", "Цахилгаан", "Яндан", "Нийт", "Trend"]

    c.setFillColor(NAVY)
    c.roundRect(x0, y, sum(widths), row_h, 5, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(FONT_BOLD, 8.7)
    x = x0
    for header, w in zip(headers, widths):
        c.drawString(x + 5, y + 9, header)
        x += w

    y -= row_h
    c.setFont(FONT_REGULAR, 8.5)
    for idx, r in top.iterrows():
        c.setFillColor(colors.white if idx % 2 == 0 else colors.HexColor("#f8fafc"))
        c.rect(x0, y, sum(widths), row_h, fill=1, stroke=0)
        c.setStrokeColor(GRID)
        c.line(x0, y, x0 + sum(widths), y)

        symbol, symbol_color = trend_symbol(r["delta"])
        values = [
            str(idx + 1),
            str(r["district"]),
            f"{int(r['khoroo'])}-р",
            pct(r["risk_ilgal"]),
            pct(r["risk_tsakhilgaan"]),
            pct(r["risk_yandan"]),
            pct(r["score"]),
            symbol,
        ]
        x = x0
        for col_idx, (value, w) in enumerate(zip(values, widths)):
            c.setFillColor(symbol_color if col_idx == 7 else TEXT)
            c.setFont(FONT_BOLD if col_idx in [0, 6, 7] else FONT_REGULAR, 8.5)
            c.drawString(x + 5, y + 9, value)
            x += w
        y -= row_h

    c.setFillColor(MUTED)
    c.setFont(FONT_REGULAR, 8)
    c.drawString(MARGIN, 80, "Trend: ↑ өссөн (>3 пункт), ↓ буурсан (<-3 пункт), → тогтвортой.")
    draw_footer(c, 2, period_label)


def page_districts(c: canvas.Canvas, df: pd.DataFrame, year: int, month: int) -> None:
    period_label = f"{year} оны {MN_MONTHS[month - 1]}"
    draw_header(c, "Дүүргийн нэгтгэл", "Дүүрэг бүрийн 3 шалтгааны дундаж эрсдэлийн оноо")
    chart = district_chart(df)
    c.drawImage(chart, MARGIN - 4, 245, width=PAGE_W - MARGIN * 2 + 8, height=405, preserveAspectRatio=True)

    summary = df.groupby("district")["score"].agg(["mean", "max", "count"]).sort_values("mean", ascending=False)
    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 11)
    c.drawString(MARGIN, 205, "Дундаж оноогоор эхний 5 дүүрэг")
    c.setFont(FONT_REGULAR, 9)
    y = 180
    for district, r in summary.head(5).iterrows():
        c.setFillColor(TEXT)
        c.drawString(MARGIN, y, str(district))
        c.setFillColor(MUTED)
        c.drawRightString(PAGE_W - MARGIN, y, f"дунд {pct(r['mean'])} · max {pct(r['max'])} · {int(r['count'])} хороо")
        y -= 20
    draw_footer(c, 3, period_label)


def page_changes(c: canvas.Canvas, df: pd.DataFrame, year: int, month: int) -> None:
    period_label = f"{year} оны {MN_MONTHS[month - 1]}"
    draw_header(c, "Сар хоорондын өөрчлөлт", f"{year - 1} оны {MN_MONTHS[month - 1]}-тай харьцуулсан нийт онооны өөрчлөлт")

    with_prev = df.dropna(subset=["delta"]).copy()
    if with_prev.empty:
        c.setFillColor(TEXT)
        c.setFont(FONT_REGULAR, 11)
        c.drawString(MARGIN, PAGE_H - 140, "Өмнөх оны ижил сарын дата олдсонгүй.")
        draw_footer(c, 4, period_label)
        return

    increased = with_prev.sort_values("delta", ascending=False).head(10)
    decreased = with_prev.sort_values("delta", ascending=True).head(10)
    inc_chart = change_chart(increased, "Хамгийн их нэмэгдсэн", "#c62828")
    dec_chart = change_chart(decreased, "Хамгийн их буурсан", "#2e7d32")
    c.drawImage(inc_chart, MARGIN, 350, width=245, height=300, preserveAspectRatio=True)
    c.drawImage(dec_chart, PAGE_W - MARGIN - 245, 350, width=245, height=300, preserveAspectRatio=True)

    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 11)
    c.drawString(MARGIN, 300, "Тайлбар")
    c.setFillColor(TEXT)
    draw_wrapped(
        c,
        "Өөрчлөлт нь тухайн хорооны 3 шалтгааны хамгийн өндөр эрсдэлийн оноог өмнөх оны ижил сарын оноотой харьцуулсан пунктын зөрүү юм. "
        "Эерэг утга эрсдэл нэмэгдсэнийг, сөрөг утга буурсныг илэрхийлнэ.",
        MARGIN,
        276,
        PAGE_W - MARGIN * 2,
        FONT_REGULAR,
        9,
        13,
    )

    draw_footer(c, 4, period_label)


def generate_report(output_path: Path | None = None) -> Path:
    register_fonts()
    df, meta, year, month = load_report_data()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or REPORT_DIR / f"monthly_fire_risk_{year}_{month:02d}.pdf"

    c = canvas.Canvas(str(output_path), pagesize=A4)
    page_cover(c, df, meta, year, month)
    c.showPage()
    page_top20(c, df, year, month)
    c.showPage()
    page_districts(c, df, year, month)
    c.showPage()
    page_changes(c, df, year, month)
    c.save()
    return output_path


def main() -> None:
    output_path = generate_report()
    print(f"✓ PDF тайлан үүслээ: {output_path}")


if __name__ == "__main__":
    main()
