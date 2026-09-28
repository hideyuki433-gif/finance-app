"""JPX の「東証上場銘柄一覧」から data/jp_company_master.csv を再生成する。

使い方: python scripts/update_jp_master.py   （要 openpyxl。JPX は毎月更新）
"""

import io
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
OUT = Path(__file__).resolve().parent.parent / "data" / "jp_company_master.csv"


def main() -> None:
    req = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        raw = pd.read_excel(io.BytesIO(resp.read()), dtype={"コード": str})

    # ETF・REIT 等を除き、株式（内国株式・外国株式）のみ残す
    stocks = raw[raw["市場・商品区分"].str.contains("株式", na=False)]

    out = pd.DataFrame(
        {
            "name": stocks["銘柄名"].map(lambda s: unicodedata.normalize("NFKC", str(s)).strip()),
            "code": stocks["コード"].str.strip(),
            "market": stocks["市場・商品区分"].str.replace("（内国株式）", "", regex=False),
        }
    )
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"{len(out)} 銘柄を書き出しました: {OUT}")


if __name__ == "__main__":
    main()
