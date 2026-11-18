import gzip
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMN_ALIASES = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "order_date",
    "price": "price",
    "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}
REQUIRED_COLUMNS = list(dict.fromkeys(COLUMN_ALIASES.values()))

class DataAnalyzer:

    def __init__(self, data_dir=None, seed=42):
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "raw"
        self.seed = seed  # 난수를 쓰는 곳은 시드를 고정해야 결과가 매번 같다
        self.df = None
        self.images = None

    def load_data(self, image_file="train-images-idx3-ubyte.gz", use_cache=True):
        self.df = self._read_transactions(use_cache)
        self.images = self._read_idx_images(self.data_dir / image_file)
        return self.df

    def _read_transactions(self, use_cache):
        cache = self.data_dir / "online_retail_ii.csv"
        if use_cache and cache.exists():
            return pd.read_csv(
                cache,
                parse_dates=["order_date"],
                dtype={"invoice": str, "stock_code": str, "customer_id": "Int64"},
            )

        excel_files = sorted(self.data_dir.glob("*.xlsx"))
        if not excel_files:
            raise FileNotFoundError(
                f"{self.data_dir} 안에 .xlsx 파일이 없습니다. "
                "README의 '데이터 내려받기'를 먼저 따라 하세요."
            )

        # 이 데이터는 엑셀 시트가 2개(기간별)라서 sheet_name=None으로 전부 읽어 이어 붙인다.
        sheets = pd.read_excel(excel_files[0], sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
        df = self._standardize_columns(df)

        df["invoice"] = df["invoice"].astype(str)  # 취소 주문은 C로 시작할 수 있어 문자열로 통일
        df["stock_code"] = df["stock_code"].astype(str)
        df["customer_id"] = df["customer_id"].astype("Int64")  # 정수 + 빈칸 허용
        df.to_csv(cache, index=False)  # 엑셀은 읽는 데 오래 걸려서 CSV로 한 번 저장해 둔다
        return df

    @staticmethod
    def _standardize_columns(df):
        renamed = {}
        for col in df.columns:
            key = "".join(ch for ch in str(col).lower() if ch.isalnum())
            if key in COLUMN_ALIASES:
                renamed[col] = COLUMN_ALIASES[key]
        df = df.rename(columns=renamed)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"필요한 열이 없습니다: {missing} / 현재 열: {list(df.columns)}")
        return df[REQUIRED_COLUMNS]

    @staticmethod
    def _read_idx_images(path):
        """Fashion-MNIST의 idx 파일(gzip 압축)을 (장수, 세로, 가로) 배열로 읽는다."""
        with gzip.open(path, "rb") as f:
            raw = f.read()
        # 파일 앞 16바이트가 머리말: [매직넘버, 장수, 세로, 가로] (각 4바이트, big-endian)
        n, rows, cols = np.frombuffer(raw, dtype=">i4", count=3, offset=4)
        pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
        return pixels.reshape(n, rows, cols)
