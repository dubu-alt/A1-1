import gzip
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = {
    "invoice": "invoice", "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "order_date",
    "price": "price", "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}
KEEP = ["invoice", "stock_code", "description", "quantity",
        "order_date", "price", "customer_id", "country"]


class DataAnalyzer:
    def __init__(self, data_dir=None, seed=42):
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "raw"
        self.seed = seed
        self.df = None
        self.images = None

    def load_data(self, image_file="train-images-idx3-ubyte.gz"):
        self.df = self._read_transactions()
        self.images = self._read_images(self.data_dir / image_file)
        return self.df

    def _read_transactions(self):
        cache = self.data_dir / "online_retail_ii.csv"
        types = {"invoice": str, "stock_code": str, "customer_id": "Int64"}
        if cache.exists():
            return pd.read_csv(cache, parse_dates=["order_date"], dtype=types)

        files = list(self.data_dir.glob("*.xlsx"))
        if not files:
            raise FileNotFoundError(f"{self.data_dir} 안에 .xlsx 파일이 없습니다. README의 데이터 내려받기를 따라 하세요.")
        df = pd.concat(pd.read_excel(files[0], sheet_name=None).values(), ignore_index=True)
        df = df.rename(columns=lambda c: COLUMNS.get(c.lower().replace(" ", "").replace("_", ""), c))
        df = df[KEEP].astype(types)
        df.to_csv(cache, index=False)  # 엑셀은 느려서 CSV로 저장
        return df

    @staticmethod
    def _read_images(path):
        with gzip.open(path, "rb") as f:
            raw = f.read()
        n, rows, cols = np.frombuffer(raw, dtype=">i4", count=3, offset=4)
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)

    # 결측치 처리 부분: 상품명(description) 결측치를 상품코드(stock_code)별로 채우기
    def handle_missing_values(self, target_col="description", group_col="stock_code", strategy="mode"):
        known = self.df.dropna(subset=[target_col]).groupby(group_col)[target_col]
        fill = known.agg(lambda s: s.mode().iloc[0]) if strategy == "mode" else known.agg(strategy)
        self.df[target_col] = self.df[target_col].fillna(self.df[group_col].map(fill))

    # 상품과 이미지의 연결은 난수(시드 고정)로 정한 학습용 시뮬레이션
    def add_image_features(self):
        codes = self.df["stock_code"].unique()
        rng = np.random.default_rng(self.seed)
        imgs = self.images[rng.choice(len(self.images), size=len(codes), replace=False)]
        products = pd.DataFrame({
            "stock_code": codes,
            "img_mean": imgs.mean(axis=(1, 2)),
            "img_std": imgs.std(axis=(1, 2)),
        })
        self.df = self.df.merge(products, on="stock_code", how="left")

    def add_text_features(self):
        text = self.df["description"]
        self.df["name_word_count"] = text.str.split().str.len()
        self.df["name_length"] = text.str.len()

    def clean_data(self):
        df = self.df.drop_duplicates()
        is_sale = (df["quantity"] > 0) & (df["price"] > 0) & ~df["invoice"].str.startswith("C")
        df = df[is_sale].reset_index(drop=True)
        df["amount"] = df["quantity"] * df["price"]
        self.df = df

    