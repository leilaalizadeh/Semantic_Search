import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRODUCTS_PATH = Path(__file__).resolve().parents[2] / "data" / "products.json"


@dataclass(frozen=True)
class Product:
    id: int
    title: str
    description: str

    @property
    def text(self) -> str:
        """The text that gets searched / embedded."""
        return f"{self.title}. {self.description}"


def load_products(path: Path = DEFAULT_PRODUCTS_PATH) -> list[Product]:
    with open(path) as f:
        return [Product(**item) for item in json.load(f)]
