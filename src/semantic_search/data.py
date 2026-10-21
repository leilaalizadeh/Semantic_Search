import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_PRODUCTS_PATH = DATA_DIR / "products.json"
AMAZON_PRODUCTS_PATH = DATA_DIR / "amazon_products.json"


@dataclass(frozen=True)
class Product:
    id: int | str
    title: str
    description: str
    # Optional fields: the small sample doesn't have them, the Amazon data does.
    category: str = ""
    price: float | None = None
    rating: float | None = None

    @property
    def text(self) -> str:
        """The text that gets searched / embedded."""
        if self.category:
            return f"{self.title}. Category: {self.category}. {self.description}"
        return f"{self.title}. {self.description}"


# class Product:
#     def __init__(self, id: int, title: str, description: str):
#         self.id = id
#         self.title = title
#         self.description = description

#     def text(self) -> str:
#         return f"{self.title}. {self.description}"

#     def __repr__(self):
#         return f"Product(id={self.id}, title={self.title!r}, description={self.description!r})"


def load_products(path: Path = DEFAULT_PRODUCTS_PATH) -> list[Product]:
    with open(path) as f:
        return [Product(**item) for item in json.load(f)]
