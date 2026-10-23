import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRODUCTS_PATH = Path(__file__).resolve().parents[4] / "data" / "products.json"


@dataclass(frozen=True)
class Product:
    id: int
    title: str
    description: str

    @property
    def text(self) -> str:
        """The text that gets searched / embedded."""
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
