from rich.console import Console
from rich.table import Table

console = Console()


def tabela(titulo: str, dados: list[dict]) -> None:
    if not dados:
        return
    table = Table(title=titulo, show_lines=True)
    for col in dados[0]:
        table.add_column(col, overflow='fold')
    for row in dados:
        table.add_row(*[str(v) if v is not None else '-' for v in row.values()])
    console.print(table)
