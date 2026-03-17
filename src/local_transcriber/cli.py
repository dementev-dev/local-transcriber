from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(file: Path) -> None:
    typer.echo("TODO: not implemented")


if __name__ == "__main__":
    app()
