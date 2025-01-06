from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from contextlib import contextmanager


# Create a custom Console class that adds a prefix
class PrefixedConsole(Console):
    def __init__(self, prefix="| ", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix

    def print(self, *args, **kwargs):
        # Add the prefix to the output
        prefixed_args = [
            f"{self.prefix}{arg}" if isinstance(arg, str) else arg for arg in args
        ]
        super().print(*prefixed_args, **kwargs)


# Context manager to temporarily use the custom console
@contextmanager
def console_prefix(prefix="| "):
    # Create a custom console instance
    custom_console = PrefixedConsole(prefix=prefix)

    # Temporarily override the global console
    import builtins

    original_console = getattr(builtins, "console", None)
    builtins.console = custom_console

    try:
        yield custom_console
    finally:
        # Restore the original console
        if original_console is not None:
            builtins.console = original_console
        else:
            delattr(builtins, "console")


# Example usage
with console_prefix(prefix="| "):
    # Regular print
    console.print("This is a regular message.")

    # Table
    table = Table(title="My Table")
    table.add_column("Name", style="cyan")
    table.add_column("Age", style="magenta")
    table.add_row("Alice", "30")
    table.add_row("Bob", "25")
    console.print(table)

    # Progress bar
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Downloading...", total=100)
        task2 = progress.add_task("[magenta]Processing...", total=100)

        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.update(task2, advance=0.3)
            import time

            time.sleep(0.1)

# Outside the context manager, everything is back to normal
console.print("Back to normal messages.")
