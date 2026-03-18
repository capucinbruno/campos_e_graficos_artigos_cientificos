"""Factory para configuracoes Dynaconf com cache Singleton."""

from pathlib import Path

from dynaconf import Dynaconf

from app.shared.singleton import SingletonMeta


class SettingsFactory(metaclass=SingletonMeta):
    """Singleton que encapsula o Dynaconf."""

    def __init__(self) -> None:
        self._settings = self._create_settings()

    @staticmethod
    def _create_settings() -> Dynaconf:
        root = Path(__file__).resolve().parents[2]
        return Dynaconf(
            envvar_prefix="AMPERE",
            settings_files=[
                str(root / "app" / "settings" / "settings.toml"),
                str(root / "settings.local.toml"),
                str(root / "app" / "settings" / ".secrets.toml"),
                str(root / "app" / "settings" / "settings.json"),
            ],
            environments=True,
            load_dotenv=True,
            root_path=str(root),
        )

    @property
    def settings(self) -> Dynaconf:
        return self._settings


def get_settings() -> Dynaconf:
    """Atalho para obter o objeto settings."""
    return SettingsFactory().settings


settings = get_settings()
