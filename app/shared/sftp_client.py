"""Cliente SFTP enxuto com Paramiko."""

from pathlib import Path

import paramiko

from app.shared.settings_factory import get_settings


class SFTPClient:
    """Gerencia conexao SFTP via context manager."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._ssh: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def __enter__(self) -> "SFTPClient":
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(
            hostname=self._settings.SSH_HOST,
            port=int(self._settings.get("SSH_PORT", 22)),
            username=self._settings.SSH_USERNAME,
            key_filename=str(Path(self._settings.SSH_KEY_PATH).expanduser()),
        )
        self._sftp = self._ssh.open_sftp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._sftp:
            self._sftp.close()
        if self._ssh:
            self._ssh.close()

    def download(self, remote_path: str, local_path: str) -> Path:
        """Baixa arquivo do servidor remoto."""
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        self._sftp.get(remote_path, str(local))
        return local

    def upload(self, local_path: str, remote_path: str) -> None:
        """Envia arquivo para o servidor remoto."""
        self._sftp.put(str(local_path), remote_path)

    def listdir(self, remote_path: str) -> list[str]:
        """Lista arquivos em diretorio remoto."""
        return self._sftp.listdir(remote_path)
