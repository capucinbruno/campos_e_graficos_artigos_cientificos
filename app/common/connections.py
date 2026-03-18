"""
Módulo de conexões SSH/SFTP para acesso a arquivos remotos.

Este módulo fornece funcionalidades para conectar a servidores remotos via SSH
e realizar operações de arquivo via SFTP. Baseado na implementação bem-sucedida
do projeto backend/agro.

Author: @Bruno Capucin
Created: 2025-10-17
"""

# Bibliotecas padrão
import os
import stat
from pathlib import Path

# Bibliotecas de terceiros
import paramiko

# Módulos locais
from app.shared.logger import logger
from app.shared.settings_factory import settings


def connect_server_with_key():
    """
    Conecta ao servidor remoto usando autenticação por chave privada.

    Lê as configurações do servidor de settings.toml (Dynaconf) e estabelece
    uma conexão SSH segura usando a chave privada especificada.

    Returns:
        tuple: (error: bool, result: paramiko.SSHClient ou str)
               - Se error=False: result é o cliente SSH conectado
               - Se error=True: result é uma string com mensagem de erro

    Exemplo:
        >>> error, ssh_client = connect_server_with_key()
        >>> if not error:
        >>>     sftp = ssh_client.open_sftp()
        >>>     # Usar SFTP...
        >>>     sftp.close()
        >>>     ssh_client.close()
        >>> else:
        >>>     print(f"Erro: {ssh_client}")

    Raises:
        Nenhuma exceção é propagada. Todos os erros são capturados e retornados
        no formato (True, mensagem_erro).
    """
    # Obtém configurações do Dynaconf
    hostname = settings.get('HOSTNAME_REMOTE', '152.67.34.247')
    username = settings.get('USER_SSH', 'ubuntu')
    port = settings.get('PORT_SSH', 22)
    key_path = settings.get('SSH_KEY_PATH', f'{settings.HOME_DIR}/.ssh/meteorologia-oracle-sp.pem')

    logger.info('🔐 Tentando conectar via SSH...')
    logger.info(f'   Servidor: {hostname}:{port}')
    logger.info(f'   Usuário: {username}')
    logger.info(f'   Chave SSH: {key_path}')

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Valida se chave SSH existe
        if not os.path.exists(key_path):
            msg = f'❌ Arquivo de chave SSH não encontrado: {key_path}'
            logger.error(msg)
            logger.error(f'   Verifique se a chave existe em: {key_path}')
            return True, msg

        # Tenta carregar a chave privada
        try:
            private_key = paramiko.RSAKey.from_private_key_file(key_path)
        except paramiko.PasswordRequiredException:
            msg = '❌ Chave SSH requer senha. Este script não suporta chaves com senha.'
            logger.error(msg)
            return True, msg
        except Exception as e:
            msg = f'❌ Erro ao carregar chave SSH: {e}'
            logger.error(msg)
            return True, msg

        # Estabelece conexão SSH
        ssh.connect(
            hostname=hostname,
            port=port,
            username=username,
            pkey=private_key,
            timeout=15,
            banner_timeout=30,
            auth_timeout=30,
        )

        logger.info('✅ Conexão SSH estabelecida com sucesso!')
        logger.info(f'   Servidor: {hostname}')
        return False, ssh

    except FileNotFoundError as err:
        msg = f'❌ Arquivo de chave SSH não encontrado: {key_path}'
        logger.error(f'{msg} | Erro: {err}')
        return True, msg

    except paramiko.AuthenticationException as err:
        msg = f'❌ Falha de autenticação SSH para {hostname}'
        logger.error(msg)
        logger.error('   Verifique:')
        logger.error('   - Chave SSH está correta')
        logger.error(f'   - Permissões da chave (chmod 400 {key_path})')
        logger.error(f'   - Usuário {username} tem acesso ao servidor')
        logger.error(f'   Erro detalhado: {err}')
        return True, msg

    except paramiko.SSHException as err:
        msg = f'❌ Erro SSH ao conectar em {hostname}: {err}'
        logger.error(msg)
        return True, msg

    except TimeoutError as err:
        msg = f'❌ Timeout ao conectar em {hostname}. Servidor pode estar inacessível.'
        logger.error(msg)
        logger.error(f'   Erro: {err}')
        return True, msg

    except Exception as err:
        msg = f'❌ ERRO inesperado ao conectar no servidor remoto {hostname}: {err}'
        logger.exception(msg)
        return True, msg


def is_remote_file(sftp, path: str) -> bool:
    """
    Verifica se um caminho aponta para um arquivo remoto válido.

    Args:
        sftp (paramiko.SFTPClient): Cliente SFTP ativo
        path (str): Caminho completo do arquivo no servidor remoto

    Returns:
        bool: True se é um arquivo regular, False caso contrário

    Exemplo:
        >>> if is_remote_file(sftp, '/home/ubuntu/dados/psi850.nc'):
        >>>     print("Arquivo existe!")
    """
    try:
        file_attr = sftp.stat(path)
        is_file = stat.S_ISREG(file_attr.st_mode)

        if is_file:
            size_mb = file_attr.st_size / (1024 * 1024)
            logger.debug(f'📄 Arquivo remoto encontrado: {path} ({size_mb:.2f} MB)')

        return is_file

    except FileNotFoundError:
        logger.warning(f'⚠️  Arquivo remoto não encontrado: {path}')
        return False

    except Exception as e:
        logger.error(f'❌ Erro ao verificar arquivo remoto {path}: {e}')
        return False


def is_remote_directory(sftp, path: str) -> bool:
    """
    Verifica se um caminho é um diretório remoto válido.

    Args:
        sftp (paramiko.SFTPClient): Cliente SFTP ativo
        path (str): Caminho completo do diretório no servidor remoto

    Returns:
        bool: True se é um diretório, False caso contrário

    Exemplo:
        >>> if is_remote_directory(sftp, '/home/ubuntu/dados'):
        >>>     print("Diretório existe!")
    """
    try:
        file_attr = sftp.stat(path)
        is_dir = stat.S_ISDIR(file_attr.st_mode)

        if is_dir:
            logger.debug(f'📁 Diretório remoto encontrado: {path}')

        return is_dir

    except FileNotFoundError:
        logger.warning(f'⚠️  Diretório remoto não encontrado: {path}')
        return False

    except Exception as e:
        logger.error(f'❌ Erro ao verificar diretório remoto {path}: {e}')
        return False


def list_remote_files(sftp, remote_path: str):
    """
    Lista arquivos regulares em um diretório remoto.

    Args:
        sftp (paramiko.SFTPClient): Cliente SFTP ativo
        remote_path (str): Caminho do diretório remoto

    Returns:
        tuple: (error: bool, files: list)
               - Se error=False: files contém lista de nomes de arquivos
               - Se error=True: files é uma lista vazia

    Exemplo:
        >>> error, files = list_remote_files(sftp, '/home/ubuntu/dados')
        >>> if not error:
        >>>     for file in files:
        >>>         print(file)
    """
    try:
        files = []
        for entry in sftp.listdir_attr(remote_path):
            if stat.S_ISREG(entry.st_mode):
                files.append(entry.filename)

        logger.info(f'📂 Listados {len(files)} arquivos em {remote_path}')
        return False, files

    except FileNotFoundError:
        msg = f'❌ Diretório remoto não encontrado: {remote_path}'
        logger.error(msg)
        return True, []

    except Exception as e:
        msg = f'❌ Erro ao listar arquivos em {remote_path}: {e}'
        logger.error(msg)
        return True, []


def list_remote_subdirectories(sftp, remote_path: str):
    """
    Lista apenas subdiretórios de um diretório remoto.

    Args:
        sftp (paramiko.SFTPClient): Cliente SFTP ativo
        remote_path (str): Caminho do diretório remoto

    Returns:
        tuple: (error: bool, subdirs: list)

    Exemplo:
        >>> error, subdirs = list_remote_subdirectories(sftp, '/home/ubuntu/dados')
        >>> if not error:
        >>>     for subdir in subdirs:
        >>>         print(subdir)
    """
    try:
        subdirs = []
        for entry in sftp.listdir_attr(remote_path):
            if stat.S_ISDIR(entry.st_mode):
                subdirs.append(entry.filename)

        logger.info(f'📂 Listados {len(subdirs)} subdiretórios em {remote_path}')
        return False, subdirs

    except FileNotFoundError:
        msg = f'❌ Diretório remoto não encontrado: {remote_path}'
        logger.error(msg)
        return True, []

    except Exception as e:
        msg = f'❌ Erro ao listar subdiretórios em {remote_path}: {e}'
        logger.error(msg)
        return True, []


def download_remote_file(sftp, remote_path: str, local_path: str) -> bool:
    """
    Baixa um arquivo remoto para o sistema local.

    Cria o diretório local automaticamente se não existir.

    Args:
        sftp (paramiko.SFTPClient): Cliente SFTP ativo
        remote_path (str): Caminho completo no servidor remoto
        local_path (str): Caminho onde salvar localmente

    Returns:
        bool: True se sucesso, False se falha

    Exemplo:
        >>> success = download_remote_file(
        >>>     sftp,
        >>>     '/home/ubuntu/dados/psi850.nc',
        >>>     '/tmp/psi850.nc'
        >>> )
        >>> if success:
        >>>     print("Download concluído!")
    """
    try:
        # Cria diretório local se não existir
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Obtém tamanho do arquivo remoto
        file_attr = sftp.stat(remote_path)
        size_mb = file_attr.st_size / (1024 * 1024)

        logger.info(f'⬇️  Baixando arquivo remoto ({size_mb:.2f} MB)...')
        logger.info(f'   Remoto: {remote_path}')
        logger.info(f'   Local:  {local_path}')

        # Realiza o download
        sftp.get(remote_path, local_path)

        logger.info('✅ Arquivo baixado com sucesso!')
        return True

    except FileNotFoundError as e:
        logger.error(f'❌ Arquivo remoto não encontrado: {remote_path}')
        logger.error(f'   Erro: {e}')
        return False

    except PermissionError as e:
        logger.error(f'❌ Sem permissão para escrever em: {local_path}')
        logger.error(f'   Erro: {e}')
        return False

    except Exception as e:
        logger.error(f'❌ Erro ao baixar arquivo {remote_path}: {e}')
        return False


def test_connection():
    """
    Testa a conexão SSH/SFTP com o servidor remoto.

    Função auxiliar para validar configurações antes de executar scripts.

    Returns:
        bool: True se conexão bem-sucedida, False caso contrário

    Exemplo:
        >>> if test_connection():
        >>>     print("Servidor acessível!")
    """
    logger.info('=' * 80)
    logger.info('🧪 TESTE DE CONEXÃO SSH/SFTP')
    logger.info('=' * 80)

    error, ssh_client = connect_server_with_key()

    if error:
        logger.error(f'❌ Falha no teste de conexão: {ssh_client}')
        return False

    try:
        # Testa canal SFTP
        sftp = ssh_client.open_sftp()
        logger.info('✅ Canal SFTP aberto com sucesso!')

        # Testa listagem de diretório remoto
        test_dir = settings.get('REMOTE_DIR_INPUT', '/home/ubuntu/meteorologia/Entrada')
        logger.info(f'🧪 Testando acesso ao diretório: {test_dir}')

        if is_remote_directory(sftp, test_dir):
            logger.info(f'✅ Diretório acessível: {test_dir}')

            # Lista alguns arquivos
            error, files = list_remote_files(sftp, test_dir)
            if not error and files:
                logger.info(f'✅ Encontrados {len(files)} arquivos no diretório')
                logger.info('   Primeiros 5 arquivos:')
                for file in files[:5]:
                    logger.info(f'   - {file}')
        else:
            logger.warning(f'⚠️  Diretório não acessível: {test_dir}')

        # Cleanup
        sftp.close()
        ssh_client.close()

        logger.info('=' * 80)
        logger.info('✅ TESTE DE CONEXÃO CONCLUÍDO COM SUCESSO!')
        logger.info('=' * 80)
        return True

    except Exception as e:
        logger.exception(f'❌ Erro durante teste de conexão: {e}')

        # Cleanup
        try:
            if 'sftp' in locals():
                sftp.close()
            if ssh_client:
                ssh_client.close()
        except:
            pass

        return False


# Script de teste standalone
if __name__ == '__main__':
    # Bibliotecas padrão
    import sys

    print('\n' + '=' * 80)
    print('🧪 TESTE DO MÓDULO DE CONEXÕES SSH/SFTP')
    print('=' * 80 + '\n')

    success = test_connection()

    if success:
        print('\n✅ Módulo connections.py funcionando corretamente!')
        sys.exit(0)
    else:
        print('\n❌ Falha no teste do módulo connections.py')
        print('   Verifique as configurações em app/settings/settings.toml')
        sys.exit(1)
