import hashlib

MD5_USER_PASSWORD = "@Cpsbrasil1278"
MD5_PASSWORD_HASH = hashlib.md5(MD5_USER_PASSWORD.encode('utf-8')).hexdigest()
print(MD5_PASSWORD_HASH)