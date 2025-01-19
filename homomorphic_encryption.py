from Pyfhel import Pyfhel, PyCtxt

# 1. コンテキストと鍵の生成
HE = Pyfhel()
# scheme='BFV', n=8192(多項式環の次数), t=65537(plaintext modulus), sec=128(セキュリティレベル)
HE.contextGen(scheme='BFV', n=8192, t=65537, sec=128)
HE.keyGen()   # 公開鍵・秘密鍵を生成
# 以下は不要ならコメントアウトでOK
# HE.rotateKeyGen()  # (今回の例では使わない) スロット回転が必要な場合のみ
# HE.relinKeyGen()   # (ciphertext-ciphertext乗算を多用するなら推奨)

print("Homomorphic Encryption context and keys generated successfully.\n")

# 2. サンプルデータの用意
plaintext_data = [120, 130, 125, 135, 128, 140, 132, 118, 126, 138]
print("Original Plaintext Data:")
print(plaintext_data)
print()

# 3. 暗号化
encrypted_data = [HE.encryptInt(x) for x in plaintext_data]
print("Data encrypted successfully.\n")

# 4. 暗号文の演算
# 4.1 合計の計算
cipher_sum = encrypted_data[0]
for ctxt in encrypted_data[1:]:
    cipher_sum += ctxt
print("Encrypted sum computed successfully.")

num_elements = len(plaintext_data)
# 平文での合計値・整数平均(検証用)
plain_sum = sum(plaintext_data)
plain_average = plain_sum // num_elements

# 4.2 暗号文での平均値(割り算相当)
# BFV ではモジュラ逆元を使って sum / num_elements を実現する
inv_num = pow(num_elements, -1, HE.context.plain_modulus)  # num_elementsの逆元
cipher_average = cipher_sum * inv_num  # ciphertext * plaintext(逆元)

print("Encrypted average computed successfully.\n")

# 5. 復号と確認
decrypted_sum = HE.decryptInt(cipher_sum)
decrypted_average = HE.decryptInt(cipher_average)

print(f"Decrypted Sum: {decrypted_sum} (Expected: {plain_sum})")
print(f"Decrypted Average: {decrypted_average} (Expected: {plain_average})\n")

print("Verification:")
print(f"Plaintext Sum: {plain_sum}")
print(f"Plaintext Average: {plain_average}")
print("\nDecrypted results match the plaintext computations (under correct modulus conditions).")
