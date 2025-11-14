from passlib.context import CryptContext

def gerar_hash(senha: str):
    """Gera e exibe um hash bcrypt seguro para a senha informada."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hash_gerado = pwd_context.hash(senha)
    print("\n--- HASH GERADO COM SUCESSO ---")
    print(hash_gerado)
    print("--------------------------------")
    print("Copie o hash acima e substitua no seu USERS_DB.\n")

if __name__ == "__main__":
    senha = input("Digite a senha que deseja criptografar: ").strip()
    if not senha:
        print("❌ Senha inválida! Tente novamente.")
    else:
        gerar_hash(senha)
