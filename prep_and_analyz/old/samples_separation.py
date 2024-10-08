import os
import random
import shutil

# Zadej cestu ke složce
slozka = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing"

# Vytvoř nové složky pro větší a menší soubory, pokud ještě neexistují
vetsi_slozka = os.path.join(slozka, 'vetsi_cast')
mensi_slozka = os.path.join(slozka, 'mensi_cast')

os.makedirs(vetsi_slozka, exist_ok=True)
os.makedirs(mensi_slozka, exist_ok=True)

# Získání seznamu všech souborů ve složce
soubory = [f for f in os.listdir(slozka) if os.path.isfile(os.path.join(slozka, f))]

# Náhodně promíchej seznam souborů
random.shuffle(soubory)

# Urči bod rozdělení (např. 70 % pro větší složku a 30 % pro menší)
split_index = int(0.7 * len(soubory))

# Rozdělení na větší a menší část
vetsi_cast = soubory[:split_index]
mensi_cast = soubory[split_index:]

# Přesun souborů do složek
for soubor in vetsi_cast:
    originalni_cesta = os.path.join(slozka, soubor)
    nova_cesta = os.path.join(vetsi_slozka, soubor)
    shutil.move(originalni_cesta, nova_cesta)

for soubor in mensi_cast:
    originalni_cesta = os.path.join(slozka, soubor)
    nova_cesta = os.path.join(mensi_slozka, soubor)
    shutil.move(originalni_cesta, nova_cesta)

print(f"Počet souborů přesunutých do 'vetsi_cast': {len(vetsi_cast)}")
print(f"Počet souborů přesunutých do 'mensi_cast': {len(mensi_cast)}")
