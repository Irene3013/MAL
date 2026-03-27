import argparse
import pandas as pd
import os

def ordenar_csv(input_path, output_path):
    # Comprobar que el archivo existe
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    # Leer CSV
    df = pd.read_csv(input_path)

    # Comprobar que existe la columna 'image'
    if "image" not in df.columns:
        raise ValueError("El CSV no contiene una columna llamada 'image'")

    # Ordenar por la columna image
    df_sorted = df.sort_values(by="image")

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar CSV
    df_sorted.to_csv(output_path, index=False)

    print(f"CSV ordenado guardado en: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ordenar un CSV por la columna 'image'")
    parser.add_argument("version", help="Ruta al CSV de entrada")
    parser.add_argument("split", help="Ruta al CSV de entrada")
    #parser.add_argument("output", help="Ruta donde guardar el CSV ordenado")

    args = parser.parse_args()
    input = f"./data/raw/RelationsDataset/{args.version}/{args.version}_{args.split}.csv"
    output = f"./data/raw/RelationsDataset/out/{args.version}_{args.split}.csv"
    ordenar_csv(input, output)