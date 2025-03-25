import requests

# Paso 1: Obtener lobbies públicos
public_lobbies_url = "https://openfront.io/api/public_lobbies"
response = requests.get(public_lobbies_url)

if response.status_code == 200:
    data = response.json()
    lobbies = data.get('lobbies', [])
    
    if lobbies:
        # Selecciona el primer lobby disponible
        lobby = lobbies[0]
        game_id = lobby['gameID']
        print(f"Unirse al Lobby ID: {game_id}")

        # Lista de posibles endpoints para unirse al lobby
        join_urls = [
            "https://openfront.io/api/join_lobby",
            "https://openfront.io/api/join",
            "https://openfront.io/api/lobby/join",
            "https://openfront.io/api/join_game",
            "https://openfront.io/api/enter_lobby"  # Endpoint alternativo
        ]

        username = "TuNombre"  # Cambia "TuNombre" por el nombre que deseas usar

        # Datos para unirse al lobby
        data = {
            "gameID": game_id,
            "username": username,
            # Agrega otros parámetros necesarios aquí
        }

        # Probar cada endpoint
        for join_url in join_urls:
            print(f"Intentando unirse al lobby usando el endpoint: {join_url}")
            join_response = requests.post(join_url, json=data)

            # Imprimir la respuesta del servidor
            print(f"Respuesta al intentar unirse al lobby: {join_response.status_code}, {join_response.text}")

    else:
        print("No hay lobbies disponibles para unirse.")
else:
    print("Error al obtener lobbies públicos:", response.status_code, response.text)