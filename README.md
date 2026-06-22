# 🎵 Spotlight Mix

Optimizador de Playlists para Spotify Automix en Python.

Reordena playlists de Spotify (2.000+ canciones) basándose en tags de un CSV
(BPM, Camelot, Energy, etc.) para maximizar la calidad de las transiciones
Automix de Spotify.

## 📦 Instalación

```bash
pip install -r requirements.txt
```

## 🔑 Configuración de Spotify Developer

1. Ve a https://developer.spotify.com/dashboard
2. Crea una nueva aplicación
3. Anota tu **Client ID** y **Client Secret**
4. Añade `http://127.0.0.1:8080` como Redirect URI
5. Solicita los scopes: `playlist-read-private`, `playlist-read-collaborative`,
   `playlist-modify-public`, `playlist-modify-private`

## 📄 Formato CSV esperado

El CSV debe contener las siguientes columnas (el header es flexible, se
normalizan sinónimos automáticamente):

```
#,Song,Artist,BPM,Camelot,Energy,Added At,Duration,Popularity,Genres,Album,
Album Date,Dance,Acoustic,Instrumental,Valence,Speech,Live,Loud (Db),Key,
Time Signature,Spotify Track Id,ISRC,Explicit
```

Herramientas recomendadas para exportar tu playlist:
- [Exportify](https://github.com/watsonbox/exportify)
- [Skifta](https://skifta.com)

## 🚀 Uso

```bash
python spotlight_mix.py \
  --csv mi_playlist.csv \
  --playlist-id 37i9dQZF1... \
  --client-id TU_CLIENT_ID \
  --client-secret TU_CLIENT_SECRET
```

El script:
1. Abre tu navegador para autenticarte con Spotify (OAuth)
2. Lee y valida el CSV
3. Deduplica canciones (fuzzy matching + ISRC)
4. Construye matriz de distancias armónicas
5. Ejecuta algoritmo greedy + 2-opt
6. Crea una nueva playlist reordenada en tu cuenta
7. Activa Automix en Spotify → reproduce → disfruta transiciones limpias

## ⚙️ Pesos configurables

Puedes ajustar la importancia de cada feature:

```bash
python spotlight_mix.py \
  --csv mi_playlist.csv \
  --playlist-id 37i9dQZF1... \
  --client-id X --client-secret Y \
  --weights bpm=0.40 key=0.20 energy=0.15 valence=0.10 dance=0.10 acoustic=0.05
```

## 🧪 Tests

```bash
pytest tests/ --cov=spotlight_mix --cov-report=term-missing
```

## 📜 Licencia

MIT
