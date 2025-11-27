Construir imagen
docker build -t weather-prediction:latest .

Ejecutar
docker run --rm -v "%cd%:/app" weather-prediction:latest