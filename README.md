# ExtendedProfile (baseline A; estático) + opción B (rolling)

- Notebook principal: `notebooks/extended_profile.ipynb`
- Configuración: `configs/params.yml`
- Código fuente: `src/`

Flujo (baseline):
1) Cargar datos entre fechas (ES, 5m).
2) Inferir sesiones y seleccionar 4 previas.
3) Construir EP, calcular Score y Score* (crowd-aware).
4) Filtrar y clusterizar balizas.
5) Medir HUE, λ1, CL, densidad e índice de reactividad por sesión.
