# FABRIK 3D - Plan de Refactorización 🚀

## 📋 Objetivos de la Refactorización

Refactorización la clase FABRIK_3D en un sistema **modular y mantenible** con múltiples componentes especializados.

## 🏗️ Arquitectura Propuesta

```
FABRIK/
├── fabrik_core/                        # [ ] Lógica del algoritmo
│   ├── ( fabrik_solver.py              # [ ] Algoritmo FABRIK puro )
│   ├── math_utils.py                   # [X] Utilidades matemáticas
│   ├── algorithms/                     # [ ] Algoritmos por implementar
│   │   ├ ( constraints.py              # [ ] Restricciones (Algorithm 2 & 3) )
│   │   ├── algorithm_4_conversion.py   # [ ] Position to Joint Angles
│   │   ├── algorithm_5_multi_target.py # [ ] Multi-Target FABRIK
│   │   ├── algorithm_6_orientation.py  # [ ] Orientation Control
│   │   └── algorithms_roadmap.py       # [ ] Hoja de ruta
│   └── __init__.py
├── visualization/                      # [ ] 👁️ Interfaz gráfica
│   ├── visualizer.py                   # [ ] Renderización 3D
│   ├── recorder.py                     # [X] Grabación de animaciones 
│   ├── controller.py                   # [ ] Mover mapeo de eventos y controles
│   └── __init__.py
├── demo_refactored_system.py           # [ ] Demostración
└── fabrik_paper_constrained_3d.py      # [ ] Código principal
```

## Changelog

## Refactorizar

- [ ] Estudiar y implementar rasgos de la estructura .yaml seguida por NVIDIA Isaac™ Lab.
- [ ] Modular clase Fabrik como un modulo para probar algoritmos genericos y reimplementar el módelo básico de Fabrik.

### Core

- [ ] Mejorar la clase TAD Robot
  - [ ] Mejorar TAD Link -> TAD Joint

### Fabrik_3D y Fabrik-R

- [ ] Bug entre la primera articulación y la base, las restricciones no se aplican de forma correcta. Permitiendo al vector entre la base y la primera articulación inclinarse y moverse de forma libre.

Necisides Fabrik-R: