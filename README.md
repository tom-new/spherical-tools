# spherical-tools

**Convert between geographic, spherical, and Cartesian coordinates.**
This is a lightweight Python library designed for geoscientists working with coordinates on the sphere.

---

## 📦 Installation

Install the latest development version:

```bash
pip install git+https://github.com/tom-new/spherical-tools.git
```

---

## 🚀 Quick Start

```python
import spherical_tools as st

# Convert from spherical (r, theta, phi) to Cartesian
# theta = azimuthal angle, phi = polar angle
x, y, z = st.sph2cart(1, 30, 60, degrees=True)

# convert from Cartesian to spherical
r, theta, phi = st.cart2sph(x, y, z, degrees=True)
```

By deafault, angles are in **radians** but input and output angles can be in degrees by setting `degrees=True`.

---

## 📚 Function Reference

- `sph2cart(r, theta, phi, degrees=False)`
- `cart2sph(x, y, z, degrees=False)`
