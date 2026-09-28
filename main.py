"""
Practica Calificada 2 - Fisica Computacional
Metodos numericos para ecuaciones no lineales f(x)=0
"""
import numpy as np
import matplotlib.pyplot as plt

R = 8.314  # J/(mol*K)

# ==========================================================
# ACTIVIDAD 1: Van der Waals - Nitrogeno
# ==========================================================
print("="*60)
print("ACTIVIDAD 1: Nitrogeno (VdW)")
print("="*60)
P1, T1, b1 = 5.0e6, 273.15, 3.87e-5
v_ideal = R*T1/P1
print(f"v_ideal  = {v_ideal:.6e} m^3/mol")
print(f"b        = {b1:.6e} m^3/mol")
print(f"v0 > b?  {v_ideal > b1}")

# ==========================================================
# ACTIVIDAD 2: Lennard-Jones - Argon
# ==========================================================
print("\n" + "="*60)
print("ACTIVIDAD 2: Lennard-Jones (Ar)")
print("="*60)
def g(u): return 2.0 - u**6

u0, u1 = 0.90, 1.50
print(f"g({u0}) = {g(u0):.6f}")
print(f"g({u1}) = {g(u1):.6f}")
for k in range(2):
    ur = u0 - g(u0)*(u1-u0)/(g(u1)-g(u0))
    print(f"Iter {k+1}: u_r = {ur:.8f}, g(u_r) = {g(ur):.8f}")
    if g(u0)*g(ur) < 0:
        u1 = ur
    else:
        u0 = ur
sigma = 3.40  # Angstrom
u_exact = 2**(1/6)
print(f"u* exacto = {u_exact:.8f}")
print(f"r_e       = {u_exact*sigma:.6f} Angstrom")

# ==========================================================
# ACTIVIDAD 3: Convergencia cuadratica - CO2
# ==========================================================
print("\n" + "="*60)
print("ACTIVIDAD 3: Constante de error (CO2)")
print("="*60)
a3, b3 = 0.3640, 4.267e-5
P3, T3, v_star = 1.0e6, 300.0, 2.3811e-3
fpp = 6*P3*v_star - 2*(P3*b3 + R*T3)
print(f"f''(v*) = {fpp:.4f}")
print(f"M       = {abs(fpp):.4f}")

# ==========================================================
# ACTIVIDAD 4: Agua - Metodo de la Secante
# ==========================================================
print("\n" + "="*60)
print("ACTIVIDAD 4: Agua (H2O) - Secante")
print("="*60)
a4, b4 = 0.5536, 3.049e-5
P4, T4 = 2.63e6, 500.0

Tc = 8*a4/(27*R*b4)
print(f"T_c = {Tc:.2f} K | T = {T4} K -> subcritico: {T4<Tc}")

def f_vdw(v): return P4*v**3 - (P4*b4 + R*T4)*v**2 + a4*v - a4*b4

def secante(f, v0, v1, tol=1e-8, maxit=100):
    for k in range(1, maxit+1):
        f0, f1 = f(v0), f(v1)
        if abs(f1-f0) < 1e-15: break
        v2 = v1 - f1*(v1-v0)/(f1-f0)
        if abs(v2-v1) < tol or abs(f(v2)) < tol:
            return v2, k, abs(f(v2))
        v0, v1 = v1, v2
    return v2, maxit, abs(f(v2))

vl, kl, rl = secante(f_vdw, 1.1*b4, 1.2*b4)
v0g = R*T4/P4
vg, kg, rg = secante(f_vdw, v0g, 0.9*v0g)
print(f"v_l = {vl:.6e} m^3/mol | it={kl} | |f|={rl:.2e}")
print(f"v_g = {vg:.6e} m^3/mol | it={kg} | |f|={rg:.2e}")

# Grafico de la isoterma
v = np.linspace(0.9*b4, 3e-3, 2000)
plt.figure(figsize=(8,5))
plt.plot(v, f_vdw(v), 'b-', lw=1.5)
plt.axhline(0, color='k', lw=0.8)
plt.scatter([vl, vg], [0,0], color='red', zorder=5,
            label=f"$v_l$={vl:.2e}, $v_g$={vg:.2e}")
plt.xlabel(r"$v$ [m$^3$/mol]"); plt.ylabel(r"$f(v)$")
plt.title("Isoterma VdW H$_2$O a T=500 K, P=2.63 MPa")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("isoterma_agua.png", dpi=150)
plt.show()
