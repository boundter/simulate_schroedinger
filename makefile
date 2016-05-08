# Makefile for simulate_schroedinger

# Replots the Scheme
plots/eigenfunction_eigenvalue.eps, plots/animation_1.mp4: code/main.py
	./code/eigenvalues.py