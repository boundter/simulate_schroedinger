# Makefile for simukate_schroedinger

# Replots the Scheme
plots/eigenfunction_eigenvalue.eps: code/eigenvalues.py
	./code/eigenvalues.py